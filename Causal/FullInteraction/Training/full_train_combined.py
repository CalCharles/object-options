# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Training.loggers.forward_logger import forward_logger
from Causal.Training.loggers.interaction_logger import interaction_logger
from Causal.Training.loggers.logging import print_errors
from Causal.Utils.weighting import get_weights
from Causal.Utils.get_error import error_types, get_error
from Network.network_utils import pytorch_model, run_optimizer
from Causal.Utils.instance_handling import compute_likelihood

def evaluate_active_interaction(full_model, args, active_params, interaction_likelihood, interaction_mask, active_log_probs, done_flags, proximity):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    mask_loss = interaction_likelihood.sum(-1).mean()
    full_loss = active_nlikelihood + mask_loss * args.full_inter.lasso_lambda
    return full_loss

# def evaluate_active_forward(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity)

def _train_combined_interaction(full_model, args, rollouts, object_rollout, weights, inter_loss, interaction_optimizer, normalize=False):
    # resamples because the interaction weights are different from the normal weights, and get the weight count for this
    full_batch, idxes = rollouts.sample(args.train.batch_size // 2, weights=weights)
    batch_uni, idxes_uni = rollouts.sample(args.train.batch_size // 2)
    idxes = idxes.tolist() + idxes_uni.tolist()
    full_batch = full_batch.cat([full_batch, batch_uni])
    batch = object_rollout[idxes]
    batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
    batch.inter_state = full_batch.obs

    # a statistic on weighting
    weight_count = np.sum(weights[idxes])
    # print("running inline iters")
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs = full_model.likelihoods(batch, normalize=normalize, mixed=args.full_inter.mixed_interaction)

    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine the cost function (extend possible interaction losses here)
    interaction_loss = evaluate_active_interaction(full_model, args, active_soft_params, interaction_likelihood, hard_interaction_mask, active_soft_log_probs, done_flags, batch.proximity)
    
    # loss and optimizer
    run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss)
    return idxes, interaction_loss, interaction_likelihood, hard_interaction_mask, weight_count, done_flags

def train_combined(full_model, rollouts, object_rollouts, test_rollout, test_object_rollout,
    args, active_weights, interaction_weights, proximal,
    active_optimizer, passive_optimizer, interaction_optimizer,
    normalize=False):    

    passive_logger = forward_logger("passive", args.inter.active.active_log_interval, full_model)
    logger = forward_logger("active", args.inter.active.active_log_interval, full_model)
    inter_logger = interaction_logger("interaction", args.inter.active.active_log_interval, full_model)
    # initialize loss function
    inter_loss = nn.BCELoss()

    # initialize interaction schedule, computes the weight to allow the active model to ignore certain values
    interaction_schedule = (lambda i: np.power(0.5, (i/args.inter.active.interaction_schedule))) if args.inter.active.interaction_schedule > 0 else (lambda i: 0.5)
    inline_iter_schedule = lambda i: max(0, min(args.inter.active.inline_iters[0],
                                         np.power(2, (i/args.inter.active.inline_iters[2])) - 1) if args.inter.active.inline_iters[2] > 0 else args.inter.active.inline_iters[0]) 
    inline_iters = inline_iter_schedule(0)

    # initialize weighting schedules, by which the sampling weights change (shrink) over training
    _,_,awl,aws = args.inter.active.weighting 
    iwl, iws = args.inter.active.interaction_weighting
    print("awliwl", awl, iwl)
    active_weighting_schedule = (lambda i: awl * np.power(0.5, (i/aws))) if aws > 0 else (lambda i: awl)
    interaction_weighting_schedule = (lambda i: iwl * np.power(0.5, (i/iws))) if iws > 0 else (lambda i: iwl)

    print_errors(full_model, rollouts, object_rollouts, error_types=[error_types.ACTIVE_RAW, error_types.ACTIVE, error_types.TRACE, error_types.DONE], prenormalize=normalize)
    start = time.time()
    for i in range(args.train.num_iters):
        # get data, weighting by active weights (generally more likely to sample high "value" states)
        # batch, idxes = object_rollouts.sample(args.train.batch_size, weights=active_weights)
        # full_batch = rollouts[idxes]

        full_batch, idxes = rollouts.sample(args.train.batch_size, weights=active_weights)
        batch = object_rollouts[idxes]


        batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
        batch.inter_state = full_batch.obs
        weight_rate = np.sum(active_weights[idxes]) / len(idxes)
        # run the networks and get both the active and passive outputs (passive for interaction binaries)
        active_hard_params, active_soft_params, active_full, passive_params, \
            interaction_likelihood, soft_interaction_mask, hard_interaction_mask,\
            target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
            active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs = full_model.likelihoods(batch, normalize=normalize)

        print(full_model.name, active_full_log_probs[0], active_log_probs[0], interaction_schedule(i), hard_interaction_mask[0] )
        # assign done flags
        done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

        # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
        active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_hard_log_probs, done_flags=done_flags, is_full=True)
        active_full_nlikelihood = compute_likelihood(full_model, args.train.batch_size, -active_full_log_probs, done_flags=done_flags, is_full=True)
        passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, is_full=True)

        loss = (active_nlikelihood * (1-interaction_schedule(i)) + active_full_nlikelihood * interaction_schedule(i))
        run_optimizer(active_optimizer, full_model.active_model, loss)

        # logging will probably break from the changing of the meaning of interactions
        # TODO: ccreate a good logger for the full dataset
        # print(active_nlikelihood.shape, active_full_nlikelihood.shape, active_log_probs.shape, active_params[0].shape, np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1).shape, batch.trace.shape, np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1).shape)
        single_trace = np.expand_dims(np.sum(np.sum(batch.trace - 1, axis=-1), axis=-1), axis=-1) if len(batch.trace.shape) == 3 else np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1)
        single_trace[single_trace > 1] = 1
        log_interval = logger.log(i, loss, active_nlikelihood * done_flags, active_full_nlikelihood * done_flags, 
                    active_hard_log_probs * done_flags, single_trace, weight_rate, full_batch.done,
                    (active_hard_params[0] * done_flags, active_hard_params[1] * done_flags), target * done_flags, np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1), full_model)

        # training the passive model with the weighted states, which is dangerous and not advisable
        if args.inter.active.intrain_passive: run_optimizer(passive_optimizer, full_model.passive_model, passive_nlikelihood)

        # run the interaction model training if the interaction model is not already trained
        # print("interaction pretrain", args.inter.interaction.interaction_pretrain)

        if args.inter.interaction.interaction_pretrain <= 0:
            for ii in range(int(inline_iters)):
                inter_idxes, interaction_loss, interaction_calc_likelihood,\
                     interaction_binaries, weight_count, inter_done_flags = _train_combined_interaction(full_model, args, rollouts, object_rollouts,
                                                                         interaction_weights, inter_loss, interaction_optimizer, normalize=normalize)
                single_trace = None
                if object_rollouts.trace is not None:
                    single_trace = np.expand_dims(np.sum(np.sum(object_rollouts.trace[inter_idxes] - 1, axis=-1), axis=-1), axis=-1) if len(object_rollouts.trace[inter_idxes].shape) == 3 else np.expand_dims(np.sum(object_rollouts.trace[inter_idxes] - 1, axis=-1), axis=-1)
                    single_trace[single_trace > 1] = 1
                # reweight only when logging TODO: probably should not link  these two
                inter_logger.log(i, interaction_loss, interaction_calc_likelihood, interaction_binaries, pytorch_model.unwrap(inter_done_flags), weight_count,
                    trace=None if single_trace is None else single_trace, no_print=ii != 0)
        
        if i % args.inter.active.active_log_interval == 0:
            print(i, "speed", (args.inter.active.active_log_interval * i) / (time.time() - start))
            # change the lambdas for reweighting, and generate new sampling weights
            # if int(inline_iters) > 0:
            #     print_errors(full_model, rollouts[inter_idxes[90:]], error_types=[error_types.ACTIVE_RAW, error_types.ACTIVE, error_types.PASSIVE_LIKELIHOOD, error_types.ACTIVE_LIKELIHOOD, error_types.TRACE, error_types.INTERACTION, error_types.INTERACTION_BINARIES, error_types.PROXIMITY, error_types.DONE], prenormalize=normalize)
            inline_iters = inline_iter_schedule(i)
            active_weighting_lambda = active_weighting_schedule(i)
            print(active_weighting_lambda)
            active_weights = get_weights(active_weighting_lambda, object_rollouts.weight_binary[:len(rollouts)].squeeze())
            print(active_weights)
            inter_weighting_lambda = interaction_weighting_schedule(i)
            mask_binary = (np.sum(np.round(full_model.apply_mask(get_error(full_model, rollouts, object_rollout=object_rollouts, error_type = error_types.INTERACTION_RAW))), axis=-1) > 1).astype(int)
            interaction_weights = get_weights(inter_weighting_lambda, object_rollouts.weight_binary[:len(rollouts)].squeeze() + mask_binary.squeeze())
            print("inline_iters", inline_iters)
                # print(trace[inter_idxes], active_weights[inter_idxes])
            # print(full_model.norm.reverse(rollouts.target[48780:48800]), full_model.norm.reverse(rollouts.next_target[48780:48800]))
            # print(interaction_schedule(i))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.inter_state, form="inter"), full_model.norm.reverse(batch.target_diff, form="dyn"), batch.true_done], axis=-1))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.target_diff, form="dyn"), full_model.norm.reverse(pytorch_model.unwrap(active_params[0]), form="dyn"), batch.true_done], axis=-1))
            # print(np.concatenate([batch.next_target,pytorch_model.unwrap(active_params[0]),
            #     # full_model.norm.reverse(pytorch_model.unwrap(active_params[0])), full_model.norm.reverse(pytorch_model.unwrap(active_params[1])),
            #     # pytorch_model.unwrap(active_params[0]), pytorch_model.unwrap(active_params[1]),
            #     pytorch_model.unwrap(active_nlikelihood), pytorch_model.unwrap(passive_nlikelihood), pytorch_model.unwrap(interaction_likelihood), trace[idxes], np.expand_dims(active_weights[idxes], 1)], axis=-1))
