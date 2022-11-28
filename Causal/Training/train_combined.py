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
from Causal.Training.test_full import test_full
from Causal.Utils.weighting import get_weights
from Causal.Utils.get_error import error_types, get_error
from Network.network_utils import pytorch_model, run_optimizer
from Causal.Utils.instance_handling import compute_likelihood

def _train_combined_interaction(full_model, args, rollouts, weights, proximal_inst, inter_loss, interaction_optimizer, normalize=False):
    # resamples because the interaction weights are different from the normal weights, and get the weight count for this
    batch, idxes = rollouts.sample(args.train.batch_size // 2, weights=weights)
    batch_uni, idxes_uni = rollouts.sample(args.train.batch_size // 2)
    idxes = idxes.tolist() + idxes_uni.tolist()
    batch = batch.cat([batch, batch_uni])
    weight_count = np.sum(weights[idxes])

    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_params, passive_params, interaction_likelihood, target, active_dist, passive_dist, active_log_probs, passive_log_probs = full_model.likelihoods(batch, normalize=normalize)     

    # done flags
    done_flags = pytorch_model.wrap(1-batch.true_done, cuda = full_model.iscuda)

    # combine likelihoods to get a single likelihood for computing binaries TODO: a per-element binary?
    active_likelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags, reduced=False)
    passive_likelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False)

    # get the interaction binaries
    interaction_binaries = full_model.test.compute_binary(active_likelihood, passive_likelihood)

    # for proximity, don't allow interactions that are not also proximal
    if proximal_inst is not None: interaction_binaries = interaction_binaries * pytorch_model.wrap(proximal_inst[idxes], cuda=full_model.iscuda)
    # print(interaction_binaries[0], interaction_likelihood[0], proximal_inst[idxes][0])
    
    # loss and optimizer
    interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
    run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss)
    return idxes, interaction_loss, interaction_likelihood, interaction_binaries, weight_count, done_flags

def train_combined(full_model, rollouts, test_rollout, args,
    trace, active_weights, interaction_weights, proximal, proximal_inst,
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

    print_errors(full_model, rollouts, error_types=[error_types.ACTIVE_RAW, error_types.ACTIVE, error_types.TRACE, error_types.DONE], prenormalize=normalize)

    for i in range(args.train.num_iters):
        # get data, weighting by active weights (generally more likely to sample high "value" states)
        batch, idxes = rollouts.sample(args.train.batch_size, weights=active_weights)
        weight_rate = np.sum(active_weights[idxes]) / len(idxes)

        # run the networks and get both the active and passive outputs (passive for interaction binaries)
        active_params, passive_params, interaction_likelihood,\
            target, active_dist, passive_dist, \
            active_log_probs, passive_log_probs = full_model.likelihoods(batch, normalize=normalize)

        # assign done flags
        done_flags = pytorch_model.wrap(1-batch.true_done, cuda = full_model.iscuda)

        # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
        active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags)
        passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags)

        # weighted values against actual likelihood TODO: 0 is used to ignore values, does it?...
        detached_interaction_likelihood = interaction_likelihood.clone().detach()
        inter_weighted_nlikelihood = active_nlikelihood * detached_interaction_likelihood

        # reduce with mean to a single value for the batch
        # active_mean_nlikelihood, passive_mean_nlikelihood, inter_mean_nlikelihood = active_nlikelihood.mean(dim=0).squeeze(), passive_nlikelihood.mean(dim=0).squeeze(), inter_weighted_nlikelihood.squeeze() / (detached_interaction_likelihood.sum() + 1e-6)

        # print(active_nlikelihood[:10], passive_nlikelihood[:10], active_weights[idxes][:10].squeeze())
        # train a combined loss to minimize the (negative) active likelihood without interaction weighting, and the interaction regulairized values (ignoring dones)
        # TODO: we used a combined interaction of binaries, proximal high error and interaction before, but with resampling it isn't clear this is necessary
        loss = (active_nlikelihood * interaction_schedule(i) + inter_weighted_nlikelihood * (1-interaction_schedule(i)))
        run_optimizer(active_optimizer, full_model.active_model, loss)

        # print(torch.cat([pytorch_model.wrap(full_model.norm.reverse(batch.inter_state, form="inter"), cuda=full_model.iscuda),active_params[0], target, active_nlikelihood * done_flags, done_flags], axis=-1)[:20])
        log_interval = logger.log(i, loss, active_nlikelihood * done_flags, inter_weighted_nlikelihood * done_flags, 
                    active_log_probs * done_flags, trace[idxes], weight_rate, batch.true_done,
                    (active_params[0] * done_flags, active_params[1] * done_flags), target * done_flags, interaction_likelihood, full_model)

        # training the passive model with the weighted states, which is dangerous and not advisable
        if args.inter.active.intrain_passive: run_optimizer(passive_optimizer, full_model.passive_model, passive_nlikelihood)

        # run the interaction model training if the interaction model is not already trained
        if args.inter.interaction.interaction_pretrain <= 0:
            for ii in range(int(inline_iters)):
                inter_idxes, interaction_loss, interaction_calc_likelihood,\
                     interaction_binaries, weight_count, inter_done_flags = _train_combined_interaction(full_model, args, rollouts,
                                                                         interaction_weights, proximal_inst,
                                                                         inter_loss, interaction_optimizer, normalize=normalize)
                
                
                # reweight only when logging TODO: probably should not link  these two
                inter_logger.log(i, interaction_loss, interaction_calc_likelihood, interaction_binaries, pytorch_model.unwrap(inter_done_flags), weight_count,
                    trace=None if trace is None else trace[inter_idxes], no_print=ii != 0)
        
        if i % args.inter.active.active_log_interval == 0:
            # change the lambdas for reweighting, and generate new sampling weights
            # if int(inline_iters) > 0:
            #     print_errors(full_model, rollouts[inter_idxes[90:]], error_types=[error_types.ACTIVE_RAW, error_types.ACTIVE, error_types.PASSIVE_LIKELIHOOD, error_types.ACTIVE_LIKELIHOOD, error_types.TRACE, error_types.INTERACTION, error_types.INTERACTION_BINARIES, error_types.PROXIMITY, error_types.DONE], prenormalize=normalize)
            inline_iters = inline_iter_schedule(i)
            active_weighting_lambda = active_weighting_schedule(i)
            print(active_weighting_lambda)
            active_weights = get_weights(active_weighting_lambda, rollouts.weight_binary[:len(rollouts)].squeeze())
            inter_weighting_lambda = interaction_weighting_schedule(i)
            error_binary = np.abs(get_error(full_model, rollouts, error_type = error_types.INTERACTION_BINARIES) - full_model.test(get_error(full_model, rollouts, error_type = error_types.INTERACTION_RAW)).astype(int))
            interaction_weights = get_weights(inter_weighting_lambda, rollouts.weight_binary[:len(rollouts)].squeeze() + error_binary.squeeze() * args.inter.active.error_binary_upweight)

            print("inline_iters", inline_iters)
            if i % (args.inter.active.active_log_interval * 10) == 0:
                test_full(full_model, test_rollout, args, args.object_names, None)
                # print(trace[inter_idxes], active_weights[inter_idxes])
            # print(full_model.norm.reverse(rollouts.target[48780:48800]), full_model.norm.reverse(rollouts.next_target[48780:48800]))
            # print(interaction_schedule(i))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.inter_state, form="inter"), full_model.norm.reverse(batch.target_diff, form="dyn"), batch.true_done], axis=-1))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.target_diff, form="dyn"), full_model.norm.reverse(pytorch_model.unwrap(active_params[0]), form="dyn"), batch.true_done], axis=-1))
            # print(np.concatenate([batch.next_target,pytorch_model.unwrap(active_params[0]),
            #     # full_model.norm.reverse(pytorch_model.unwrap(active_params[0])), full_model.norm.reverse(pytorch_model.unwrap(active_params[1])),
            #     # pytorch_model.unwrap(active_params[0]), pytorch_model.unwrap(active_params[1]),
            #     pytorch_model.unwrap(active_nlikelihood), pytorch_model.unwrap(passive_nlikelihood), pytorch_model.unwrap(interaction_likelihood), trace[idxes], np.expand_dims(active_weights[idxes], 1)], axis=-1))
