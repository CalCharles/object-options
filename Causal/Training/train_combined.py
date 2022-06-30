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
from Causal.Utils.get_error import error_types
from Network.network_utils import pytorch_model, run_optimizer
from Record.file_management import create_directory

def _train_combined_interaction(full_model, args, rollouts, weights, proximal, inter_loss, interaction_optimizer):
    # resamples because the interaction weights are different from the normal weights, and get the weight count for this
    batch, idxes = rollouts.sample(args.train.batch_size // 2, weights=weights)
    batch_uni, idxes_uni = rollouts.sample(args.train.batch_size // 2)
    idxes = idxes.tolist() + idxes_uni.tolist()
    batch = batch.cat([batch, batch_uni])
    weight_count = np.sum(weights[idxes])

    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_params, passive_params, interaction_likelihood, target, active_dist, passive_dist, active_log_probs, passive_log_probs = full_model.likelihoods(batch)     

    # done flags
    done_flags = pytorch_model.wrap(1-batch.done, cuda = full_model.iscuda)

    # combine likelihoods to get a single likelihood for computing binaries TODO: a per-element binary?
    active_likelihood = - active_log_probs.sum(dim=-1).unsqueeze(-1) * done_flags
    passive_likelihood = - passive_log_probs.sum(dim=-1).unsqueeze(-1) * done_flags

    # get the interaction binaries
    interaction_binaries = full_model.test.compute_binary(active_likelihood, passive_likelihood)


    # for proximity, don't allow interactions that are not also proximal
    if proximal is not None: interaction_binaries = interaction_binaries * pytorch_model.wrap(proximal[idxes], cuda=full_model.iscuda)
    
    # loss and optimizer
    interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
    run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss)
    return interaction_loss, interaction_likelihood, interaction_binaries, weight_count

def train_combined(full_model, rollouts, test_rollout, args,
    trace, active_weights, interaction_weights, proximal,
    active_optimizer, passive_optimizer, interaction_optimizer):    

    passive_logger = forward_logger("passive", args.inter.active.active_log_interval, full_model)
    logger = forward_logger("active", args.inter.active.active_log_interval, full_model)
    inter_logger = interaction_logger("interaction", args.inter.active.active_log_interval, full_model)
    # initialize loss function
    inter_loss = nn.BCELoss()

    # initialize interaction schedule, computes the weight to allow the active model to ignore certain values
    interaction_schedule = (lambda i: np.power(0.5, (i/args.inter.active.interaction_schedule))) if args.inter.active.interaction_schedule > 0 else (lambda i: 0.5)
    inline_iter_schedule = lambda i: min(1, args.inter.active.inline_iters[0],
                                         args.inter.active.inline_iters[1] * np.power(2, (i/args.inter.active.inline_iters[2])))
    inline_iters = inline_iter_schedule(0)

    # initialize weighting schedules, by which the sampling weights change (shrink) over training
    _,_,awl,aws = args.inter.active.weighting 
    iwl, iws = args.inter.active.interaction_weighting
    active_weighting_schedule = (lambda i: awl * np.power(0.5, (i/aws))) if awl > 0 else (lambda i: 1)
    interaction_weighting_schedule = (lambda i: iwl * np.power(0.5, (i/iws))) if iws > 0 else (lambda i: 1)

    print_errors(full_model, rollouts, error_types=[error_types.PASSIVE_RAW, error_types.PASSIVE, error_types.TRACE, error_types.DONE])

    for i in range(args.train.num_iters):
        # get data, weighting by active weights (generally more likely to sample high "value" states)
        batch, idxes = rollouts.sample(args.train.batch_size, weights=active_weights)
        weight_rate = np.sum(active_weights[idxes]) / len(idxes)

        # run the networks and get both the active and passive outputs (passive for interaction binaries)
        active_params, passive_params, interaction_likelihood,\
            target, active_dist, passive_dist, \
            active_log_probs, passive_log_probs = full_model.likelihoods(batch)

        # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
        active_nlikelihood = - active_log_probs.sum(dim=-1).unsqueeze(-1)
        passive_nlikelihood = - passive_log_probs.sum(dim=-1).unsqueeze(-1)

        # weighted values against actual likelihood TODO: 0 is used to ignore values, does it?...
        detached_interaction_likelihood = interaction_likelihood.clone().detach()
        inter_weighted_nlikelihood = active_nlikelihood * detached_interaction_likelihood

        # assign done flags
        done_flags = pytorch_model.wrap(1-batch.done, cuda = full_model.iscuda)

        # reduce with mean to a single value for the batch
        active_mean_nlikelihood, passive_mean_nlikelihood, inter_mean_nlikelihood = active_nlikelihood.mean(dim=0).squeeze(), passive_nlikelihood.mean(dim=0).squeeze(), inter_weighted_nlikelihood.squeeze() / (detached_interaction_likelihood.sum() + 1e-6)

        # train a combined loss to minimize the (negative) active likelihood without interaction weighting, and the interaction regulairized values (ignoring dones)
        # TODO: we used a combined interaction of binaries, proximal high error and interaction before, but with resampling it isn't clear this is necessary
        loss = (active_nlikelihood * interaction_schedule(i) + inter_weighted_nlikelihood * (1-interaction_schedule(i))) * done_flags
        run_optimizer(active_optimizer, full_model.active_model, loss)
        
        log_interval = logger.log(i, loss, active_nlikelihood * done_flags, inter_weighted_nlikelihood * done_flags, 
                    active_log_probs * done_flags, trace[idxes], weight_rate,
                    active_params, target, interaction_likelihood, full_model)

        # training the passive model with the weighted states, which is dangerous and not advisable
        if args.inter.active.intrain_passive: run_optimizer(passive_optimizer, full_model.passive_model, passive_nlikelihood)

        # run the interaction model training if the interaction model is not already trained
        if args.inter.interaction.interaction_pretrain <= 0:
            for ii in range(int(inline_iters)):
                interaction_loss, interaction_calc_likelihood,\
                 interaction_binaries, weight_count = _train_combined_interaction(full_model, args, rollouts,
                                                                     interaction_weights, proximal,
                                                                     inter_loss, interaction_optimizer)
        
                # reweight only when logging TODO: probably should not link  these two
                inter_logger.log(i, interaction_loss, interaction_calc_likelihood, interaction_binaries, batch.done, weight_count,
                    trace=None if trace is None else trace[idxes], no_print=ii != 0)
        
        if i % args.inter.active.active_log_interval == 0:
            # change the lambdas for reweighting, and generate new sampling weights
            inline_iters = inline_iter_schedule(i)
            active_weighting_lambda = active_weighting_schedule(i)
            active_weights = get_weights(active_weighting_lambda, rollouts.weight_binary)
            inter_weighting_lambda = interaction_weighting_schedule(i)
            interaction_weights = get_weights(inter_weighting_lambda, rollouts.weight_binary)
            print(interaction_schedule(i))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.inter_state, form="inter"), full_model.norm.reverse(batch.next_target),
            print("inter tar", np.concatenate([full_model.norm.reverse(batch.next_target), full_model.norm.reverse(pytorch_model.unwrap(active_params[0]))], axis=-1))
            print(np.concatenate([batch.next_target,pytorch_model.unwrap(active_params[0]),
                # full_model.norm.reverse(pytorch_model.unwrap(active_params[0])), full_model.norm.reverse(pytorch_model.unwrap(active_params[1])),
                # pytorch_model.unwrap(active_params[0]), pytorch_model.unwrap(active_params[1]),
                trace[idxes], np.expand_dims(active_weights[idxes], 1), pytorch_model.unwrap(active_nlikelihood), pytorch_model.unwrap(passive_nlikelihood), pytorch_model.unwrap(interaction_likelihood)], axis=-1))
