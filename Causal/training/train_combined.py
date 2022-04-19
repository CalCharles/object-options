# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from file_management import save_to_pickle, load_from_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_weights, get_targets
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import assess_losses, get_interaction_vals
from DistributionalModels.InteractionModels.InteractionTraining.assessment_functions import assess_error
from Rollouts.rollouts import ObjDict, merge_rollouts

def _train_combined_interaction(full_model, train_args, rollouts, weights, inter_loss, interaction_optimizer):
    # resamples because the interaction weights are different from the normal weights
    batch, idxes = rollouts.sample(train_args.batch_size, weights=weights)

    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_params, passive_params, interaction_likelihood, target, active_dist, passive_dist, active_log_probs, passive_log_probs = full_model.likelihoods(batch)     

    # combine likelihoods to get a single likelihood for computing binaries TODO: a per-element binary?
    active_likelihood = - active_log_probs.sum(dim=-1).unsqueeze(-1)
    passive_likelihood = - active_log_probs.sum(dim=-1).unsqueeze(-1)

    # get the interaction binaries
    interaction_binaries = full_model.test.compute_binary(active_likelihood, passive_likelihood)

    # for proximity, don't allow interactions that are not also proximal
    if proximal is not None: interaction_binaries *= pytorch_model.wrap(proximal[idxes], cuda=full_model.iscuda).unsqueeze(1)
    
    # loss and optimizer
    interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
    run_optimizer(interaction_optimizer, full_model.interaction_model, interaction_loss)
    return interaction_loss, interaction_likelihood, interaction_binaries

def train_combined(full_model, rollouts, test_rollout, train_args,
    trace, active_weights, interaction_weights,
    active_optimizer, passive_optimizer, interaction_optimizer):    
    
    # initialize loss function
    inter_loss = nn.BCELoss()

    # initialize interaction schedule, computes the weight to allow the active model to ignore certain values
    interaction_schedule = lambda i: np.power(0.5, (i/train_args.interaction_schedule))
    inline_iter_schedule = lambda i: max(train_args.inline_iters[2],
                                         train_args.inline_iters[0] * np.power(2, (i/train_args.inline_iters[1])))

    # initialize weighting schedules, by which the sampling weights change (shrink) over training
    active_weighting_schedule = lambda i: train_args.active_weight_lambda * np.power(0.5, (i/train_args.active_weight_schedule))
    interaction_weighting_schedule = lambda i: train_args.interaction_weight_lambda * np.power(0.5, (i/train_args.interaction_weight_schedule))

    for i in range(train_args.num_iters):
        # get data, weighting by active weights (generally more likely to sample high "value" states)
        batch, idxes = rollouts.sample(train_args.batch_size, weights=active_weights)

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
        done_flags = 1-batchvals.values.done

        # reduce with mean to a single value for the batch
        active_mean_nlikelihood, passive_mean_nlikelihood, inter_mean_nlikelihood = active_nlikelihood.mean(dim=0).squeeze(), passive_nlikelihood.mean(dim=0).squeeze(), inter_weighted_nlikelihood.squeeze() / (detached_interaction_likelihood.sum() + 1e-6)
        
        # train a combined loss to minimize the (negative) active likelihood without interaction weighting, and the interaction regulairized values (ignoring dones)
        # TODO: we used a combined interaction of binaries, proximal high error and interaction before, but with resampling it isn't clear this is necessary
        loss = (active_nlikelihood * interaction_schedule(i) + inter_weighted_nlikelihood * (1-interaction_schedule(i))) * done_flags
        run_optimizer(active_optimizer, full_model.forward_model, loss)
        
        # training the passive model with the weighted states, which is dangerous and not advisable
        if train_args.intrain_passive: run_optimizer(passive_optimizer, full_model.passive_model, passive_nlikelihood)

        # run the interaction model training if the interaction model is not already trained
        if train_args.pretrain_interaction_iters <= 0:
            for ii in range(inline_iters):
                interaction_loss, interaction_likelihood,\
                 interaction_binaries = _train_combined_interaction(full_model, train_args, rollouts,
                                                                     weights, inter_loss, interaction_optimizer)
        
        # reweight only when logging TODO: probably should not link  these two
        if i % train_args.log_interval == 0:
            combined_logging(full_model, train_args, rollouts, test_rollout, i, idxes, batchvals,
                     interaction_likelihood, interaction_binaries, true_binaries,
                     prediction_params, passive_prediction_params,
                     target, active_l2, passive_l2, done_flags, interaction_schedule,
                     forward_error, forward_loss, passive_error, trace)
            
            # change the lambdas for reweighting, and generate new sampling weights
            inline_iters = inline_iter_schedule(i)
            active_weighting_lambda = active_weighting_schedule(i)
            active_weights = get_weights(active_weighting_lambda, rollouts.weight_binary)
            inter_weighting_lambda = interaction_weighting_schedule(i)
            interaction_weights = get_weights(inter_weighting_lambda, rollouts.weight_binary)