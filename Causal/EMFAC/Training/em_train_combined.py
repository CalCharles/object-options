# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from Causal.Training.loggers.forward_logger import forward_logger
from Causal.Training.loggers.interaction_logger import interaction_logger
from Causal.Training.loggers.logging import print_errors
from Causal.FullInteraction.Training.full_test import test_full
from Causal.Utils.weighting import get_weights
from Causal.Utils.get_error import error_types, get_error
from Network.network_utils import pytorch_model, run_optimizer
from Causal.Utils.instance_handling import compute_likelihood, get_batch
from Causal.FullInteraction.Training.full_train_combined_interaction import get_masking_gradients

def train_forward_combined(full_model, args,
    rollouts, object_rollouts, test_rollout, test_object_rollout,
    full_optimizer, logger, i,
    normalize=False, given_mask=None):
    _,_,awl,aws = args.inter.active.weighting 
    full_weighting_schedule = (lambda i: awl * np.power(0.5, (i/aws))) if aws > 0 else (lambda i: awl)
    full_weighting_lambda = full_weighting_schedule(i)
    full_weights = get_weights(full_weighting_lambda, (object_rollouts if full_model.form == "full" else rollouts).weight_binary[:len(rollouts)].squeeze() if full_model.form == "full" else rollouts.weight_binary[:len(rollouts)].squeeze())

    full_model.active_model.reset_index(0, args.active_net.optimizer)
    start = time.time()
    interaction_schedule = (lambda i: np.power(0.5, (i/args.inter.active.interaction_schedule))) if args.inter.active.interaction_schedule > 0 else (lambda i: 0.5)
    for j in range(max(1, args.EMFAC.refine_iters)):
        full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollouts, full_weights)
        # print("target", batch.target_diff[:6])
        weight_rate = np.sum(full_weights[idxes]) / len(idxes)
        # run the networks and get both the active and passive outputs (passive for interaction binaries)
        active_hard_params, active_soft_params, active_full, \
            interaction_likelihood, hot_likelihood, hard_interaction_mask, soft_interaction_mask, full_interaction_mask, target, \
            active_hard_dist, active_soft_dist, active_full_dist, \
            active_hard_log_probs, active_soft_log_probs, active_full_log_probs, \
            active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.reduced_likelihoods(batch, 
                                            normalize=normalize, mixed=args.full_inter.mixed_interaction,
                                            input_grad = True, soft_eval = True, masking=["hard", "soft", "full"]) # TODO: the return signature has changed

        # assign done flags
        done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

        # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
        active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_soft_log_probs, done_flags=done_flags, is_full=True) if args.full_inter.mixed_interaction == "weighting" else compute_likelihood(full_model, args.train.batch_size, - active_hard_log_probs, done_flags=done_flags, is_full=True)
        active_full_nlikelihood = compute_likelihood(full_model, args.train.batch_size, -active_full_log_probs, done_flags=done_flags, is_full=True)
        loss = (active_nlikelihood * min(0.95, (1-interaction_schedule(i))) + active_full_nlikelihood * max(0.05, interaction_schedule(i)))

        if args.inter.active.active_steps > 0: run_optimizer(full_optimizer, full_model.active_model, active_nlikelihood)

        # logging will probably break from the changing of the meaning of interactions
        # TODO: create a good EM logger
        # print(active_nlikelihood.shape, active_full_nlikelihood.shape, active_log_probs.shape, active_params[0].shape, np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1).shape, batch.trace.shape, np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1).shape)
        single_trace = np.expand_dims(np.sum(np.sum(batch.trace - 1, axis=-1), axis=-1), axis=-1) if len(batch.trace.shape) == 3 else np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1)
        if full_model.form == "all": single_trace = np.sum(single_trace, axis= -2)
        single_trace[single_trace > 1] = 1
        log_interval = logger.log(i, loss, active_nlikelihood * done_flags, active_full_nlikelihood * done_flags, 
                    active_hard_log_probs * done_flags, single_trace, weight_rate, full_batch.done,
                    (active_hard_params[0] * done_flags, active_hard_params[1] * done_flags), target * done_flags, 
                    np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1), 
                    full_model, no_print=j != 0)

    if i % args.inter.active.active_log_interval == 0:
        print(i, "speed", (args.inter.active.active_log_interval * i) / (time.time() - start))

        if args.full_inter.log_gradients:
            grad_variables = get_masking_gradients(full_model, args, rollouts, object_rollouts, 0,
                0, args.EMFAC.binary_cost, 0, full_weights, loss, normalize=normalize)
        test_dict = test_full(full_model, test_rollout, test_object_rollout, args, None)
        logger.log_testing(test_dict)
