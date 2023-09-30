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
from Causal.FullInteraction.Training.full_train_combined_interaction import get_given_gradients

def train_forward_log_likelihood(full_model, args,
    rollouts, object_rollouts, test_rollout, test_object_rollout,
    active_weights, active_optimizer, logger, i,
    normalize=False, given_mask=None):

    start = time.time()
    for j in range(max(1, args.EMFAC.E_step_iters)):
        full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollouts, active_weights, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
        # print("target", batch.target_diff[:6])
        weight_rate = np.sum(active_weights[idxes]) / len(idxes) if active_weights is not None else 1.0
        # run the networks and get both the active and passive outputs (passive for interaction binaries)
        active_given_params, \
            computed_interaction_likelihood, hot_likelihood, returned_given_mask, \
            target, \
            active_given_dist, \
            active_given_log_probs, \
            active_given_inputs= full_model.given_likelihoods(batch, given_mask, 
                                            normalize=normalize)
        # assign done flags
        done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

        # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
        active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_given_log_probs, done_flags=done_flags, is_full=True)
        active_nlikelihood = (active_weights[idxes] / np.mean(active_weights)) * active_nlikelihood if args.EMFAC.weight_forward else active_nlikelihood

        # print(given_mask, pytorch_model.unwrap(active_nlikelihood[0]), pytorch_model.unwrap(active_given_log_probs[0]), pytorch_model.unwrap(target[0]), pytorch_model.unwrap(active_given_params[0])[0], pytorch_model.unwrap(active_given_params[1])[0])
        run_optimizer(active_optimizer, full_model.active_model, active_nlikelihood)

        # logging will probably break from the changing of the meaning of interactions
        # print(active_nlikelihood.shape, active_full_nlikelihood.shape, active_log_probs.shape, active_params[0].shape, np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1).shape, batch.trace.shape, np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1).shape)
        single_trace = np.expand_dims(np.sum(np.sum(batch.trace - 1, axis=-1), axis=-1), axis=-1) if len(batch.trace.shape) == 3 else np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1)
        if full_model.form == "all": single_trace = np.sum(single_trace, axis= -2)
        single_trace[single_trace > 1] = 1
        log_interval = logger.log(i, active_nlikelihood.mean(), active_nlikelihood * done_flags, None, 
                    None, single_trace, weight_rate, full_batch.done,
                    (active_given_params[0] * done_flags, active_given_params[1] * done_flags), target * done_flags, 
                    np.expand_dims(np.sum(pytorch_model.unwrap(computed_interaction_likelihood) - 1, axis=-1), axis=-1), 
                    full_model, no_print=j != 0)

    if i % args.inter.active.active_log_interval == 0:
        print(i, "speed", (args.inter.active.active_log_interval * i) / (time.time() - start))

        if args.full_inter.log_gradients:
            grad_variables = get_given_gradients(full_model, args, rollouts, object_rollouts,
                active_weights, given_mask, normalize=normalize)
        test_dict = test_full(full_model, test_rollout, test_object_rollout, args, None, printouts=False)
        logger.log_testing(test_dict)
        print("mask", given_mask)
    if given_mask is not None:
        test_like_active = get_error(full_model, rollouts, object_rollout=object_rollouts,
                                    error_type = error_types.ACTIVE_GIVEN_LIKELIHOOD, reduced=True, given_mask=given_mask)
    else:
        test_like_active = get_error(full_model, rollouts, object_rollout=object_rollouts,
                                    error_type = error_types.ACTIVE_LIKELIHOOD, reduced=True, given_mask=given_mask)

    return test_like_active
