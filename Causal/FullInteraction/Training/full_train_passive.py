# train passive model
import logging
import numpy as np
import os, cv2, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Training.loggers.forward_logger import forward_logger
from Causal.Utils.instance_handling import compute_likelihood
from Network.network_utils import pytorch_model, run_optimizer

def train_passive_all(full_model, args, rollouts, object_rollout, active_optimizers, passive_optimizers):
    logger = forward_logger("passive", args.inter.passive.passive_log_interval, full_model, filename=args.record.log_filename)
    active_logger = forward_logger("active", args.inter.passive.passive_log_interval, full_model)
    if args.full_inter.proximal_weights:
        binaries = proximity_binary(full_model, object_rollout, full=True) 
    else:
        binaries = np.ones(len(object_rollout))
    weights = binaries / np.sum(binaries)

    outputs = list()
    for i in range(args.inter.passive.passive_iters):
        # get input-output values
        full_batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
        batch = object_rollout[idxes]
        # print (batch.inter_state.shape, full_model.norm.reverse(batch.inter_state[0], form = "inter"))
        weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0

        # the values to be predicted, values in the buffer are pre-normalized
        target = batch.target_diff if args.inter.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=full_model.iscuda)

        # compute network values
        passive_prediction_params = full_model[target_name].passive_model(pytorch_model.wrap(batch.target, cuda=full_model.iscuda)) # batch.target != target
        
        # Train the passive model
        done_flags = 1-batch.done
        passive_likelihood_full = - full_model.dist(*passive_prediction_params).log_prob(target)
        passive_loss = compute_likelihood(full_model, args.train.batch_size, passive_likelihood_full, done_flags=done_flags)
        run_optimizer(passive_optimizers[target_name], full_model[target_name].passive_model, passive_loss)

        # logging the passive model outputs
        logger.log(i, passive_loss, None, None, passive_likelihood_full  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                    passive_prediction_params, target  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, full_model)

        # If pretraining the active model, trains with a fully permissible interaction model
        if args.inter.passive.pretrain_active:
            active_prediction_params = full_model[target_name].active_model(pytorch_model.wrap(full_batch.obs, cuda=full_model.iscuda), pytorch_model.wrap(torch.ones(len(full_model.active_model.total_object_sizes)), cuda = full_model.iscuda)) # THE ONLY DIFFERENT LINE FROM train_passive.py
            active_likelihood_full = - full_model.dist(*active_prediction_params).log_prob(target)
            active_loss = active_likelihood_full.sum(dim=-1).unsqueeze(1) * pytorch_model.wrap(done_flags, cuda = full_model.iscuda)
            run_optimizer(active_optimizers[target_name], full_model[target_name].active_model, active_loss)
            # logging the active model outputs
            active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                                active_prediction_params, target, None, full_model) 
        if i % args.inter.passive.passive_log_interval == 0:
            print(full_model.norm.reverse(batch.inter_state[0], form="inter"), full_model.norm.reverse(batch.next_target[0]))
    return outputs