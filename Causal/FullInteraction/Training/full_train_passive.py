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

def train_passive(full_model, args, rollouts, object_rollout, active_optimizer, passive_optimizers):
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
        batch, idxes = object_rollout.sample(args.train.batch_size, weights=weights)
        full_batch = rollouts[idxes]
        batch.inter_state = full_batch.obs
        batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
        # print (batch.inter_state.shape, full_model.norm.reverse(batch.inter_state[0], form = "inter"))
        weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0

        # the values to be predicted, values in the buffer are pre-normalized
        target = batch.target_diff if args.inter.predict_dynamics else batch.obs_next
        target = pytorch_model.wrap(target, cuda=full_model.iscuda)

        # compute network values
        passive_prediction_params = full_model.passive_model(pytorch_model.wrap(batch.obs, cuda=full_model.iscuda)) # batch.target != target
        # print(passive_prediction_params[0].shape)
        
        # Train the passive model
        done_flags = np.expand_dims(1-batch.done, -1)
        # print(full_model.passive_model.mean.aggregate_final, batch.obs.shape, target.shape, full_model.name)
        passive_likelihood_full = - full_model.dist(*passive_prediction_params).log_prob(target)
        passive_loss = compute_likelihood(full_model, args.train.batch_size, passive_likelihood_full, done_flags=done_flags, is_full = True)
        run_optimizer(passive_optimizers, full_model.passive_model, passive_loss)

        # logging the passive model outputs
        logger.log(i, passive_loss, None, None, passive_likelihood_full  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                    passive_prediction_params, target  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, full_model)

        # If pretraining the active model, trains with a fully permissible interaction model
        if args.inter.passive.pretrain_active:
            # print(full_model.target_num, len(full_model.all_names))
            active_prediction_params = full_model.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=full_model.iscuda), pytorch_model.wrap(torch.ones(len(full_model.all_names) * full_model.target_num), cuda = full_model.iscuda)) # THE ONLY DIFFERENT LINE FROM train_passive.py
            # print(active_prediction_params[0].shape, batch.tarinter_state.shape, full_model.active_model.mean.object_dim, full_model.active_model.mean.single_object_dim, full_model.active_model.mean.first_obj_dim)
            active_likelihood_full = - full_model.dist(*active_prediction_params).log_prob(target)
            active_loss = compute_likelihood(full_model, args.train.batch_size, active_likelihood_full, done_flags=done_flags, is_full = True)
            run_optimizer(active_optimizer, full_model.active_model, active_loss)
            # logging the active model outputs
            active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                                active_prediction_params, target, None, full_model) 
        if i % args.inter.passive.passive_log_interval == 0:
            print(full_model.norm.reverse(full_batch.obs[0], form="inter", name=full_model.name), full_model.norm.reverse(target[0], name=full_model.name, form="dyn" if full_model.predict_dynamics else "target"))
    return outputs