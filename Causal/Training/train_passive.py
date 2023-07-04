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

def train_passive(full_model, rollouts, args, active_optimizer, passive_optimizer, weights=None):
    logger = forward_logger("passive", args.record.record_graphs, args.inter.passive.passive_log_interval, full_model, filename=args.record.log_filename)
    active_logger = forward_logger("active", args.record.record_graphs, args.inter.passive.passive_log_interval, full_model)

    outputs = list()
    for i in range(args.inter.passive.passive_iters):
        # get input-output values
        batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
        # print (batch.inter_state.shape, full_model.norm.reverse(batch.inter_state[0], form = "inter"))
        weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0

        # the values to be predicted, values in the buffer are pre-normalized
        target = batch.target_diff if args.inter.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=full_model.iscuda)

        # compute network values
        passive_prediction_params = full_model.passive_model(pytorch_model.wrap(batch.target, cuda=full_model.iscuda)) # batch.target != target
        
        # Train the passive model
        done_flags = 1-batch.done
        # print(np.concatenate([pytorch_model.unwrap(passive_prediction_params[0][:10]), pytorch_model.unwrap(passive_prediction_params[1][:10]), pytorch_model.unwrap(target[:10]),done_flags[:10]], axis=-1))
        # print(np.concatenate([np.expand_dims(weights[idxes][:10], -1), full_model.norm.reverse(batch.inter_state[:10], form="inter"), full_model.norm.reverse(batch.next_target[:10]), pytorch_model.unwrap(target[:10]), done_flags[:10]], axis=-1))
        passive_likelihood_full = - full_model.dist(*passive_prediction_params).log_prob(target)
        passive_loss = compute_likelihood(full_model, args.train.batch_size, passive_likelihood_full, done_flags=done_flags)
        run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)

        # logging the passive model outputs
        logger.log(i, passive_loss, None, None, passive_likelihood_full  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                    passive_prediction_params, target  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, full_model)

        # If pretraining the active model
        if args.inter.passive.pretrain_active:
            active_prediction_params = full_model.active_model(pytorch_model.wrap(batch.inter_state, cuda=full_model.iscuda))
            active_likelihood_full = - full_model.dist(*active_prediction_params).log_prob(target)
            active_loss = active_likelihood_full.sum(dim=-1).unsqueeze(1) * pytorch_model.wrap(done_flags, cuda = full_model.iscuda)
            run_optimizer(active_optimizer, full_model.active_model, active_loss)
            # logging the active model outputs
            active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                                active_prediction_params, target, None, full_model) 
        if i % args.inter.passive.passive_log_interval == 0:
            print(full_model.norm.reverse(batch.inter_state[0], form="inter"), full_model.norm.reverse(batch.next_target[0]))
    return outputs