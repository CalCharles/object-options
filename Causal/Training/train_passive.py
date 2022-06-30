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
from Network.network_utils import pytorch_model, run_optimizer
from Record.file_management import create_directory

def train_passive(full_model, rollouts, args, active_optimizer, passive_optimizer, weights=None):
    logger = forward_logger("passive", args.inter.passive.passive_log_interval, full_model, filename=args.record.log_filename)
    active_logger = forward_logger("active", args.inter.passive.passive_log_interval, full_model)

    outputs = list()
    for i in range(args.inter.passive.passive_iters):
        # get input-output values
        batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
        weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0

        # the values to be predicted, values in the buffer are pre-normalized
        target = batch.target_diff if args.inter.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=full_model.iscuda)

        # compute network values
        passive_prediction_params = full_model.passive_model(pytorch_model.wrap(batch.target, cuda=full_model.iscuda))
        
        # Train the passive model
        done_flags = 1-batch.done
        passive_likelihood_full = - full_model.dist(*passive_prediction_params).log_prob(target)
        passive_loss = passive_likelihood_full.sum(dim=-1).unsqueeze(1) * pytorch_model.wrap(done_flags, cuda=full_model.iscuda)
        run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)

        # logging the passive model outputs
        logger.log(i, passive_loss, None, None, passive_likelihood_full  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate,
                    passive_prediction_params, target, None, full_model)

        # If pretraining the active model
        if args.inter.passive.pretrain_active:
            active_prediction_params = full_model.active_model(pytorch_model.wrap(batch.inter_state, cuda=full_model.iscuda))
            active_likelihood_full = - full_model.dist(*active_prediction_params).log_prob(target)
            active_loss = active_likelihood_full.sum(dim=-1).unsqueeze(1) * pytorch_model.wrap(done_flags, cuda = full_model.iscuda)
            run_optimizer(active_optimizer, full_model.active_model, active_loss)

            # logging the active model outputs
            active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate,
                                active_prediction_params, target, None, full_model) 

    return outputs