# train passive model
import numpy as np
import os, cv2, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from file_management import save_to_pickle, load_from_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_targets

def train_passive(full_model, rollouts, train_args, active_optimizer, passive_optimizer, weights=None):
    outputs = list()
    for i in range(train_args.pretrain_iters):
        # get input-output values
        batch, idxes = rollouts.sample(train_args.batch_size, weights=weights)

        # the values to be predicted, values in the buffer are pre-normalized
        target = batch.target_diff if train_args.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target)

        # compute network values
        passive_prediction_params = full_model.passive_model(pytorch_model.wrap(batch.target))
        
        # Train the passive model
        done_flags = 1-batch.done
        passive_loss = - full_model.dist(*passive_prediction_params).log_prob(target)
        if full_model.multi_instanced: passive_loss = split_instances(passive_loss).sum(dim=2) * done_flags
        else: passive_loss = passive_loss.sum(dim=1).unsqueeze(1) * done_flags
        run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)

        # logging the passive model outputs
        log_model(full_model, "passive", train_args, i, batch, target, passive_loss, passive_prediction_params)


        # If pretraining the active model
        if train_args.pretrain_active:
            active_prediction_params = full_model.forward_model(pytorch_model.wrap(batch.inter_state))
            active_loss = - full_model.dist(*prediction_params).log_prob(target)
            if full_model.multi_instanced: active_loss = split_instances(active_loss).sum(dim=2) * done_flags
            else: active_loss = - full_model.dist(*prediction_params).log_prob(target).sum(dim=1).unsqueeze(1) * done_flags
            run_optimizer(active_optimizer, full_model.forward_model, active_loss)

            # logging the active model outputs
            log_model(full_model, "active", train_args, i, batch, target, active_loss, active_prediction_params)

    return outputs