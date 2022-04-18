# train interaction directly
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
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_weights, get_targets

def train_interaction(full_model, rollouts, train_args, batchvals, trace, trace_targets, interaction_optimizer):
    outputs = list()
    inter_loss = nn.BCELoss()
    if train_args.pretrain_interaction_iters > 0:
        # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
        # trw encodes binaries of where interactions occur, which are converted into normalized weights
        trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        weights = get_weights(ratio_lambda=train_args.weighting[2], binaries=trw)

        # weights the values
        for i in range(train_args.pretrain_interaction_iters):
            # get the input and target values
            batch, idxes = rollouts.sample(train_args.batch_size, weights=weights)
            target = trace_targets[idxes]# if full_model.multi_instanced else trace[idxes]
            
            # get the network outputs
            # multi-instanced will have shape [batch, num_instances]
            interaction_likelihood = full_model.interaction_model(pytorch_model.wrap(batch.inter_state))

            # compute loss
            trace_loss = inter_loss(interaction_likelihood.squeeze(), target)
            run_optimizer(interaction_optimizer, full_model.interaction_model, trace_loss)
            
            # logging
            interaction_logging(full_model, train_args, batchvals, i, trace_loss, trace, interaction_likelihood, target)
            
            # change the weighting if necesary
            weights = reweighting(i, train_args, trw)
    return outputs