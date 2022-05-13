# train interaction directly
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


def train_interaction(full_model, rollouts, args, trace, interaction_optimizer):
    outputs = list()
    inter_loss = nn.BCELoss()
    if args.inter.interaction.interaction_pretrain > 0:
        # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
        # trw encodes binaries of where interactions occur, which are converted into normalized weights
        trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        weights = get_weights(ratio_lambda=args.inter.combined.weighting[2], binaries=trw)

        # weights the values
        for i in range(args.inter.interaction.interaction_pretrain):
            # get the input and target values
            batch, idxes = rollouts.sample(args.batch_size, weights=weights)
            target = trace_targets[idxes]# if full_model.multi_instanced else trace[idxes]

            # get the network outputs
            # multi-instanced will have shape [batch, num_instances]
            interaction_likelihood = full_model.interaction_model(pytorch_model.wrap(batch.inter_state))

            # compute loss
            trace_loss = inter_loss(interaction_likelihood.squeeze(), target)
            run_optimizer(interaction_optimizer, full_model.interaction_model, trace_loss)
            
            # logging
            interaction_logging(full_model, args, batchvals, i, trace_loss, trace, interaction_likelihood, target)
            
            # change the weighting if necesary
            weights = reweighting(i, args, trw)
    return outputs