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


def train_interaction(full_model, rollouts, object_rollout, args, interaction_optimizer):
    outputs = list()
    inter_loss = nn.BCELoss()
    if args.inter.interaction.interaction_pretrain > 0:
        # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
        # trw encodes binaries of where interactions occur, which are converted into normalized weights
        binaries = np.sum(rollouts.weights, axis=-1).astype(bool).astype(int)
        weights = get_weights(ratio_lambda=args.inter.combined.weighting[2], binaries=binaries)

        # weights the values
        for i in range(args.inter.interaction.interaction_pretrain):
            # get the input and target values
            full_batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
            batch = object_rollouts[target_name][idxes]
            trace = batch.trace

            # get the network outputs
            # outputs the binary over all instances, in order of names, instance number
            interaction_likelihood = full_model[target_name].interaction_model(pytorch_model.wrap(full_batch.obs))

            # compute loss
            trace_loss = (interaction_likelihood.squeeze() - trace).abs().sum(axis=-1).mean()
            run_optimizer(interaction_optimizer[target_name], full_model[target_name].interaction_model, trace_loss)
        
            # logging
            interaction_logging(full_model[target_name], args, batchvals, i, trace_loss, trace, interaction_likelihood, target)
            
            # change the weighting if necesary
            weights = reweighting(i, args, trw)
    return outputs