# train interaction directly
import numpy as np
import os, cv2, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Training.loggers.interaction_logger import interaction_logger
from Causal.Utils.weighting import get_weights
from Network.network_utils import pytorch_model, run_optimizer
from Record.file_management import create_directory


def train_interaction(full_model, rollouts, object_rollout, args, interaction_optimizer):
    outputs = list()
    inter_loss = nn.BCELoss()
    inter_logger = interaction_logger("trace_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename)
    # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
    # trw encodes binaries of where interactions occur, which are converted into normalized weights
    print(object_rollout.weight_binary.shape)
    binaries = object_rollout.weight_binary[:len(rollouts)]
    weights = get_weights(ratio_lambda=args.inter.active.weighting[2], binaries=binaries)
    # weights the values
    for i in range(args.inter.interaction.interaction_pretrain):
        # get the input and target values
        full_batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
        batch = object_rollout[idxes]
        batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
        batch.inter_state = full_batch.obs
        trace = batch.trace

        # get the network outputs
        # outputs the binary over all instances, in order of names, instance number
        interaction_likelihood = full_model.interaction(batch)
        done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

        # compute loss
        # trace_loss = (interaction_likelihood.squeeze() - pytorch_model.wrap(trace, cuda = full_model.iscuda)).abs().sum(axis=-1).mean()
        trace_loss = inter_loss(interaction_likelihood.squeeze(), pytorch_model.wrap(trace, cuda = full_model.iscuda))
        # done corrected traces
        trace_loss = trace_loss * done_flags
        # error
        run_optimizer(interaction_optimizer, full_model.interaction_model, trace_loss)
    
        # logging
        weight_rate = np.sum(weights[idxes]) / len(idxes)
        inter_logger.log(i, trace_loss, interaction_likelihood, interaction_likelihood, pytorch_model.unwrap(done_flags), weight_rate,
                trace=trace)
        if i % args.inter.active.active_log_interval == 0:
            print(np.concatenate((trace, pytorch_model.unwrap(interaction_likelihood)), axis=-1)[:5])
        # change the weighting if necesary
        weights = get_weights(args.inter.active.weighting[2], binaries.squeeze())
    return outputs, binaries