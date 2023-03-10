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
from Causal.Utils.weighting import get_weights
from Causal.Training.loggers.interaction_logger import interaction_logger



def train_interaction(full_model, rollouts, args, trace, interaction_optimizer, weights):
    outputs = list()
    inter_loss = nn.BCELoss()
    inter_logger = interaction_logger("interaction", args.record.record_graphs, args.inter.active.active_log_interval, full_model)
    if args.inter.interaction.interaction_pretrain > 0:
        # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
        # trw encodes binaries of where interactions occur, which are converted into normalized weights
        trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace

        # weights the values
        for i in range(args.inter.interaction.interaction_pretrain):
            # get the input and target values
            batch, idxes = rollouts.sample(args.train.batch_size // 2, weights=weights)
            batch_uni, idxes_uni = rollouts.sample(args.train.batch_size // 2)
            idxes = idxes.tolist() + idxes_uni.tolist()
            batch = batch.cat([batch, batch_uni])
            weight_count = np.sum(weights[idxes])
            target = pytorch_model.wrap(trace[idxes], cuda = full_model.iscuda)# if full_model.multi_instanced else trace[idxes]
            done_flags = pytorch_model.wrap(1-batch.true_done, cuda = full_model.iscuda)

            # get the network outputs
            # multi-instanced will have shape [batch, num_instances]
            interaction_likelihood = full_model.interaction_model(pytorch_model.wrap(batch.inter_state))

            # compute loss
            trace_loss = inter_loss(interaction_likelihood.squeeze(), target.squeeze())
            run_optimizer(interaction_optimizer, full_model.interaction_model, trace_loss)
            
            # logging
            inter_logger.log(i, trace_loss, interaction_likelihood, None, pytorch_model.unwrap(done_flags), None,
                trace=trace[idxes])
        
            
        test_full(full_model, rollouts, args, full_model.names, None)

    return outputs