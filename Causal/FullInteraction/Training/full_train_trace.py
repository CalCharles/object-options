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

def hot_traces(traces, num_clusters):
    hots = list()
    for i in range(num_clusters):
        a = np.zeros(num_clusters)
        a[i] = 1
        hots.append(a)
    seen = list()
    return_traces = list()
    at = 0
    for tr in traces:
        not_seen = True
        for i, v in enumerate(seen):
            if np.linalg.norm(tr - v) < 0.0001:
                not_seen = False
                break
        if not_seen:
            seen.append(copy.deepcopy(tr))
            return_traces.append(copy.deepcopy(hots[at]))
            at += 1
        else:
            return_traces.append(copy.deepcopy(hots[i]))
    return np.array(return_traces)

def train_interaction(full_model, rollouts, object_rollout, args, interaction_optimizer):
    outputs = list()
    inter_loss = nn.BCELoss()
    inter_logger = interaction_logger("trace_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename)
    print(object_rollout.weight_binary.shape)
    # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
    # trw encodes binaries of where interactions occur, which are converted into normalized weights
    binaries = object_rollout.weight_binary[:len(rollouts)]
    weights = get_weights(ratio_lambda=args.inter.active.weighting[2], binaries=binaries)
    # forms of training:
    traces = object_rollout.sample(0)[0].trace
    if args.full_inter.selection_train == "separate":
        traces = hot_traces(traces, args.interaction_net.cluster.num_clusters)
    elif args.full_inter.selection_train == "softened":
        SOFT_EPSILON = 0.2 
        traces = np.clip(traces, SOFT_EPSILON,1.0 - SOFT_EPSILON)
    elif args.full_inter.selection_train == "random":
        traces = np.clip(traces + (np.random.binomial(p=0.1, size=traces.shape)  * ((np.random.randint(0,2) - 0.5) * 2)), 0,1)
    elif args.full_inter.selection_train == "random_ones":
        traces = np.clip(traces + (np.random.binomial(p=0.1, size=traces.shape), 0,1))
    elif args.full_inter.selection_train == "proximity":
        traces = object_rollout.proximity
    elif args.full_inter.selection_train == "gradient":
        THRESHOLD = 0
        inp_grad_vals = input_gradients(full_model, rollouts, object_rollout, args)
        traces[inp_grad_vals > THRESHOLD] = 1 
        traces[inp_grad_vals <= THRESHOLD] = 0 
    if args.inter.interaction.subset_training > 0: # replace the subsets with the values
        rollouts, indices = rollouts.sample(args.inter.interaction.subset_training)
        object_rollout = object_rollout[indices]
        traces = traces[indices] if len(args.inter.interaction.selection_train) > 0 else None
        weights = weights[indices] / np.sum(weights[indices])
    # weights the values
    for i in range(args.inter.interaction.interaction_pretrain):
        # get the input and target values
        if full_model.name == "all":
            if args.inter.interaction.subset_training > 0:
                idxes = np.random.choice(np.arange(len(rollouts)), replace=True, p=weights, size=args.train.batch_size)
                batch = rollouts[idxes]
                full_batch = batch
            else:
                batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
                full_batch = batch
            batch.tarinter_state = batch.obs
        else:
            if args.inter.interaction.subset_training > 0:
                idxes = np.random.choice(np.arange(len(rollouts)), replace=True, p=weights, size=args.train.batch_size)
                full_batch = rollouts[idxes]
                batch = object_rollout[idxes]
            else:
                full_batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
                batch = object_rollout[idxes]
            batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
        batch.inter_state = full_batch.obs
        trace = traces[idxes] # in the all mode, this should be the full trace

        # get the network outputs
        # outputs the binary over all instances, in order of names, instance number
        if separate: # get the selection output
            interaction_likelihood = full_model.interaction_model.selection_network(pytorch_model.wrap(batch.tarinter_state, cuda=full_model.iscuda))
        else:
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
            if args.inter.interaction.subset_training > 0: # replace the subsets with the values
                weights = weights[indices] / np.sum(weights[indices])
    return outputs, binaries