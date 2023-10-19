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
from Causal.Utils.instance_handling import compute_likelihood, get_batch, get_valid
from Causal.Utils.weighting import proximity_binary, get_weights
from Network.network_utils import pytorch_model, run_optimizer, get_gradient


def compute_attention_cause(full_model, full_batch, batch, args):
    active_full, weights = full_model.return_weights(batch)
    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices)

    # convert the weights to binaries and inter_grads, averages over layers, even though this is not quite principled unless there is only oen layer
    # weights of shape batch x num_layers x num_heads x keys x queries
    print("head_weights", weights[0].mean(dim=0).mean(dim=1))
    input_weights = pytorch_model.unwrap(weights.mean(dim=1).mean(dim=1) * done_flags.unsqueeze(-1) )
    input_weights = input_weights[:,0] # assumes only one key

    bins = (input_weights > args.inter_baselines.attention_threshold).astype(int)
    return bins, input_weights

def compute_attention_loss(full_model, batch, args):
    active_full, weights = full_model.return_weights(batch)
    done_flags = pytorch_model.wrap(1-batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices)
    input_weights = weights.mean(dim=1) * done_flags.unsqueeze(-1).unsqueeze(-1) # average layers dimension
    input_weights = input_weights.reshape(-1,full_model.num_inter) # assumes only one key
    # print(input_weights)
    entropy_loss = (-input_weights * torch.log(input_weights + 1e-6)).sum(axis = -1).mean()
    return entropy_loss
