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


def compute_gradient_cause(full_model, full_batch, batch, args):
    active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, active_full_inputs = full_model.active_open_likelihoods(batch, input_grad=True)
    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices) # TODO: valid not implemented

    # combine the cost function (extend possible interaction losses here)
    active_likelihood_full, active_prediction_params = - active_full_log_probs, active_full if not full_model.cluster_mode else (active_full[0][...,target.shape[-1]:target.shape[-1] * 2], active_full[1][...,target.shape[-1]:target.shape[-1] * 2])
    active_loss = compute_likelihood(full_model, args.train.batch_size, active_likelihood_full, done_flags=done_flags, is_full = True, valid = valid)

    # full loss
    grad_variables = [active_full_inputs]
    grad_variables = get_gradient(full_model, active_loss, grad_variables=grad_variables)[0][...,batch.obs.shape[-1]:]
    grads = grad_variables.reshape(len(batch), full_model.num_inter, -1) # reshapes into the gradients per input
    inter_grads = pytorch_model.unwrap(grads.abs().sum(dim=-1) * done_flags)
    bins = (inter_grads > args.inter_baselines.gradient_threshold).astype(int)
    return bins, inter_grads
