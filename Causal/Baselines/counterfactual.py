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

def generate_counterfactual_perturbation(idx_set, full_model, full_batch, batch, args): # generates a single counterfactual perturbation
    query_reshaped = batch.inter_state.reshape(len(batch), full_model.num_inter, -1)
    # assumes no padding, values normalized between -1, 1
    reassigned_values = np.random.rand(len(batch), full_model.num_inter, full_model.norm.pad_size)
    query_reshaped[np.arange(len(batch)),idx_set,:full_model.norm.pad_size] = reassigned_values[np.arange(len(batch)),idx_set,:full_model.norm.pad_size]
    batch.inter_state = query_reshaped.reshape(len(batch), -1)
    batch.tarinter_state = np.concatenate([batch.obs, batch.inter_state], axis=-1)
    return batch

def compute_counterfactual_cause(full_model, full_batch, batch, args):
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices)
    bins, counterfactual_variance = np.zeros((len(batch), full_model.num_inter)), np.zeros((len(batch), full_model.num_inter))
    for i in range(full_model.num_inter):
        reassign_batch = copy.deepcopy(batch)
        log_probs = list()
        actual_full, _, _, _, _, _, actual_active_full_log_probs, _ = full_model.active_open_likelihoods(reassign_batch)
        for j in range(args.inter_baselines.num_counterfactual):
            reassign_batch = generate_counterfactual_perturbation([i], full_model, full_batch, batch, args)
            active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, active_full_inputs = full_model.active_open_likelihoods(reassign_batch)
            # log_probs.append(pytorch_model.unwrap(active_full_log_probs.mean(dim=-1).unsqueeze(-1) * done_flags).squeeze()) # don't use batch size of 1
            log_probs.append(pytorch_model.unwrap(compute_distributional_distance(args, actual_full, actual_active_full_log_probs, active_full, active_full_log_probs) * done_flags).squeeze()) # don't use batch size of 1
        log_probs = np.stack(log_probs, axis=1) # shapes: batch -> batch x num_counterfactuals
        counterfactual_variance[:,i] = np.std(log_probs, axis=-1)
        bins[:,i] = counterfactual_variance[:,i] > args.inter_baselines.counterfactual_threshold
    return bins, counterfactual_variance

def compute_counterfactual_loss(full_model, full_batch, batch, args):
    return 0
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices)
    bins, counterfactual_variance = np.zeros((len(batch), full_model.num_inter)), np.zeros((len(batch), full_model.num_inter))
    for i in range(full_model.num_inter):
        reassign_batch = copy.deepcopy(batch)
        log_probs = list()
        actual_full, _, _, _, _, _, actual_active_full_log_probs, _ = full_model.active_open_likelihoods(reassign_batch)
        for j in range(args.inter_baselines.num_counterfactual):
            reassign_batch = generate_counterfactual_perturbation([i], full_model, full_batch, batch, args)
            active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, active_full_inputs = full_model.active_open_likelihoods(reassign_batch)


def compute_distributional_distance(args, actual_full, actual_log_probs, reassign_full, reassign_log_probs):
    # TODO: add  other distributional distances
    if args.inter_baselines.dist_distance == "likelihood": return (actual_log_probs.mean(dim=-1).unsqueeze(-1) - reassign_log_probs.mean(dim=-1).unsqueeze(-1)).abs()
    elif args.inter_baselines.dist_distance == "mean": return (actual_full[0] - reassign_full[0]).abs().mean()
    elif args.inter_baselines.dist_distance == "meanvar": return (actual_full[0] - reassign_full[0]).abs().mean() + (actual_full[1] - reassign_full[1]).abs().mean()

def compute_splitting(full_model, inter_masks, full_batch, batch, args, is_zero=False):
    # generates counterfactuals on the ones/zeros, computes the splitting comparison with the actual distribution
    # then returns the mean of the 1/0 splitting of the modeled distributions
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    valid = get_valid(batch.valid, full_model.valid_indices)
    inter_masks = 1 - inter_masks if is_zero else inter_masks # switch to the zeros if is_zero is True
    log_probs = list()
    actual_full, _, _, _, _, _, actual_active_full_log_probs, _ = full_model.active_open_likelihoods(reassign_batch)
    for j in range(args.inter_baselines.num_counterfactual):
        reassign_batch = generate_counterfactual_perturbation(np.nonzero(inter_masks), full_model, batch, full_batch, args)
        active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, active_full_inputs = full_model.active_open_likelihoods(reassign_batch)
        # the mean difference in log probability of the actual versus the active. we could use distributional distance instead, since we have the parameters of the gaussians
        log_probs.append((compute_distributional_distance(actual_full, actual_active_full_log_probs, active_full, active_full_log_probs) * done_flags).squeeze()) # don't use batch size of 1
    return torch.stack(log_probs, dim=1).mean(dim=-1) # batch_size x num_counterfactuals -> batch_size, 
