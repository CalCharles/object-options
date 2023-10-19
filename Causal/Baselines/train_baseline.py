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
from Causal.Baselines.gradient import compute_gradient_cause, compute_gradient_loss
from Causal.Baselines.attention import compute_attention_cause, compute_attention_loss
from Causal.Baselines.counterfactual import compute_counterfactual_cause, compute_counterfactual_loss
from Causal.Baselines.baseline_logger import baseline_interaction_logger

def get_baseline_type(inter_baseline_args):
    if inter_baseline_args.gradient_threshold > 0:
        return "gradient_baseline"
    elif inter_baseline_args.attention_threshold > 0:
        return "attention_baseline"
    elif inter_baseline_args.counterfactual_threshold > 0:
        return "counterfactual_baseline"

def train_basic_model(full_model, args, rollouts, object_rollout, test_rollouts, test_object_rollout, active_optimizer):
    active_logger = forward_logger("train_active", args.record.record_graphs, args.inter.passive.passive_log_interval, full_model)
    train_baseline_logger = baseline_interaction_logger("train_baseline_" + get_baseline_type(args.inter_baselines), args.record.record_graphs, args.inter.passive.passive_log_interval, full_model)
    test_baseline_logger = baseline_interaction_logger("test_baseline_" + get_baseline_type(args.inter_baselines), args.record.record_graphs, args.inter.passive.passive_log_interval, full_model)

    passive_likelihoods = list()
    active_likelihoods = list()
    outputs = list()
    for i in range(args.train.num_iters):
        start = time.time()
        full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollout, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
        # weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0
        valid = get_valid(batch.valid, full_model.valid_indices) # valid is batch x num target indices binary vector indicating which targets are valid (NOT the full batch x num instances)
        done_flags = np.expand_dims(1-full_batch.done.squeeze(), -1)

        # forward modeling loss
        active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, active_full_inputs = full_model.active_open_likelihoods(batch)
        active_likelihood_full, active_prediction_params = - active_full_log_probs, active_full if not full_model.cluster_mode else (active_full[0][...,target.shape[-1]:target.shape[-1] * 2], active_full[1][...,target.shape[-1]:target.shape[-1] * 2])
        active_loss = compute_likelihood(full_model, args.train.batch_size, active_likelihood_full, done_flags=done_flags, is_full = True, valid = valid)
        
        additional_losses = 0
        if args.inter_baselines.gradient_threshold > 0:
            additional_losses = compute_gradient_loss(full_model, full_batch, batch, args) * args.inter_baselines.grad_lasso_lambda
        elif args.inter_baselines.attention_threshold > 0:
            additional_losses = compute_attention_loss(full_model, batch, args) * args.inter_baselines.attention_lambda
        elif args.inter_baselines.counterfactual_threshold > 0:
            additional_losses = compute_counterfactual_loss(full_model, batch, args) * args.inter_baselines.counterfactual_lambda

        
        run_optimizer(active_optimizer, full_model.active_model, active_loss * 0.01 + additional_losses)
        active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, 1.0, batch.done,
                            active_prediction_params, target, None, full_model, valid=valid)
        if i % args.inter.passive.passive_log_interval == 0:
            if args.inter_baselines.gradient_threshold > 0:
                train_bins, soft_train_bins = compute_gradient_cause(full_model, full_batch, batch, args)
                test_full_batch, test_batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", test_rollouts, test_object_rollout, None, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
                test_bins, soft_test_bins = compute_gradient_cause(full_model, test_full_batch, test_batch, args)
            elif args.inter_baselines.attention_threshold > 0:
                train_bins, soft_train_bins = compute_attention_cause(full_model, full_batch, batch, args)
                test_full_batch, test_batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", test_rollouts, test_object_rollout, None, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
                test_bins, soft_test_bins = compute_attention_cause(full_model, full_batch, batch, args)
            elif args.inter_baselines.counterfactual_threshold > 0:
                train_bins, soft_train_bins = compute_counterfactual_cause(full_model, full_batch, batch, args)
                test_full_batch, test_batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", test_rollouts, test_object_rollout, None, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
                test_bins, soft_test_bins = compute_counterfactual_cause(full_model, full_batch, batch, args)
            train_baseline_logger.log(i, soft_train_bins, train_bins, batch.trace, active_loss, batch.done, active_prediction_params, target, full_model, valid=valid)
            test_baseline_logger.log(i, soft_test_bins, train_bins, test_batch.trace, active_loss, batch.done, active_prediction_params, target, full_model, valid=valid)
            outputs.append((train_bins, soft_train_bins, test_bins, soft_test_bins))
    return outputs