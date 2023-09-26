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
from Network.network_utils import pytorch_model, run_optimizer

def get_passive_weights(args, full_model, object_rollout):
    if args.full_inter.proximal_weights:
        binaries, _, _, proximal = proximity_binary(full_model, object_rollout, full=True) # binaries are the states with NO proximity in all_models
    else:
        binaries = np.ones(len(object_rollout))
    if full_model.form == "full":
        valid = get_valid(object_rollout.sample(0)[0].valid, full_model.valid_indices)
        invalid = (1-valid).squeeze().nonzero()
        binaries[invalid] = 0
        # print(np.concatenate([np.expand_dims(binaries, axis=-1), valid], axis=-1))
    weights = get_weights(-1, binaries)
    return weights


def train_passive(full_model, args, rollouts, object_rollout, weights, active_optimizer, passive_optimizer):
    logger = forward_logger("pretrain_passive", args.record.record_graphs, args.inter.passive.passive_log_interval, full_model, filename=args.record.log_filename)
    active_logger = forward_logger("pretrain_active", args.record.record_graphs, args.inter.passive.passive_log_interval, full_model)

    passive_likelihoods = list()
    active_likelihoods = list()
    for i in range(args.inter.passive.passive_iters):
        start = time.time()
        full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollout, weights)
        weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0
        valid = get_valid(batch.valid, full_model.valid_indices) # valid is batch x num target indices binary vector indicating which targets are valid (NOT the full batch x num instances)
        done_flags = np.expand_dims(1-full_batch.done.squeeze(), -1)
        if args.inter.passive.train_passive:
            # get input-output values

            # the values to be predicted, values in the buffer are pre-normalized

            # compute network values
            # passive_prediction_params = full_model.passive_model(pytorch_model.wrap(batch.obs, cuda=full_model.iscuda)) # batch.target != target
            # print(np.concatenate([batch.target_diff, batch.valid, valid], axis=-1)[:10])
            # print("running passive")
            passive_prediction_params, passive_mask, target, passive_dist, passive_log_probs, passive_input = full_model.passive_likelihoods(batch)
            passive_log_probs = - passive_log_probs
            passive_done_log_probs = passive_log_probs * pytorch_model.wrap(done_flags, cuda=full_model.iscuda)

            # passive_prediction_params = full_model.apply_passive((pytorch_model.wrap(batch.tarinter_state, cuda=full_model.iscuda), pytorch_model.wrap(batch.obs, cuda=full_model.iscuda))) # batch.target != target
            # # Train the passive model
            # passive_log_probs = - full_model.dist(*passive_prediction_params).log_prob(target)
            passive_loss = compute_likelihood(full_model, args.train.batch_size, passive_log_probs, done_flags=done_flags, is_full = True, valid=valid)
            run_optimizer(active_optimizer, full_model.active_model, passive_loss) if args.full_inter.use_active_as_passive or full_model.cluster_mode else run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)
            # logging the passive model outputs
            logger.log(i, passive_loss, None, None, passive_log_probs  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                        passive_prediction_params, target  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, full_model, valid=valid)
            passive_likelihoods.append(pytorch_model.unwrap(passive_loss))
            # If pretraining the active model, trains with a fully permissible interaction model
        if args.inter.passive.pretrain_active:
            # print(full_model.target_num, len(full_model.all_names))
            # full_mask = torch.ones(args.train.batch_size, len(full_model.all_names) * full_model.target_num)
            # active_prediction_params, mask = full_model.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=full_model.iscuda), pytorch_model.wrap(full_mask, cuda = full_model.iscuda)) # THE ONLY DIFFERENT LINE FROM train_passive.py
            # # print(active_prediction_params[0].shape, batch.tarinter_state.shape, full_model.active_model.mean.object_dim, full_model.active_model.mean.single_object_dim, full_model.active_model.mean.first_obj_dim)
            # active_likelihood_full = - full_model.dist(*active_prediction_params).log_prob(target)

            active = time.time()
            # active_prediction_params, active_dist, active_log_probs = full_model.active_open_likelihoods(batch)
            # active_likelihood_full = - active_log_probs
            # active_prediction_params, active_dist, active_log_probs = full_model.active_likelihoods(batch, soft=True, cluster_choice=2)
            # active_likelihood_full = - active_log_probs
            # print("active run", time.time() - active)
            # aoptim = time.time()
            # active_hard_params, active_soft_params, active_full, \
            #     interaction_likelihood, hot_likelihood, hard_interaction_mask, soft_interaction_mask, full_interaction_mask, target, \
            #     active_hard_dist, active_soft_dist, active_full_dist, \
            #     active_hard_log_probs, active_soft_log_probs, active_full_log_probs, \
            #     active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.reduced_likelihoods(batch, 
            #                                     normalize=False, mixed=args.full_inter.mixed_interaction,
            #                                     input_grad = True, soft_eval = True, masking=["hard", "soft", "full"]) # TODO: the return signature has changed
            
            # print("running active")
            active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, _ = full_model.active_open_likelihoods(batch)
            active_likelihood_full, active_prediction_params = - active_full_log_probs, active_full if not full_model.cluster_mode else (active_full[0][...,target.shape[-1]:target.shape[-1] * 2], active_full[1][...,target.shape[-1]:target.shape[-1] * 2])
            active_loss = compute_likelihood(full_model, args.train.batch_size, active_likelihood_full, done_flags=done_flags, is_full = True, valid = valid)
            # print(pytorch_model.unwrap(active_loss.mean(0)),
            #     pytorch_model.unwrap((active_log_probs * pytorch_model.wrap(done_flags, cuda=full_model.iscuda)).mean(0)),
            #     pytorch_model.unwrap(active_prediction_params[0][0]), pytorch_model.unwrap(active_prediction_params[1][0]), target[0])
                # pytorch_model.unwrap(active_full[0][0][...,target.shape[-1]:]), pytorch_model.unwrap(active_full[1][0][...,target.shape[-1]:]), target[0])
            # print(full_model.name, batch.tarinter_state[0], target[0])
            # print("prediction", torch.ones(len(full_model.all_names) * full_model.target_num).shape, active_prediction_params[0][0], full_model.active_open_likelihoods(batch)[0][0][0], full_model.likelihoods(batch)[1][0][0])
            # print("likelihood", active_likelihood_full[0], full_model.active_open_likelihoods(batch)[-1][0], full_model.likelihoods(batch)[-2][0])
            run_optimizer(active_optimizer, full_model.active_model, active_loss)
            # logging the active model outputs
            # error
            # print("aoptim", time.time() - aoptim)
            active_likelihoods.append(pytorch_model.unwrap(active_loss))
            active_logger.log(i, active_loss, None, None, active_likelihood_full * pytorch_model.wrap(done_flags, cuda=full_model.iscuda), None, weight_rate, batch.done,
                                active_prediction_params, target, None, full_model, valid=valid)
            if args.EMFAC.train_reconstruction:
                mean_r, std_r, query_state = full_model.get_embed_recon(batch, reconstruction=True)
                loss = mean_r.reshape(mean_r.shape[0], mean_r.shape[1], -1) - pytorch_model.wrap(batch.tarinter_state, cuda=full_model.iscuda)
        if i % args.inter.passive.passive_log_interval == 0:
            print("trace values", np.mean(batch.trace, axis=0))
            print(full_model.name,batch.tarinter_state[0]  ,  full_model.norm.reverse(full_batch.obs[0], form="inter", name=full_model.name), target[0], full_model.norm.reverse(target[0], name=full_model.name, form="dyn" if full_model.predict_dynamics else "target"))
            print(full_model.extractor.reverse_extract(full_model.norm.reverse(target[0], name=full_model.name, form="dyn" if full_model.predict_dynamics else "target")))
            print("total time", time.time() - start)
            # print(passive_done_log_probs[:20])
            # print(target[:20])
            # print(passive_done_log_probs.mean())
        # print("passive", time.time()- start)
    outputs = list(zip(passive_likelihoods, active_likelihoods))
    return outputs, weights