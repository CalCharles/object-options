# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Training.loggers.forward_logger import forward_logger
from Causal.Training.loggers.interaction_logger import interaction_logger
from Causal.Training.loggers.logging import print_errors
from Causal.FullInteraction.Training.full_test import test_full
from Causal.Utils.weighting import get_weights
from Causal.Utils.get_error import error_types, get_error
from Network.network_utils import pytorch_model, run_optimizer
from Causal.Utils.instance_handling import compute_likelihood, get_batch
from Causal.FullInteraction.Training.full_train_combined_interaction import evaluate_active_interaction, get_masking_gradients, _train_combined_interaction

def _train_passive(i, full_model, args, rollouts, object_rollouts, passive_optimizer,active_optimizer, passive_weights, normalize=False):
    full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollouts, weights)
    passive_params,passive_mask, target, dist, log_probs, passive_input = full_model.passive_likelihoods(batch, normalize=normalize)
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - log_probs, done_flags=done_flags, is_full=True)
    run_optimizer(active_optimizer, full_model.active_model, passive_nlikelihood) if args.full_inter.use_active_as_passive else run_optimizer(passive_optimizer, full_model.passive_model, passive_nlikelihood)
    return idxes, target, passive_params, log_probs, passive_nlikelihood, done_flags



def train_combined(full_model, rollouts, object_rollouts, test_rollout, test_object_rollout,
    args, passive_weights, active_weights, interaction_weights, proximal,
    active_optimizer, passive_optimizer, interaction_optimizer,
    normalize=False):

    # initialize loggers
    full_model.toggle_active_as_passive(args.full_inter.use_active_as_passive)
    passive_logger = forward_logger("passive_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename) # replace the loggers with the all-logger
    logger = forward_logger("active_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename) # replace the loggers with the all-logger
    inter_logger = interaction_logger("interaction_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename) # replace the loggers with the all-logger
    
    # initialize loss function
    inter_loss = nn.BCELoss()

    # initialize interaction schedule, computes the weight to allow the active model to ignore certain values
    interaction_schedule = (lambda i: np.power(0.5, (i/args.inter.active.interaction_schedule))) if args.inter.active.interaction_schedule > 0 else (lambda i: 0.5)
    if args.full_inter.train_full_only: interaction_schedule = lambda i: 0
    inline_iter_schedule = lambda i: max(0, min(args.inter.active.inline_iters[0],
                                         np.power(2, (i/args.inter.active.inline_iters[2])) - 1) if args.inter.active.inline_iters[2] > 0 else args.inter.active.inline_iters[0]) 
    inline_iters = inline_iter_schedule(0)

    # initialize weighting schedules, by which the sampling weights change (shrink) over training
    _,_,awl,aws = args.inter.active.weighting 
    active_weighting_schedule = (lambda i: awl * np.power(0.5, (i/aws))) if aws > 0 else (lambda i: awl)
    
    # weighting value for training the forward model with the outputs of the interaction model versus using the open (all 1) mask
    iwl, iws = args.inter.active.interaction_weighting
    interaction_weighting_schedule = (lambda i: iwl * np.power(0.5, (i/iws))) if iws > 0 else (lambda i: iwl)
    
    # entropy schedules, entropy is used in the heads, or on the interaction likelihood (forcing the interaction mask towards 0,1 outputs)
    ewl, ews = args.full_inter.entropy_lambda
    entropy_lambda_schedule = (lambda i: ewl * np.power(0.5, (i/ews))) if ews > 0 else (lambda i: ewl)
    entropy_lambda = entropy_lambda_schedule(0)

    # initialize lambdas for the lasso value TODO: dual gradient descent, class dependent lasso values
    lasso_lambda, lwl, lhl, olls, lws = args.full_inter.lasso_lambda
    lasso_oneloss_schedule = (lambda i: lwl * np.power(0.5, (i/olls))) if olls > 0 else (lambda i: lwl) 
    lasso_halfloss_schedule = (lambda i: lhl * np.power(0.5, (i/olls))) if olls > 0 else (lambda i: lhl) 
    lasso_schedule = (lambda i: args.full_inter.lasso_lambda[0] * (1-np.power(0.5, (i * 3.0/lws)))) if lws > 0 else (lambda i: args.full_inter.lasso_lambda[0]) 
    lasso_oneloss_lambda = lasso_oneloss_schedule(0)
    lasso_halfloss_lambda = lasso_halfloss_schedule(0)
    lasso_lambda = lasso_schedule(0)
    print("awliwl", awl, iwl, olls, lws, lasso_schedule(100))

    uw = pytorch_model.unwrap
    if full_model.name == "all":
        full_vals, idxes = rollouts.sample(0, weights=active_weights)
        vals = full_vals
        tarinter_all = full_vals.obs
    else:
        full_vals, idxes = rollouts.sample(0, weights=active_weights)
        vals = object_rollouts[idxes]
        tarinter_all = np.concatenate([vals.obs, full_vals.obs], axis=-1)


    # print_errors(full_model, rollouts, object_rollouts, error_types=[error_types.ACTIVE_OPEN_RAW, error_types.ACTIVE_RAW, 
    #                                                                  error_types.ACTIVE, error_types.ACTIVE_OPEN, 
    #                                                                  error_types.ACTIVE_OPEN_LIKELIHOOD, error_types.PASSIVE_LIKELIHOOD,
    #                                                                  error_types.TRACE, error_types.DONE], prenormalize=normalize)
    start = time.time()
    for i in range(args.train.num_iters):
        # get data, weighting by active weights (generally more likely to sample high "value" states)
        # batch, idxes = object_rollouts.sample(args.train.batch_size, weights=active_weights)
        # full_batch = rollouts[idxes]

        for j in range(max(1, args.inter.active.active_steps)):
            full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollouts, active_weights)
            # print("target", batch.target_diff[:6])
            weight_rate = np.sum(active_weights[idxes]) / len(idxes)
            # run the networks and get both the active and passive outputs (passive for interaction binaries)
            # active_hard_params, active_soft_params, active_full, passive_params, \
            #     interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
            #     target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
            #     active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs_act, \
            #     active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.likelihoods(batch, 
            #                                         normalize=normalize, mixed=args.full_inter.mixed_interaction,
            #                                         input_grad = True, soft_eval = True) # TODO: the return signature has changed
            active_hard_params, active_soft_params, active_full, \
                interaction_likelihood, hot_likelihood, hard_interaction_mask, soft_interaction_mask, full_interaction_mask, target, \
                active_hard_dist, active_soft_dist, active_full_dist, \
                active_hard_log_probs, active_soft_log_probs, active_full_log_probs, \
                active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.reduced_likelihoods(batch, 
                                                normalize=normalize, mixed=args.full_inter.mixed_interaction,
                                                input_grad = True, soft_eval = True, masking=["hard", "soft", "full"]) # TODO: the return signature has changed
            # assign done flags
            done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

            # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?
            active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_soft_log_probs, done_flags=done_flags, is_full=True) if args.full_inter.mixed_interaction == "weighting" else compute_likelihood(full_model, args.train.batch_size, - active_hard_log_probs, done_flags=done_flags, is_full=True)
            active_full_nlikelihood = compute_likelihood(full_model, args.train.batch_size, -active_full_log_probs, done_flags=done_flags, is_full=True)

            if full_model.cluster_mode:
                mask_loss = (soft_interaction_mask - full_model.check_passive_mask(interaction_likelihood)).abs().sum(-1).mean() ## replace with a rincipled method, this is the l1 loss
                loss = (active_nlikelihood * (1-interaction_schedule(i)) + active_full_nlikelihood * interaction_schedule(i)) + lasso_lambda * mask_loss # TODO: regularize the magnitude of the mask when training the active model more carefully
            else:
                loss = (active_nlikelihood * (1-interaction_schedule(i)) + active_full_nlikelihood * interaction_schedule(i))
            if args.inter.active.active_steps > 0: run_optimizer(active_optimizer, full_model.active_model, loss)

            # logging will probably break from the changing of the meaning of interactions
            # TODO: ccreate a good logger for the all dataset
            # print(active_nlikelihood.shape, active_full_nlikelihood.shape, active_log_probs.shape, active_params[0].shape, np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1).shape, batch.trace.shape, np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1).shape)
            single_trace = np.expand_dims(np.sum(np.sum(batch.trace - 1, axis=-1), axis=-1), axis=-1) if len(batch.trace.shape) == 3 else np.expand_dims(np.sum(batch.trace - 1, axis=-1), axis=-1)
            single_trace[single_trace > 1] = 1
            log_interval = logger.log(i, loss, active_nlikelihood * done_flags, active_full_nlikelihood * done_flags, 
                        active_hard_log_probs * done_flags, single_trace, weight_rate, full_batch.done,
                        (active_hard_params[0] * done_flags, active_hard_params[1] * done_flags), target * done_flags, 
                        np.expand_dims(np.sum(pytorch_model.unwrap(interaction_likelihood) - 1, axis=-1), axis=-1), 
                        full_model, no_print=j != 0)

        # training the passive model with the weighted states, which is dangerous and not advisable
        if args.inter.active.intrain_passive > 0:
            # print("training passive")
            for ii in range(args.inter.active.intrain_passive):
                idxes, passive_target, passive_params, passive_log_probs, passive_nlikelihood, passive_done_flags = _train_passive(i, full_model, args, rollouts, object_rollouts, passive_optimizer, active_optimizer, passive_weights, normalize=normalize)
                passive_weight_rate = np.sum(passive_weights[idxes]) / len(idxes)
                log_interval = passive_logger.log(i, passive_nlikelihood, passive_nlikelihood, passive_nlikelihood, 
                    passive_log_probs * passive_done_flags, None, passive_weight_rate, 1-passive_done_flags,
                    (passive_params[0] * done_flags, passive_params[1] * done_flags), passive_target * passive_done_flags, None, full_model)


        # run the interaction model training if the interaction model is not already trained
        # print("interaction pretrain", args.inter.interaction.interaction_pretrain)

        if args.inter.interaction.interaction_pretrain <= 0:
            for ii in range(int(inline_iters)):
                inter_idxes, interaction_loss, interaction_calc_likelihood,\
                     interaction_binaries, hot_likelihood, weight_count, inter_done_flags, grad_variables = _train_combined_interaction(full_model, args, rollouts, object_rollouts,
                                                                         lasso_oneloss_lambda,lasso_halfloss_lambda, lasso_lambda, entropy_lambda, interaction_weights, inter_loss, interaction_optimizer, normalize=normalize)
                single_trace = None
                if full_model.name == "all":
                    single_trace = np.expand_dims(np.sum(rollouts.trace[inter_idxes] - 1, axis=-1), axis=-1)
                elif (object_rollouts is not None and object_rollouts.trace is not None):
                    single_trace = np.expand_dims(np.sum(np.sum(object_rollouts.trace[inter_idxes] - 1, axis=-1), axis=-1), axis=-1) if len(object_rollouts.trace[inter_idxes].shape) == 3 else np.expand_dims(np.sum(object_rollouts.trace[inter_idxes] - 1, axis=-1), axis=-1)
                    single_trace[single_trace > 1] = 1
                # reweight only when logging TODO: probably should not link  these two
                inter_logger.log(i, interaction_loss, interaction_calc_likelihood, interaction_binaries, pytorch_model.unwrap(inter_done_flags), weight_count,
                    trace=None if single_trace is None else single_trace, no_print=ii != 0)
        
        # print(i, lasso_lambda, lasso_oneloss_lambda, uw(loss.mean()), uw(interaction_loss.mean()), full_model.name,interaction_schedule(i),args.full_inter.mixed_interaction, 
        #     uw(active_nlikelihood.mean()), uw(active_full_nlikelihood.mean()), uw(passive_nlikelihood.mean()), np.sum(np.abs(batch.trace - uw(hard_interaction_mask))) / args.train.batch_size / batch.trace.shape[-1])
        # print([passive_log_probs_act.shape, active_hard_log_probs.shape, active_full_log_probs.shape, soft_interaction_mask.shape, batch.trace.shape])
        # print(active_hard_log_probs.shape, active_full_log_probs.shape, interaction_likelihood.shape, soft_interaction_mask.shape, batch.trace.shape)
        print(i, "combined_vals",np.concatenate([
                            # uw(passive_log_probs.sum(dim=-1).unsqueeze(-1)), 
                            uw(active_hard_log_probs.sum(dim=-1).unsqueeze(-1)),
                            uw(active_full_log_probs.sum(dim=-1).unsqueeze(-1)),
                            # uw(hot_likelihood),
                            uw(interaction_likelihood),
                            uw(soft_interaction_mask),
                            batch.trace], axis=-1)[:1])

        if i % args.inter.active.active_log_interval == 0:
            print(i, "speed", (args.inter.active.active_log_interval * i) / (time.time() - start))
            # change the lambdas for reweighting, and generate new sampling weights
            # if int(inline_iters) > 0:
            #     print_errors(full_model, rollouts[inter_idxes[90:]], error_types=[error_types.ACTIVE_RAW, error_types.ACTIVE, error_types.PASSIVE_LIKELIHOOD, error_types.ACTIVE_LIKELIHOOD, error_types.TRACE, error_types.INTERACTION, error_types.INTERACTION_BINARIES, error_types.PROXIMITY, error_types.DONE], prenormalize=normalize)
            entropy_lambda = entropy_lambda_schedule(i)
            lasso_oneloss_lambda = lasso_oneloss_schedule(i)
            lasso_halfloss_lambda = lasso_halfloss_schedule(i)
            lasso_lambda = lasso_schedule(i)
            inline_iters = inline_iter_schedule(i)
            active_weighting_lambda = active_weighting_schedule(i)
            # print(active_weighting_lambda)
            active_weights = get_weights(active_weighting_lambda, object_rollouts.weight_binary[:len(rollouts)].squeeze())
            # print(active_weights)
            inter_weighting_lambda = interaction_weighting_schedule(i)

            check_error = error_types.INTERACTION_HOT if full_model.cluster_mode else error_types.INTERACTION_RAW
            # print(get_error(full_model, rollouts, object_rollout=object_rollouts, error_type = check_error))
            # print(type(full_model.apply_mask(get_error(full_model, rollouts, object_rollout=object_rollouts, error_type = check_error), x=tarinter_all)))
            # print(type(get_error(full_model, rollouts, object_rollout=object_rollouts, error_type = check_error)), type(tarinter_all))
            mask_binary = (np.sum(np.round(full_model.apply_mask(get_error(full_model, rollouts, object_rollout=object_rollouts, error_type = check_error), x=tarinter_all)), axis=-1) > 1).astype(int)
            inter_bin = (object_rollouts.weight_binary[:len(rollouts)].squeeze() + mask_binary.squeeze())
            inter_bin[inter_bin> 1] = 1
            interaction_weights = get_weights(inter_weighting_lambda, inter_bin)
            print("inline_iters", inline_iters, inter_bin, object_rollouts.weight_binary)
            if args.full_inter.log_gradients:
                grad_variables = get_masking_gradients(full_model, args, rollouts, object_rollouts, lasso_oneloss_lambda,
                    lasso_halfloss_lambda, lasso_lambda, entropy_lambda, interaction_weights, inter_loss, normalize=normalize)
            # if i % (args.inter.active.active_log_interval * 2) == 0:
            #     print("log testing")
            #     test_dict = test_full(full_model, test_rollout, test_object_rollout, args, None)
            #     logger.log_testing(test_dict)
            # print(object_rollouts.weight_binary[:100], mask_binary.squeeze()[:100], inter_weighting_lambda)
            # print(interaction_weights[:100], interaction_weights[100:200],interaction_weights[200:300],interaction_weights[300:400],
            #     interaction_weights[400:500],interaction_weights[500:600],interaction_weights[600:700],interaction_weights[700:800],
            #     interaction_weights[800:900],interaction_weights[900:])
                # print(trace[inter_idxes], active_weights[inter_idxes])
            # print(full_model.norm.reverse(rollouts.target[48780:48800]), full_model.norm.reverse(rollouts.next_target[48780:48800]))
            # print(interaction_schedule(i))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.inter_state, form="inter"), full_model.norm.reverse(batch.target_diff, form="dyn"), batch.true_done], axis=-1))
            # print("inter tar", np.concatenate([full_model.norm.reverse(batch.target_diff, form="dyn"), full_model.norm.reverse(pytorch_model.unwrap(active_params[0]), form="dyn"), batch.true_done], axis=-1))
            # print(np.concatenate([batch.next_target,pytorch_model.unwrap(active_params[0]),
            #     # full_model.norm.reverse(pytorch_model.unwrap(active_params[0])), full_model.norm.reverse(pytorch_model.unwrap(active_params[1])),
            #     # pytorch_model.unwrap(active_params[0]), pytorch_model.unwrap(active_params[1]),
            #     pytorch_model.unwrap(active_nlikelihood), pytorch_model.unwrap(passive_nlikelihood), pytorch_model.unwrap(interaction_likelihood), trace[idxes], np.expand_dims(active_weights[idxes], 1)], axis=-1))
    test_dict = test_full(full_model, test_rollout, test_object_rollout, args, None)
    logger.log_testing(test_dict)
