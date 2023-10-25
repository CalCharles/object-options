import numpy as np
from Network.network_utils import pytorch_model, run_optimizer, get_gradient
from Causal.Utils.instance_handling import compute_likelihood, get_batch
import torch
import torch.nn.functional as F
import time

LOSS_DIFFERENCE_CONSTANT = 0

def compute_likelihood_adaptive_lasso(active_nlikelihood, args, lasso_lambda):
    # print(args.full_inter.converged_active_loss_value, 
    #       np.exp(-np.abs(args.full_inter.converged_active_loss_value- 
    #                     args.full_inter.adaptive_lasso_bias[0] - pytorch_model.unwrap(active_nlikelihood.mean())) / args.full_inter.adaptive_lasso[1]), 
    #       np.abs(args.full_inter.converged_active_loss_value - args.full_inter.adaptive_lasso_bias[0]- pytorch_model.unwrap(active_nlikelihood.mean())), 
    #       pytorch_model.unwrap(active_nlikelihood.mean()), 
    #       args.full_inter.adaptive_lasso[0] * (np.exp(-np.abs(args.full_inter.converged_active_loss_value- 
    #       args.full_inter.adaptive_lasso_bias[0] - pytorch_model.unwrap(active_nlikelihood.mean())) / args.full_inter.adaptive_lasso[1])))
    return (args.full_inter.adaptive_lasso[0] * (np.exp(-np.abs(args.full_inter.converged_active_loss_value - 
                                                                args.full_inter.adaptive_lasso_bias[0] - 
                                                                pytorch_model.unwrap(active_nlikelihood.mean())) / args.full_inter.adaptive_lasso[1]))
                                                      if args.full_inter.adaptive_lasso[0] > 0 else lasso_lambda)
    # return (args.full_inter.adaptive_lasso * (np.exp(-np.abs(active_nlikelihood.shape[-1] * 3 + pytorch_model.unwrap(active_nlikelihood.mean()))))
    #                                                   if args.full_inter.adaptive_lasso > 0 else lasso_lambda)

def compute_mean_adaptive_lasso(means, target, args, lasso_lambda):
    mean_difference = pytorch_model.unwrap(torch.linalg.norm(means - target, p=2, dim=-1))
    return (args.full_inter.adaptive_lasso[0] * np.exp(-mean_difference / args.full_inter.adaptive_lasso[1])
                if args.full_inter.adaptive_lasso[0] > 0 else lasso_lambda)


def compute_mean_var_adaptive_lasso(active_params, target, args, lasso_lambda):
    mean_difference = pytorch_model.unwrap(torch.linalg.norm(active_params[0] - target, ord=1, dim=-1))
    confidence = pytorch_model.unwrap(torch.linalg.norm(active_params[1], ord=1, dim=-1))
    return (args.full_inter.adaptive_lasso[0] * np.exp(-(mean_difference + confidence) / args.full_inter.adaptive_lasso[1])
                if args.full_inter.adaptive_lasso[0] > 0 else lasso_lambda)


def evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, active_params, interaction_likelihood, interaction_mask, active_log_probs, done_flags, proximity, target):
    active_nlikelihood = compute_likelihood(full_model, len(active_log_probs), - active_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # print(active_nlikelihood[:10])
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # adapts the lasso_lambda based on the input. If the active nlikelihood is low (large negative), it will approach e^{-0} = 1 if the error is high, it will approach e^-inf = 0
    # TODO: adaptive weighting value 3.0 is environment specific based on the level of natural stochasticity
    if args.full_inter.adaptive_lasso_type == "likelihood": lasso_lambda = compute_likelihood_adaptive_lasso(active_nlikelihood[done_flags.nonzero()[:,0]], args, lasso_lambda)
    elif args.full_inter.adaptive_lasso_type == "mean": lasso_lambda = compute_mean_adaptive_lasso(active_params[0][done_flags.nonzero()[:,0]], target[done_flags.nonzero()[:,0]], args, lasso_lambda)
    elif args.full_inter.adaptive_lasso_type == "meanvar": lasso_lambda = compute_mean_var_adaptive_lasso((active_params[0][done_flags.nonzero()[:,0]], active_params[1][done_flags.nonzero()[:,0]]), target[[done_flags.nonzero()[:,0]]], args, lasso_lambda)
    mask_loss = (interaction_likelihood - full_model.check_passive_mask(interaction_likelihood)).norm(p=args.full_inter.lasso_order, dim=-1).unsqueeze(-1) # penalize for deviating from the passive mask
    zero_mask_loss = (interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).unsqueeze(-1) # penalize for deviating from the passive mask
    one_mask_loss = (1-interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).unsqueeze(-1)
    half_mask_loss = (0.5 - interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).unsqueeze(-1)
    entropy_loss = torch.sum(-interaction_mask*torch.log(interaction_mask + 1e-10), dim=-1).unsqueeze(-1)
    
    # print(active_nlikelihood.shape, mask_loss.shape, active_log_probs.shape, full_model.check_passive_mask(interaction_likelihood).shape, interaction_likelihood.shape)
    full_loss = ((active_nlikelihood
                    + mask_loss * lasso_lambda
                    + one_mask_loss * (onemask_lambda)
                    + half_mask_loss * (halfmask_lambda)
                    + entropy_loss * entropy_lambda) * done_flags)
    # print(pytorch_model.unwrap(torch.cat([full_loss, active_nlikelihood, mask_loss * lasso_lambda * (1-onemask_lambda - halfmask_lambda), one_mask_loss * onemask_lambda * lasso_lambda, half_mask_loss * halfmask_lambda * lasso_lambda, entropy_loss * entropy_lambda, done_flags, interaction_likelihood], dim=-1)[:2]))
    # full_loss = - active_log_probs.sum(-1)
    # print(mask_loss, one_mask_loss, lasso_lambda, active_nlikelihood, entropy_loss)
    # print(mask_loss.mean()  * args.full_inter.lasso_lambda, active_log_probs[0], full_loss, full_loss.mean())
    # print(pytorch_model.unwrap(full_loss.mean()), pytorch_model.unwrap(mask_loss.mean()))
    return full_loss.mean(), active_nlikelihood.mean(), lasso_lambda

def evaluate_active_interaction_expert(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, active_params, 
                                        hot_likelihood, interaction_mask, active_log_probs, done_flags, proximity, target):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, 
                                            done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    lasso_lambda = compute_adaptive_lasso(active_nlikelihood, args, lasso_lambda)
    mean_mask_loss =  (hot_likelihood - (1./full_model.num_clusters)).unsqueeze(-1) * halfmask_lambda # make the agent select more diverse masks (to prevent mode collapse early)
    mask_loss = (interaction_mask).unsqueeze(-1) * lasso_lambda # penalize the selection of interaction masks
    entropy_loss = torch.sum(-interaction_mask*torch.log(interaction_mask), dim=-1).unsqueeze(-1) * entropy_lambda
    full_loss = ((active_nlikelihood + mean_mask_loss + mask_loss + entropy_loss) * done_flags).mean()
    # full_loss = - active_log_probs.sum(-1)
    # print(mask_loss.mean(), mean_mask_loss.mean(), active_nlikelihood.mean(), full_loss.mean())
    # print(mask_loss, mean_mask_loss, active_nlikelihood.shape, torch.cat([done_flags[:10],active_nlikelihood[:10],interaction_mask[:10], hot_likelihood[:10]], dim=-1), full_loss, halfmask_lambda, lasso_lambda)
    # print(pytorch_model.unwrap(full_loss.mean()), pytorch_model.unwrap(mask_loss.mean()))
    return full_loss, active_nlikelihood.mean(), lasso_lambda

def get_masking_gradients(full_model, args, rollouts=None, object_rollout=None, batch=None, full_batch=None, onemask_lambda=0, halfmask_lambda=0, lasso_lambda=0, entropy_lambda=0, weights=None, normalize=False):
    # prints out the gradients of the interaction mask, the active inputs and the full inputs
    if batch is None:
        full_batch, batch, idxes = get_batch(512, full_model.form == "all", rollouts, object_rollout, weights, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)

        # a statistic on weighting
        weight_count = np.sum(weights[idxes])
    # print("running inline iters")
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_hard_params, active_soft_params, active_full, \
        interaction_likelihood, hot_likelihood, hard_interaction_mask, soft_interaction_mask, full_interaction_mask, target, \
        active_hard_dist, active_soft_dist, active_full_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, \
        active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.reduced_likelihoods(batch, 
                                        normalize=normalize, mixed="mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction,
                                        input_grad = True, soft_eval = True, masking=["hard", "soft", "full"])

    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine the cost function (extend possible interaction losses here)
    interaction_loss, active_nlikelihood, lasso_lambda = evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, interaction_likelihood, soft_interaction_mask, active_soft_log_probs, done_flags, batch.proximity, target)
    
    # # loss and optimizer interaction_mask
    # grad_variables = [interaction_likelihood, active_soft_inputs]
    # grad_variables = get_gradient(full_model, interaction_loss, grad_variables=grad_variables)

    # full loss
    grad_variables = [interaction_likelihood, active_full_inputs]
    active_full_nlikelihood = compute_likelihood(full_model, 512, -active_full_log_probs, done_flags=done_flags, is_full=True)
    grad_variables = get_gradient(full_model, active_full_nlikelihood, grad_variables=grad_variables)

    # print(interaction_likelihood, active_softlambda_inputs, grad_variables[1])
    grad_variables[1] = grad_variables[1][...,batch.obs.shape[-1]:] if full_model.form == "full" else grad_variables[1]
    # print(grad_variables[1].shape, active_soft_inputs[...,batch.obs.shape[-1]:].shape, active_soft_inputs.shape, batch.obs.shape)
    grad_variable_statistics = list()
    grad_variable_revalue = list()
    for gv in grad_variables:
        gv = pytorch_model.unwrap(gv)
        # stdv = np.mean(np.std(gv, axis=0))
        # grad_variable_revalue.append(gv / stdv)
        if gv is not None:
            print ("gv", gv.shape, interaction_likelihood.shape, active_full_inputs.shape, batch.obs.shape[-1])
            gv = np.log(np.abs(gv))
            std = np.log(np.std(gv, axis=0))
        else: std = None
        grad_variable_revalue.append(gv)
        # grad_variable_statistics.append(np.std(gv, axis=0) / stdv)
        grad_variable_statistics.append(std)
    # print(active_soft_inputs.shape, grad_variables[1].shape)
    num_objects = int(np.sqrt(interaction_likelihood.shape[-1])) if full_model.form == "all" else interaction_likelihood.shape[-1] # in the all case the interactions are n interactions for each of the n objects (n * n in the final dimension)
    print("grad", np.concatenate((batch.trace.reshape(512,-1), 
                    pytorch_model.unwrap(interaction_likelihood), 
                    # grad_variable_revalue[0], 
                    # grad_variable_revalue[1],
                    
                    # np.sum(grad_variable_revalue[0].reshape(512, num_objects, -1), axis=-1)), axis=-1)[:10])
                    np.mean(grad_variable_revalue[1].reshape(512, num_objects, -1), axis=-1),
                    np.max(grad_variable_revalue[1].reshape(512, num_objects, -1), axis=-1)), axis=-1)[:10])
                    # , axis=-1)[:3], grad_variable_revalue[1].reshape(512, num_objects, -1)[:3])
    # print(grad_variable_revalue[1].reshape(512, num_objects, -1).shape, interaction_likelihood.shape)
    # print("grad il variance", interaction_likelihood)
    # print("grad as variance", np.sum(grad_variable_statistics[1].reshape(interaction_likelihood.shape[1], -1), axis=-1))
    # return idxes, interaction_loss, interaction_likelihood, hard_interaction_mask, weight_count, done_flags, grad_variables
    return grad_variables, 
# def evaluate_active_forward(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity)

def get_given_gradients(full_model, args, rollouts, object_rollout, weights, given_mask, normalize=False):
    # prints out the gradients of the interaction mask, the active inputs and the full inputs
    full_batch, batch, idxes = get_batch(512, full_model.form == "all", rollouts, object_rollout, weights, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
    # print("target", batch.target_diff[:6])
    weight_rate = np.sum(weights[idxes]) / len(idxes) if weights is not None else 1.0
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_given_params, \
        computed_interaction_likelihood, hot_likelihood, returned_given_mask, \
        target, \
        active_given_dist, \
        active_given_log_probs, \
        active_given_inputs= full_model.given_likelihoods(batch, given_mask, 
                                        normalize=normalize, input_grad=True)
    # assign done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine likelihoods to get a single likelihood for losses TODO: a per-element binary?

    # full loss
    grad_variables = [active_given_inputs]
    active_nlikelihood = compute_likelihood(full_model, 512, - active_given_log_probs, done_flags=done_flags, is_full=True)
    grad_variables = get_gradient(full_model, active_nlikelihood, grad_variables=grad_variables)

    grad_variables[0] = grad_variables[0][...,batch.obs.shape[-1]:] if full_model.form == "full" else grad_variables[1]
    grad_variable_statistics = list()
    grad_variable_revalue = list()
    for gv in grad_variables:
        gv = pytorch_model.unwrap(gv)
        # stdv = np.mean(np.std(gv, axis=0))
        # grad_variable_revalue.append(gv / stdv)
        if gv is not None:
            gv = np.log(np.abs(gv))
            std = np.log(np.std(gv, axis=0))
        else: std = None
        grad_variable_revalue.append(gv)
        grad_variable_statistics.append(std)
    num_objects = int(np.sqrt(computed_interaction_likelihood.shape[-1])) if full_model.form == "all" else computed_interaction_likelihood.shape[-1] # in the all case the interactions are n interactions for each of the n objects (n * n in the final dimension)
    print("grad", np.concatenate((batch.trace.reshape(512,-1), 
                    np.mean(grad_variable_revalue[0].reshape(512, num_objects, -1), axis=-1),
                    np.max(grad_variable_revalue[0].reshape(512, num_objects, -1), axis=-1)), axis=-1)[:10])
    return grad_variables, 


def _train_combined_interaction(full_model, args, rollouts, object_rollout, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, weights, inter_loss, interaction_optimizer, normalize=False, time_dict=None):
    # resamples because the interaction weights are different from the normal weights, and get the weight count for this
    full_model.dist_temperature = args.full_inter.dist_temperature

    full_batch, batch, idxes = get_batch(args.train.batch_size, full_model.form == "all", rollouts, object_rollout, weights, num_inter=full_model.num_inter, predict_valid=None if full_model.predict_next_state else full_model.valid_indices)
    if time_dict is not None: time_dict["inter_batch"] = time.time()
    # print(batch.trace[:20])

    # a statistic on weighting
    weight_count = np.sum(weights[idxes])
    # print("running inline iters")
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_soft_params, interaction_likelihood, hot_likelihood, hard_interaction_mask, soft_interaction_mask, target, active_soft_dist, active_soft_log_probs, active_soft_inputs = \
                            full_model.active_likelihoods(batch, normalize=normalize, soft=True, mixed = "mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction)
    if time_dict is not None: time_dict["inter_forward"] = time.time()

    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    # print(batch.trace[:10])

    # print("interaction_likelihood", interaction_likelihood.shape, active_soft_log_probs.shape)
    # combine the cost function (extend possible interaction losses here)
    # print((active_hard_log_probs * done_flags).mean(), (active_soft_log_probs * done_flags).mean(), (active_full_log_probs * done_flags), (passive_log_probs * done_flags).mean())
    # lasso_lambda = F.sigmoid(lasso_lambda) * 10
    interaction_loss, active_nlikelihood, lasso_lambda = evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, interaction_likelihood, soft_interaction_mask, active_soft_log_probs, done_flags, batch.proximity, target) \
                        if not full_model.cluster_mode else evaluate_active_interaction_expert(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, hot_likelihood, soft_interaction_mask, active_soft_log_probs, done_flags, batch.proximity, target)
    
    # print(interaction_loss)
    # loss and optimizer
    grad_variables = [interaction_likelihood, active_soft_inputs] if args.inter.active.log_gradients else list()
    grad_variables = run_optimizer(interaction_optimizer, full_model.active_model if full_model.attention_mode else full_model.interaction_model, interaction_loss, grad_variables=grad_variables)
    if time_dict is not None: time_dict["inter_grad"] = time.time()
    return idxes, interaction_loss, active_nlikelihood, interaction_likelihood, hard_interaction_mask, hot_likelihood,active_soft_params, target, weight_count, done_flags, grad_variables, lasso_lambda