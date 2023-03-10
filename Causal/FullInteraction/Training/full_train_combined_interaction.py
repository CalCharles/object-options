import numpy as np
from Network.network_utils import pytorch_model, run_optimizer, get_gradient
from Causal.Utils.instance_handling import compute_likelihood
import torch

def evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, active_params, interaction_likelihood, interaction_mask, active_log_probs, done_flags, proximity):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    mask_loss = (interaction_likelihood - full_model.check_passive_mask(interaction_likelihood)).norm(p=args.full_inter.lasso_order, dim=-1).mean() # penalize for deviating from the passive mask
    zero_mask_loss = (interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).mean() # penalize for deviating from the passive mask
    one_mask_loss = (1-interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).mean()
    half_mask_loss = (0.5 - interaction_likelihood).norm(p=args.full_inter.lasso_order, dim=-1).mean()
    entropy_loss = torch.sum(-interaction_mask*torch.log(interaction_mask + 1e-10), dim=-1).mean()
    full_loss = (active_nlikelihood.mean()
                    + (mask_loss * (1-onemask_lambda - halfmask_lambda)
                    + one_mask_loss * (onemask_lambda)
                    + half_mask_loss * (halfmask_lambda)) * lasso_lambda
                    + entropy_loss * entropy_lambda)
    # full_loss = - active_log_probs.sum(-1)
    # print(mask_loss, one_mask_loss, lasso_lambda, active_nlikelihood, entropy_loss)
    # print(mask_loss.mean()  * args.full_inter.lasso_lambda, active_log_probs[0], full_loss, full_loss.mean())
    # print(pytorch_model.unwrap(full_loss.mean()), pytorch_model.unwrap(mask_loss.mean()))
    return full_loss

def evaluate_active_interaction_expert(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, active_params, 
                                        hot_likelihood, interaction_mask, active_log_probs, done_flags, proximity):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, 
                                            done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    mean_mask_loss =  (hot_likelihood - (1./full_model.num_clusters)).mean() * halfmask_lambda # make the agent select more diverse masks (to prevent mode collapse early)
    mask_loss = (interaction_mask).mean() * lasso_lambda # penalize the selection of interaction masks
    entropy_loss = torch.sum(-interaction_mask*torch.log(interaction_mask), dim=-1).mean() * entropy_lambda
    full_loss = active_nlikelihood.mean() + mean_mask_loss + mask_loss + entropy_loss
    full_loss = - active_log_probs.sum(-1)
    # print(mask_loss.mean(), mean_mask_loss.mean(), active_nlikelihood.mean(), full_loss.mean())
    # print(mask_loss, mean_mask_loss, active_nlikelihood.shape, torch.cat([done_flags[:10],active_nlikelihood[:10],interaction_mask[:10], hot_likelihood[:10]], dim=-1), full_loss, halfmask_lambda, lasso_lambda)
    # print(pytorch_model.unwrap(full_loss.mean()), pytorch_model.unwrap(mask_loss.mean()))
    return full_loss

def get_masking_gradients(full_model, args, rollouts, object_rollout, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, weights, inter_loss, normalize=False):
    # prints out the gradients of the interaction mask, the active inputs and the full inputs
    full_batch, idxes = rollouts.sample(512, weights=weights)
    batch = object_rollout[idxes]
    batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
    batch.inter_state = full_batch.obs

    # a statistic on weighting
    weight_count = np.sum(weights[idxes])
    # print("running inline iters")
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.likelihoods(batch, normalize=normalize, 
                                                                    mixed="mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction,
                                                                    input_grad = True, soft_eval=True)

    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine the cost function (extend possible interaction losses here)
    interaction_loss = evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, interaction_likelihood, soft_interaction_mask, active_soft_log_probs, done_flags, batch.proximity)
    
    # loss and optimizer interaction_mask
    grad_variables = [interaction_likelihood, active_soft_inputs]
    grad_variables = get_gradient(full_model, interaction_loss, grad_variables=grad_variables)
    print(interaction_likelihood, active_soft_inputs, grad_variables[1])
    grad_variables[1] = grad_variables[1][...,batch.obs.shape[-1]:]
    # print(grad_variables[1].shape, active_soft_inputs[...,batch.obs.shape[-1]:].shape, active_soft_inputs.shape, batch.obs.shape)
    grad_variable_statistics = list()
    grad_variable_revalue = list()
    for gv in grad_variables:
        gv = pytorch_model.unwrap(gv)
        stdv = np.mean(np.std(gv, axis=0))
        grad_variable_revalue.append(gv / stdv)
        grad_variable_statistics.append(np.std(gv, axis=0) / stdv)
    print(active_soft_inputs.shape, grad_variables[1].shape)
    print("grad", np.concatenate((batch.trace, pytorch_model.unwrap(interaction_likelihood), 
                    grad_variable_revalue[0], 
                    np.sum(grad_variable_revalue[1].reshape(512, grad_variable_statistics[0].shape[0], -1), axis=-1)), axis=-1)[:3])
    print("grad il variance", grad_variable_statistics[0])
    print("grad as variance", np.sum(grad_variable_statistics[1].reshape(grad_variable_statistics[0].shape[0], -1), axis=-1))
    # return idxes, interaction_loss, interaction_likelihood, hard_interaction_mask, weight_count, done_flags, grad_variables
    return grad_variables, 
# def evaluate_active_forward(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity)

def _train_combined_interaction(full_model, args, rollouts, object_rollout, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda, weights, inter_loss, interaction_optimizer, normalize=False):
    # resamples because the interaction weights are different from the normal weights, and get the weight count for this
    full_model.dist_temperature = args.full_inter.dist_temperature
    full_batch, idxes = rollouts.sample(args.train.batch_size, weights=weights)
    # full_batch, idxes = rollouts.sample(args.train.batch_size // 2, weights=weights)
    # batch_uni, idxes_uni = rollouts.sample(args.train.batch_size // 2)
    # idxes = idxes.tolist() + idxes_uni.tolist()
    # full_batch = full_batch.cat([full_batch, batch_uni])
    batch = object_rollout[idxes]
    batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
    batch.inter_state = full_batch.obs

    # a statistic on weighting
    weight_count = np.sum(weights[idxes])
    # print("running inline iters")
    # run the networks and get both the active and passive outputs (passive for interaction binaries)
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        active_hard_inputs, active_soft_inputs, active_full_inputs = full_model.likelihoods(batch, normalize=normalize, 
                                                                    mixed="mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction, 
                                                                    soft_select=True, soft_eval=False)

    # done flags
    done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine the cost function (extend possible interaction losses here)
    # print((active_hard_log_probs * done_flags).mean(), (active_soft_log_probs * done_flags).mean(), (active_full_log_probs * done_flags), (passive_log_probs * done_flags).mean())
    interaction_loss = evaluate_active_interaction(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, interaction_likelihood, soft_interaction_mask, active_soft_log_probs, done_flags, batch.proximity) \
                        if not full_model.cluster_mode else evaluate_active_interaction_expert(full_model, args, onemask_lambda, halfmask_lambda, lasso_lambda, entropy_lambda,
                        active_soft_params, hot_likelihood, soft_interaction_mask, active_full_log_probs, done_flags, batch.proximity)
    
    print(interaction_loss)
    # loss and optimizer
    grad_variables = [interaction_likelihood, active_hard_inputs, active_soft_inputs, active_full_inputs] if args.inter.active.log_gradients else list()
    grad_variables = run_optimizer(interaction_optimizer, full_model.active_model if full_model.attention_mode else full_model.interaction_model, interaction_loss, grad_variables=grad_variables)
    return idxes, interaction_loss, interaction_likelihood, hard_interaction_mask, hot_likelihood, weight_count, done_flags, grad_variables
