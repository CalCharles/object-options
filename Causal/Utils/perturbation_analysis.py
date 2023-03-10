import numpy as np
import time, copy
from Network.network_utils import pytorch_model, get_gradient
from Causal.Utils.get_error import get_error, error_types
import logging
from tianshou.data import Batch
from Causal.Utils.instance_handling import compute_likelihood

def perturb_outputs(factored_state, full_model, environment, target_name, perturbation_range, num_samples, eval_function, args, sample_env=False):
    # for each input factor separately
    # if perturbation range is negative, sample perturbation_range number of values in the range of possible values
    # otherwise, sample values +- perturbation range evenly
    # eval function takes in a state, environment, target_name and full model and outputs some evaluation
    # i.e. gradient, perturbation, etc.
    all_states = list()
    eval_list = list()
    env_list= list()
    for name in environment.all_names:
        class_name = name.strip('0123456789')
        if np.sum(perturbation_range) < 0:
            perturbation_range = environment.object_ranges[class_name]
        else:
            perturbation_range = [factored_state[name] - perturbation_range, factored_state[name] + perturbation_range]
        spaces = list()
        for s,e in zip(*perturbation_range):
            spaces.append(np.linspace(s,e,num = num_samples))
        sample_points = np.meshgrid(*spaces).T.reshape(-1, environment.object_ranges[class_name][0].shape)
        
        for p in sample_points:
            fs = copy.deepcopy(factored_state)
            fs[name] = p
            eval = eval_function(fs, full_model, environment, target_name, args)
            eval_list.append(fs, eval)

            if sample_env: # records the change in state of the target for a transition from fs
                environment.set_from_factored_state(fs)
                nfs = environment.step(factored_state["Action"]) # WARNING: requires time shifted actions
                env_list.append((fs, nfs[target_name] - factored_state[target_name]))

    return eval_list, env_list

def gradient_eval(factored_state, full_model, environment, target_name, args):
    full_state = args.inter_select(factored_state)
    target_state = args.target_select(factored_state)
    batch = Batch(tarinter_state = np.expand_dims(np.concatenate([full_state, target_state], axis=-1), axis = 0),
             obs = np.expand_dims(target_state, axis = 0))
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        active_hard_inputs, active_soft_inputs, active_full_inputs= full_model.likelihoods(batch, normalize=normalize, 
                                                                    mixed="mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction,
                                                                    input_grad = True, soft_eval=True)
    done_flags = pytorch_model.wrap(1-factored_state["Done"], cuda = full_model.iscuda).squeeze().unsqueeze(-1)

    # combine the cost function (extend possible interaction losses here)
    active_nlikelihood = compute_likelihood(full_model, 1, - active_soft_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    full_loss = active_nlikelihood.mean() + interaction_likelihood.sum() * args.lasso_lambda
    
    # loss and optimizer interaction_mask
    grad_variable = interaction_likelihood if args.inter_grad else active_full_inputs
    grad_variable = get_gradient(full_model, full_loss, grad_variables=grad_variable)
    # print(grad_variables[1].shape, active_soft_inputs[...,batch.obs.shape[-1]:].shape, active_soft_inputs.shape, batch.obs.shape)
    return grad_variable

def compute_full_model(factored_state, next_factored_state, full_model, target_name, args):
    full_state = args.inter_select(factored_state)
    target_state = args.target_select(factored_state)
    batch = Batch(tarinter_state = np.expand_dims(np.concatenate([full_state, target_state], axis=-1), axis = 0),
             obs = np.expand_dims(target_state, axis = 0),
             obs_next = args.target_select(next_factored_state),
            target_diff = full_model.norm(next_factored_state[target_name] - factored_state[target_name], form="dyn", name=full_model.name))
    return full_model.likelihoods(batch, normalize=False, 
                                        mixed="mixed" if args.full_inter.mixed_interaction == "hard" else args.full_inter.mixed_interaction,
                                        input_grad = True, soft_eval=True)

def likelihood_eval(factored_state, next_factored_state, full_model, environment, target_name, args):
    # returns the likelihood of the target (obs)
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        active_hard_inputs, active_soft_inputs, active_full_inputs= compute_full_model(factored_state, next_factored_state, full_model, environment, target_name, args)
    done_flags = pytorch_model.wrap(1-factored_state["Done"], cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    active_nlikelihood = compute_likelihood(full_model, 1, - active_soft_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    return active_nlikelihood

def prediction_eval(factored_state, next_factored_state, full_model, environment, target_name, args):
    active_hard_params, active_soft_params, active_full, passive_params, \
        interaction_likelihood, soft_interaction_mask, hard_interaction_mask, hot_likelihood,\
        target, active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        active_hard_inputs, active_soft_inputs, active_full_inputs= compute_full_model(factored_state, next_factored_state, full_model, environment, target_name, args)
    done_flags = pytorch_model.wrap(1-factored_state["Done"], cuda = full_model.iscuda).squeeze().unsqueeze(-1)
    return pytorch_model.unwrap(active_hard_params[0]) - factored_state[target_name]