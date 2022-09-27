import numpy as np
import string, os
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps
from train_interaction import init_names
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.Utils.get_error import check_proximity
from Causal.Utils.weighting import passive_binary
from Causal.interaction_model import NeuralInteractionForwardModel
from Causal.Training.inline_trainer import InlineTrainer
from State.object_dict import ObjDict
from Environment.Normalization.norm import MappedNorm, NormalizationModule
from Testing.normalization_test import create_dummy_data
from Environment.Environments.initialize_environment import initialize_environment
from tianshou.data import Batch
import torch
from Network.network_utils import pytorch_model

  # Interaction:
  #   interaction values
  #   forward dynamics
  #   normalization
  #   proximity, passive error and binaries
def generate_interaction(args):
    environment, record = initialize_environment(args.environment, args.record)
    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] +".pkl")
    next_full_states = full_states[1:]
    args.object_names = init_names(args.train.train_edge)
    args.controllable = None
    interaction_model = torch.load(os.path.join(args.record.load_dir, make_name(object_names) + "_inter_model.pt")) if len(args.record.load_dir) > 0 else NeuralInteractionForwardModel(args, environment) 
    prox_results, hypo_results, like_results = list(), list(), list()
    for full_state, next_full_state in zip(full_states, next_full_states):
        parent, target, target_diff, next_target = interaction_model.parent_select(full_state["factored_state"]), interaction_model.target_select(full_state["factored_state"]), interaction_model.target_select(next_full_state["factored_state"]) - interaction_model.target_select(full_state["factored_state"]), interaction_model.target_select(next_full_state["factored_state"])
        prox = check_proximity(interaction_model, parent, target, normalized=False)
        if interaction_model.multi_instanced: prox = np.sum(prox, axis=1)
        prox_inst = check_proximity(interaction_model, parent, target, normalized=False)
        new_data = Batch(target = interaction_model.norm(target), target_diff = interaction_model.norm(target_diff, form="diff"), next_target = interaction_model.norm(next_target))
        passive_error = - pytorch_model.unwrap(interaction_model.passive_likelihoods(new_data)[-1].sum().unsqueeze(-1).unsqueeze(-1)) # TODO: fails for multi-instanced
        binaries = passive_binary(passive_error, args.inter.active.weighting, prox, full_state["factored_state"]["Done"])
        prox_results.append((prox, prox_inst, passive_error))

        state_tuple = (interaction_model.norm(interaction_model.inter_select(full_state["factored_state"]), form ='inter'), interaction_model.norm(interaction_model.target_select(full_state["factored_state"])))
        internd, prednd = interaction_model.predict_next_state(state_tuple, normalized=True, difference=False)
        internnd, prednnd = interaction_model.predict_next_state(state_tuple, normalized=False, difference=False)
        interN, predN = interaction_model.predict_next_state(state_tuple, normalized=True, difference=False)
        interd, predd = interaction_model.predict_next_state(state_tuple, normalized=False, difference=True)

        data = create_batch(interaction_model, full_state, next_full_state)
        inter_state, (mu_inter, var_inter), (pmu_inter, pvar_inter) = interaction_model.hypothesize(state_tuple)
        inter = interaction_model.interaction(data, prenormalize=False, use_binary=False)
        inter_bin = interaction_model.interaction(data, prenormalize=False, use_binary=True)

        inter_pn = interaction_model.interaction(data, prenormalize=True, use_binary=False)
        test_inter = interaction_model.check_interaction(inter)
        hypo_results.append((internd, prednd, internnd, prednnd, interN, predN, interd, predd, inter_state, mu_inter, var_inter, pmu_inter, pvar_inter, inter, inter_bin, inter_pn, test_inter))


        full_active_params, full_passive_params, full_inter, full_target, full_active_dist, full_passive_dist, full_active_log_probs, full_passive_log_probs = interaction_model.likelihoods(data, normalize = True)
        weighted_active_params, weighted_inter, weighted_dist, weighted_log_probs = interaction_model.weighted_likelihoods(data)
        passive_params, passive_dist, passive_log_probs = interaction_model.passive_likelihoods(data)
        active_params, active_dist, active_log_probs = interaction_model.active_likelihoods(data)
        like_results.append((full_active_params, full_passive_params, full_inter, full_target, full_active_dist, full_passive_dist, full_active_log_probs, full_passive_log_probs, weighted_active_params, weighted_inter, weighted_dist, weighted_log_probs, passive_params, passive_dist, passive_log_probs, active_params, active_dist, active_log_probs))
    prox_out_path = "Testing/testing_data/interaction_prox_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + ".pkl"
    save_to_pickle(prox_out_path, prox_results)
    hypothesis_out_path = "Testing/testing_data/interaction_hypo_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + "_hypothesis.pkl"
    save_to_pickle(hypothesis_out_path, hypo_results)

    likelihood_out_path = "Testing/testing_data/interaction_like_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + "_likelihood.pkl"
    save_to_pickle(likelihood_out_path, like_results)
    return prox_results, hypo_results, like_results

def check_set_values(args, full_states, interaction_model, pbins):
    trainer = InlineTrainer(args.inline, interaction_model, None) # does not check terminate reward
    results = list()
    for full_state, pb in zip(full_states, pbins):
        pt, pit, bt = pb
        data = create_batch(interaction_model, full_state, full_state)
        proximity, proximity_inst, binaries=trainer.set_values(data)
        results.append((proximity == pt, proximity_inst == pit, binaries == bt))
    return results

def check_hypothesis(full_states, next_full_states, interaction_model, true_values):
    results = list()
    for full_state, next_full_state, tv in zip(full_states, next_full_states, true_values):
        data = create_batch(interaction_model, full_state, next_full_state)
        state_tuple = (interaction_model.norm(interaction_model.inter_select(full_state["factored_state"]), form ='inter'), interaction_model.norm(interaction_model.target_select(full_state["factored_state"])))
        internd, prednd = interaction_model.predict_next_state(state_tuple, normalized=True, difference=False)
        internnd, prednnd = interaction_model.predict_next_state(state_tuple, normalized=False, difference=False)
        interN, predN = interaction_model.predict_next_state(state_tuple, normalized=True, difference=False)
        interd, predd = interaction_model.predict_next_state(state_tuple, normalized=False, difference=True)

        inter_state, (mu_inter, var_inter), (pmu_inter, pvar_inter) = interaction_model.hypothesize(full_state)
        inter = interaction_model.interaction(data, prenormalize=False, use_binary=False)
        inter_bin = interaction_model.interaction(data, prenormalize=False, use_binary=True)
        inter_pn = interaction_model.interaction(data, prenormalize=True, use_binary=False)
        test_inter = interaction_model.check_interaction(inter)

        true_internd, true_internnd, true_interN, true_interd, true_prednd,\
         true_prednnd, true_predN, true_predd, true_inter_state, true_mu_inter,\
          true_var_inter, true_pmu_inter, true_pvar_inter, true_inter, true_inter_bin,\
          true_inter_pn, true_test_inter = tv
        results.append( (internd == true_internd, internnd == true_internd, interN == true_internd, interd == true_internd, np.linalg.norm(prednd - true_prednd), np.linalg.norm(prednnd - true_prednnd), np.linalg.norm(predN - true_predN), np.linalg.norm(predd - true_predd), inter_state == true_inter_state, np.linalg.norm(mu_inter - true_mu_inter), np.linalg.norm(var_inter - true_var_inter), np.linalg.norm(pmu_inter - true_pmu_inter), np.linalg.norm(pvar_inter - true_pvar_inter), inter == true_inter, inter_bin == true_inter_bin, inter_pn == true_inter_pn, test_inter == true_test_inter))
    return results

def check_likelihoods(full_states, next_full_states, interaction_model, true_values):
    results = list()
    for full_state, next_full_state, tv in zip(full_states, next_full_states, true_values):
        data = create_batch(interaction_model, full_state, next_full_state)
        full_active_params, full_passive_params, full_inter, full_target, full_active_dist, full_passive_dist, full_active_log_probs, full_passive_log_probs = interaction_model.likelihoods(data, normalize = True)
        data = interaction_model.normalize_batch(data)
        weighted_active_params, weighted_inter, weighted_dist, weighted_log_probs = interaction_model.weighted_likelihoods(data)
        passive_params, passive_dist, passive_log_probs = interaction_model.passive_likelihoods(data)
        active_params, active_dist, active_log_probs = interaction_model.active_likelihoods(data)

        true_full_active_params, true_full_passive_params, true_full_inter, true_full_target, true_full_active_dist, true_full_passive_dist, true_full_active_log_probs, true_full_passive_log_probs, true_weighted_active_params, true_weighted_inter, true_weighted_dist, true_weighted_log_probs, true_passive_params, true_passive_dist, true_passive_log_probs, true_active_params, true_active_dist, true_active_log_probs = tv
        results.append(
                (np.linalg.norm(pytorch_model.unwrap(true_full_active_params[0] - full_active_params[0])),
                np.linalg.norm(pytorch_model.unwrap(true_full_passive_params[0] - full_passive_params[0])),
                np.linalg.norm(pytorch_model.unwrap(true_full_active_params[1] - full_active_params[1])),
                np.linalg.norm(pytorch_model.unwrap(true_full_passive_params[1] - full_passive_params[1])),
                np.linalg.norm(pytorch_model.unwrap(true_full_inter - full_inter)),
                np.linalg.norm(pytorch_model.unwrap(true_full_target - full_target)),
                np.linalg.norm(pytorch_model.unwrap(true_full_active_log_probs - full_active_log_probs)),
                np.linalg.norm(pytorch_model.unwrap(true_full_passive_log_probs - full_passive_log_probs)),
                np.linalg.norm(pytorch_model.unwrap(true_weighted_active_params[0] - weighted_active_params[0])),
                np.linalg.norm(pytorch_model.unwrap(true_weighted_active_params[1] - weighted_active_params[1])),
                np.linalg.norm(pytorch_model.unwrap(true_weighted_inter - weighted_inter)),
                np.linalg.norm(pytorch_model.unwrap(true_weighted_log_probs - weighted_log_probs)),
                np.linalg.norm(pytorch_model.unwrap(true_passive_params[0] - passive_params[0])),
                np.linalg.norm(pytorch_model.unwrap(true_passive_params[1] - passive_params[1])),
                np.linalg.norm(pytorch_model.unwrap(true_passive_log_probs - passive_log_probs)),
                np.linalg.norm(pytorch_model.unwrap(true_active_params[0] - active_params[0])),
                np.linalg.norm(pytorch_model.unwrap(true_active_params[1] - active_params[1])),
                np.linalg.norm(pytorch_model.unwrap(true_active_log_probs - active_log_probs)))
            )
    return results

def create_batch(interaction_model, full_state, next_full_state):
    target = interaction_model.target_select(full_state["factored_state"])
    parent_state = interaction_model.parent_select(full_state["factored_state"])
    done = full_state["factored_state"]["Done"]
    target_diff = interaction_model.target_select(full_state["factored_state"])
    inter_state = interaction_model.inter_select(full_state["factored_state"])
    next_target = interaction_model.target_select(next_full_state["factored_state"])
    data = Batch({"target": target, "parent_state": parent_state, "done": done, "target_diff": target_diff, "inter_state": inter_state, "next_target": next_target})
    return data

def test_interaction_env(args):

    environment, record = initialize_environment(args.environment, args.record)
    args.object_names = init_names(args.train.train_edge)
    args.controllable = None
    interaction_model = NeuralInteractionForwardModel(args, environment)

    inp_path = "Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] +".pkl"
    prox_out_path = "Testing/testing_data/interaction_prox_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + ".pkl"
    full_states = load_from_pickle(inp_path)
    prox_bin = load_from_pickle(prox_out_path)
    set_values_checks = check_set_values(args, full_states, interaction_model, prox_bin)

    # check hypothesis
    hypothesis_out_path = "Testing/testing_data/interaction_hypo_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + "_hypothesis.pkl"
    true_values = load_from_pickle(hypothesis_out_path)
    results_checks = check_hypothesis(full_states, full_states[1:], interaction_model, true_values)

    # check likelihoods
    likelihood_out_path = "Testing/testing_data/interaction_like_outputs_"+environment.name+"_" + args.environment.variant +"_" + args.train.train_edge[-1] + "_likelihood.pkl"
    true_values = load_from_pickle(likelihood_out_path)
    results_checks = check_likelihoods(full_states, full_states[1:], interaction_model, true_values)
