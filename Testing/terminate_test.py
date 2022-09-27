import numpy as np
import string, os, copy
import torch
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps
from train_interaction import init_names
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.interaction_model import NeuralInteractionForwardModel, make_name
from State.object_dict import ObjDict
from Environment.Normalization.norm import MappedNorm, NormalizationModule
from Testing.normalization_test import create_dummy_data
from Environment.Environments.initialize_environment import initialize_environment
from State.state_extractor import StateExtractor
from tianshou.data import Batch

from Option.General.rew_term_fns import terminate_reward

  # Reward_terminate:
  #   full terminate reward test without timers
  #   interaction value consistent
  #   Combined reward with settings
  #   terminate components
  #   done components
  #   param vs interaction
  #   Termination and done connect properly
  #   Reward consistent with termination
  #   Reward correctly assigned for instance
  #   True reward/done used properly
terminal_forms = ["combined", "param", "reward", "timed"]

def generate_terms(args):
    vals = list()
    for form in terminal_forms:
        vals.append(generate_term_form(form, args))
    return vals

def generate_term_form(form, args):
    environment, record = initialize_environment(args.environment, args.record)
    names = init_names(args.train.train_edge)
    args.inter_extractor = CausalExtractor(names, environment)
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = args.inter_extractor.get_selectors()
    args.norm = NormalizationModule(environment.object_range, environment.object_dynamics, names, environment.object_instanced, args.inter_extractor.active)
    args.object_names = names
    args.extract.object_names = names
    args.extract.single_obs_setting = np.ones(len(args.extract.single_obs_setting)).tolist()
    args.extract.relative_obs_setting = np.ones(len(args.extract.relative_obs_setting)).tolist()
    args.extract.norm = args.norm
    args.extract.inter_extractor = args.inter_extractor
    state_extractor = StateExtractor(args.extract)
    interaction_model = torch.load(os.path.join(args.record.load_dir, make_name(names) + "_inter_model.pt")) if len(args.record.load_dir) > 0 else NeuralInteractionForwardModel(args, environment)

    args.option.interaction_model = interaction_model
    args.option.time_cutoff = -1 if form != "timed" else 8
    term_rew = terminate_reward[form if form != "timed" else "combined"](**args.option)

    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl")
    params, masks = load_from_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
    targets = list()
    batches = list()
    current_batch = list()
    for full_state, next_full_state, param, mask in zip(full_states, full_states[1:], params, masks):
        true_done, true_reward = full_state['factored_state']["Done"], full_state['factored_state']["Reward"]
        param, mask = np.array(param), np.array(mask)
        inter_state, target, next_target, target_diff = state_extractor.get_inter(full_state), state_extractor.get_target(full_state), state_extractor.get_target(next_full_state), state_extractor.get_diff(full_state, next_full_state)
        term, reward, done, inter, time_cutoff = term_rew(inter_state, target, next_target, target_diff, param, mask, true_done, true_reward) # TODO: true_inter = ?
        targets.append((term, reward, done, inter, time_cutoff))
        current_batch.append([term, reward, done, inter, time_cutoff])
        if len(current_batch) > 5:
            new_batch = list()
            for j in range(5):
                new_batch.append([current_batch[i][j] for i in range(5)])
            batches.append(copy.deepcopy(np.array(new_batch)))
            current_batch = list()
    save_to_pickle("Testing/testing_data/rewtermdone_com" + form + "_outputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl", targets)
    save_to_pickle("Testing/testing_data/rewtermdone_combined_batch_outputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl", batches)
    return targets, batches
    
# test one by one values
def term_reward_traj(args, state_extractor, input_path, param_path, output_path, time_cutoff):
    inputs, outputs = load_from_pickle(input_path), load_from_pickle(output_path)
    params, masks = load_from_pickle(param_path)
    args.option.time_cutoff = time_cutoff
    results = list()
    term_rew = terminate_reward[args.option.term_form](**args.option)
    for tar, fs, nfs, param, mask in zip(outputs, inputs, inputs[1:], params, masks):
        true_done, true_reward = fs['factored_state']["Done"], fs['factored_state']["Reward"]
        param, mask = np.array(param), np.array(mask)
        inter_state, target, next_target, target_diff = state_extractor.get_inter(fs), state_extractor.get_target(fs), state_extractor.get_target(nfs), state_extractor.get_diff(fs, nfs)
        term, reward, done, inter, time_cutoff = term_rew(inter_state, target, next_target, target_diff, param, mask, true_done, true_reward) # TODO: true_inter = ?
        results.append((term==tar[0], np.linalg.norm(reward - tar[1]), done==tar[2], inter==tar[3], time_cutoff==tar[4]))
    return results

# test batch values, but we can make it so that we don't need to be able to handle this
def term_reward_batch(args, state_extractor, input_path, param_path, output_path):
    inputs, outputs = load_from_pickle(input_path), load_from_pickle(output_path)
    full, next_full = inputs, inputs[1:]
    args.option.time_cutoff = 8
    params, masks = load_from_pickle(param_path)
    results = list()
    term_rew = terminate_reward[args.option.term_form](**args.option)
    for i in range(int(len(full[:100]) // 5)):
        tar = outputs[i]
        full_state = Batch(full[i*5:(i+1)*5])
        next_full_state = Batch(next_full[i*5:(i+1)*5])
        true_done, true_reward = full_state['factored_state']["Done"], full_state['factored_state']["Reward"]
        param, mask = np.array(params[i*5:(i+1)*5]), np.array(masks[i*5:(i+1)*5])
        inter_state, target, next_target, target_diff = state_extractor.get_inter(full_state), state_extractor.get_target(full_state), state_extractor.get_target(next_full_state), state_extractor.get_diff(full_state, next_full_state)
        term, reward, done, inter, time_cutoff = term_rew(inter_state, target, next_target, target_diff, param, mask, true_done, true_reward, reset = False) # TODO: true_inter = ?
        results.append((term==tar[0], np.linalg.norm(reward - tar[1]), done==tar[2], inter==tar[3], time_cutoff==tar[4]))
    return results

def test_reward_terminate_done(args):
    environment, record = initialize_environment(args.environment, args.record)
    names = init_names(args.train.train_edge)
    args.inter_extractor = CausalExtractor(names, environment)
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = args.inter_extractor.get_selectors()
    args.norm = NormalizationModule(environment.object_range, environment.object_dynamics, names, environment.object_instanced, args.inter_extractor.active)
    args.object_names = names
    args.extract.object_names = names
    args.extract.single_obs_setting = np.ones(len(args.extract.single_obs_setting)).tolist()
    args.extract.relative_obs_setting = np.ones(len(args.extract.relative_obs_setting)).tolist()
    args.extract.norm = args.norm
    args.extract.inter_extractor = args.inter_extractor
    state_extractor = StateExtractor(args.extract)
    interaction_model = torch.load(os.path.join(args.record.load_dir, make_name(names) + "_inter_model.pt")) if len(args.record.load_dir) > 0 else NeuralInteractionForwardModel(args, environment)

    args.option.interaction_model = interaction_model
    
    # test batches
    input_path = "Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl"
    param_path = "Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl"

    for form in terminal_forms:
        output_path = "Testing/testing_data/rewtermdone_com" + form + "_outputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl"
        args.option.term_form = form if form != "timed" else "combined"
        term_reward_traj(args, state_extractor, input_path, param_path, output_path, -1 if form != "timed" else 8)

    # batch tests
    for form in terminal_forms[:-1]:
        output_path = "Testing/testing_data/rewtermdone_combined_batch_outputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl"
        args.option.term_form = form
        term_reward_batch(args, state_extractor, input_path, param_path, output_path)
