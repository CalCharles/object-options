# action handler
import numpy as np
import copy, os
import gym
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps
from train_interaction import init_names
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.Utils.get_error import check_proximity
from Causal.Utils.weighting import passive_binary
from Causal.interaction_model import NeuralInteractionForwardModel, make_name
from Causal.Training.inline_trainer import InlineTrainer
from State.object_dict import ObjDict
from Environment.Normalization.norm import MappedNorm, NormalizationModule
from Testing.normalization_test import create_dummy_data
from Testing.interaction_test import create_batch
from Environment.Environments.initialize_environment import initialize_environment
from tianshou.data import Batch
from Network.network_utils import pytorch_model
from Option.instantiation import instantiate_action_map
from Graph.graph import load_graph
import torch

def generate_action_map(args):
    object_names = init_names(args.train.train_edge)
    environment, record = initialize_environment(args.environment, args.record)
    interaction_model = torch.load(os.path.join(args.record.load_dir, make_name(object_names) + "_inter_model.pt")) if len(args.record.load_dir) > 0 else NeuralInteractionForwardModel(args, environment)
    if args.record.load_dir:
        graph = load_graph(args.record.load_dir, args.torch.gpu)
        parent_interaction = graph.nodes[object_names.primary_parent].interaction
        parent_option = graph.nodes[object_names.primary_parent].option
    else: parent_interaction = None
    extractor = CausalExtractor(object_names, environment)
    target_select, full_parent_select, additional_select, additional_selectors, \
        padi_selector, parent_select, inter_select = extractor.get_selectors()

    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + object_names.target + ".pkl")
    filtered_active_set = parent_interaction.mask.filtered_active_set if parent_interaction is not None else None
    active_mask = parent_interaction.mask.active_mask if parent_interaction is not None else create_mask(full_states[0], parent_select)
    save_to_pickle("Testing/testing_data/action_params_"+environment.name+"_" + args.environment.variant +"_" + object_names.target + ".pkl", (filtered_active_set, active_mask))


    if not args.record.load_dir:
        parent_option = ObjDict({'action_map': ObjDict({'filtered_active_set': filtered_active_set})})
        parent_interaction = ObjDict({'active_mask': active_mask, "mask": ObjDict({"filtered_active_set": filtered_active_set, })})
    action_map = instantiate_action_map(args, parent_option, parent_interaction, environment, object_names)

    actions = [action_map.sample_policy_space() for a in range(len(full_states))]

    true_mapped, true_reversed = list(), list()
    for act, full_state, next_full_state in zip(actions, full_states, full_states[1:]):
        batch = create_batch(interaction_model, full_state, next_full_state)
        mapped_action = action_map.map_action(act, batch)
        reverse = action_map.reverse_map_action(mapped_action, batch)
        true_mapped.append(mapped_action)
        true_reversed.append(reverse)
    save_to_pickle("Testing/testing_data/true_action_"+environment.name+"_" + args.environment.variant +"_" + object_names.target + "_" + args.config_name + ".pkl", (actions, true_mapped, true_reversed))

    np.random.seed(1)
    policy_samples, mapped_policy_samples, mapped_samples, reversed_samples = list(), list(), list(), list()
    for full_state, next_full_state in zip(full_states, full_states[1:]):
        ps = action_map.sample_policy_space()
        batch = create_batch(interaction_model, full_state, next_full_state)
        mps = action_map.map_action(ps, batch)
        samp = action_map.sample()
        samp_reverse = action_map.reverse_map_action(samp, batch)
        policy_samples.append(ps)
        mapped_policy_samples.append(mps)
        mapped_samples.append(samp)
        reversed_samples.append(samp_reverse)
    save_to_pickle("Testing/testing_data/true_action_samples_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +"_" + args.config_name + ".pkl", (policy_samples, mapped_policy_samples, mapped_samples, reversed_samples))
    return (true_mapped, true_reversed), (policy_samples, mapped_policy_samples, mapped_samples, reversed_samples)

def test_action_mapping(actions, action_map, interaction_model, true_mapped_actions, true_reversed_actions, full_states):
    results = list()
    for act, true_mapped, true_reversed, full_state, next_full_state in zip(actions, true_mapped_actions, true_reversed_actions, full_states, full_states[1:]):
        batch = create_batch(interaction_model, full_state, next_full_state)
        mapped_action = action_map.map_action(act, batch)
        reverse = action_map.reverse_map_action(mapped_action, batch)
        results.append((np.linalg.norm(mapped_action - true_mapped), np.linalg.norm(true_reversed - reverse)))
    return results

def test_sample_spaces(action_map, interaction_model, true_policy_samples, true_mapped_policy_samples, true_samples, true_reversed, full_states):
    np.random.seed(1)
    results = list()
    for tps, tmps, ts, tr, full_state, next_full_state in zip(true_policy_samples, true_mapped_policy_samples, true_samples, true_reversed, full_states, full_states[1:]):
        ps = action_map.sample_policy_space()
        batch = create_batch(interaction_model, full_state, next_full_state)
        mps = action_map.map_action(ps, batch)
        samp = action_map.sample()
        samp_reverse = action_map.reverse_map_action(samp, batch)
        results.append((np.linalg.norm(ps - tps), np.linalg.norm(mps - tmps),
            np.linalg.norm(ts - samp), np.linalg.norm(tr - samp_reverse)))
    return results

def test_action_map_env(args):
    object_names = init_names(args.train.train_edge)
    environment, record = initialize_environment(args.environment, args.record)
    filtered_active_set, active_mask = load_from_pickle("Testing/testing_data/action_params_"+environment.name+"_" + args.environment.variant +"_" + object_names.target + ".pkl")
    interaction_model = torch.load(os.path.join(args.record.load_dir, make_name(object_names) + "_inter_model.pt")) if len(args.record.load_dir) > 0 else NeuralInteractionForwardModel(args, environment)
    parent_option = ObjDict({'filtered_active_set': filtered_active_set, "action_map": ObjDict({"filtered_active_set": filtered_active_set})})
    parent_interaction = ObjDict({'active_mask': active_mask, "mask": ObjDict({"filtered_active_set": filtered_active_set, })})
    action_map = instantiate_action_map(args, parent_option, parent_interaction, environment, object_names)

    inp_path = "Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + object_names.target + ".pkl"
    discrete_primitive = environment.num_actions if environment.discrete_actions and object_names.primary_parent == "Action" else 0
    discrete, box = action_map.param_space(discrete_primitive)
    disc_check = discrete == (environment.discrete_actions and object_names.primary_parent == "Action") or len(filtered_active_set) < args.action_map.min_active_size
    if not discrete: box_check = box.low == true_low and box.high==true_high

    full_states = load_from_pickle(inp_path)
    actions, true_mapped, true_reversed = load_from_pickle("Testing/testing_data/true_action_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +"_" + args.config_name + ".pkl")
    am_results = test_action_mapping(actions, action_map, interaction_model, true_mapped, true_reversed, full_states)
    true_policy_samples, true_mapped_policy_samples, true_samples, true_reversed = load_from_pickle("Testing/testing_data/true_action_samples_"+environment.name+"_" + args.environment.variant +"_" + object_names.target + "_" + args.config_name + ".pkl")
    samp_results = test_sample_spaces(action_map, interaction_model, true_policy_samples, true_mapped_policy_samples, true_samples, true_reversed, full_states)

    return am_results, samp_results
