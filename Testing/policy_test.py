import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import copy, os, cv2
import tianshou as ts
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
import gym
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps, numpy_factored
from Testing.interaction_test import create_batch
from Testing.option_test import create_option_batch
from train_option import init_option


# policy_forms = {"basic": BasicPolicy, "image": ImagePolicy, 'grid': GridWorldPolicy, 'actorcritic': BasicActorCriticPolicy}

def generate_policy(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant + "_" +object_names.target + ".pkl")
    params, masks = load_from_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
    buffer, her_buffer = load_from_pickle("Testing/testing_data/saved_buffer_inputs_"+environment.name+"_" + args.environment.variant +"_" +object_names.target+".pkl")
    
    # generate buffer indices
    sample_merged = policy.sample_form == "merged" and len(her_buffer) > policy.MIN_HER
    indices = list()
    true_losses = list()
    for i in range(100):
        batch, indice, her_batch, her_indice, main_batch, main_indice = policy.sample_indices(args.train.batch_size, her_buffer, buffer)
        if sample_merged: indices.append({"main": main_indice, "her_indice": her_indice})
        else: indices.append({"main": indice})
        true_losses.append(policy.update(args.train.batch_size, buffer, her_buffer, pre_chosen=indices[-1]))
    save_to_pickle("Testing/testing_data/policy_indices_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl", indices)
    save_to_pickle("Testing/testing_data/policy_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl", true_losses)

    # generate forward updates
    epsilons, outputs, Qs = list(), list(), list()
    for last_full_state, full_state, next_full_state, param, mask in zip(full_states, full_states[1:], full_states[2:], params, masks):
        batch = create_option_batch(option, last_full_state, full_state, next_full_state, param, mask)
        epsilons.append(np.random.rand())
        policy.set_eps(epsilons[-1])
        outputs.append(policy.forward(batch, None))
        Qs.append(policy.compute_Q(batch, False))
    save_to_pickle("Testing/testing_data/forward_policy_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl", (epsilons, outputs, Qs))
    return true_losses, (epsilons, outputs, Qs)

def test_policy_batch(pol_batch, true_pol_batch):
    results = list()
    for k in pol_batch.keys():
        if type(pol_batch[k]) == np.ndarray:
            results.append(np.linalg.norm(pol_batch[k] - true_pol_batch[k]))
        elif pol_batch[k] is not None:
            results.append(pol_batch[k] - true_pol_batch[k])
    return results

def test_policy_update(policy, true_losses, buffer_indices, buffer, her_buffer):
    results = list()
    for bi, tl in zip(buffer_indices, true_losses):
        policy_losses = policy.update(1, buffer, her_buffer, pre_chosen = bi)
        results.append([policy_losses[k] - tl[k] for k in policy_losses.keys()])
    return results

def test_forward(option, policy,  full_states, params, masks, epsilons, true_outputs, true_Qs):
    results = list()
    for last_full_state, full_state, next_full_state, param, mask, epsilon, true_output, true_Q in zip(full_states, full_states[1:], full_states[2:], params, masks, epsilons, true_outputs, true_Qs):
        batch = create_option_batch(option, last_full_state, full_state, next_full_state, param, mask)
        policy.set_eps(epsilon)
        policy_output = policy.forward(batch, None)
        results.append((test_policy_batch(policy_output, true_output), np.linalg.norm(pytorch_model.unwrap(true_Q) - pytorch_model.unwrap(policy.compute_Q(batch, False)))))
    return results

def test_policy(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    buffer_path = "Testing/testing_data/saved_buffer_inputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target+".pkl"
    buffer_indices_path = "Testing/testing_data/policy_indices_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1] +".pkl"
    inp_path = "Testing/testing_data/trajectory_inputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target+".pkl"
    inp_param_mask = "Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + object_names.target + ".pkl"
    out_path = "Testing/testing_data/policy_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1] +".pkl"
    forward_path = "Testing/testing_data/forward_policy_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1] +".pkl"
    # rand_out_path = "Testing/testing_data/rand_policy_outputs_"+environment.name+"_"+ args.environment.variant + "_" + args.config + "_" +object_names.target+".pkl"

    # pre-chosen has the form: list-dict: her: her_indices, main: main_indices
    true_losses = load_from_pickle(out_path)
    buffer, her_buffer = load_from_pickle(buffer_path)
    buffer_indices = load_from_pickle(buffer_indices_path)
    learn_results = test_policy_update(policy, true_losses, buffer_indices, buffer, her_buffer)

    full_states = load_from_pickle(inp_path)
    params, masks = load_from_pickle(inp_param_mask)
    epsilons, true_outputs, true_Qs = load_from_pickle(forward_path)
    forward_results = test_forward(option, policy, full_states, params, masks, epsilons, true_outputs, true_Qs)
    return learn_results, forward_results