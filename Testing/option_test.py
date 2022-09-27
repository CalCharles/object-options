from train_option import init_option, init_buffer
import numpy as np
import copy, os
import gym
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps, numpy_factored
from Testing.interaction_test import create_batch

  # Temporal extension
  #   option chain param
  #   action mapping
  #   Mapped action + Reparameterization for next option (correct param/mask)
  #   Next option reachability
  #   Cutoff epsilons and usage
  #   Stopping criteria between actions
  #   Full action chain
  #   option internal value update
  #   Resetting properly

def generate_option_data(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)

    states, params, masks, acts, action_chains = list(), list(), list(), list(), list()
    for i in range(args.train.num_iters):
        res = train_collector.collect(n_step=args.train.num_steps, demonstrate = args.collect.demonstrate_option, debug=True)
        for (added, data, aggregate, her) in res["debug"]:
            full_state, param, mask, act, action_chain = data.full_state, data.param, data.mask, data.act, data.action_chain
            states.append(full_state), params.append(param), masks.append(mask), acts.append(act), action_chains.append(action_chain)
    save_to_pickle("Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant +"_" +object_names.target+".pkl", states)
    save_to_pickle("Testing/testing_data/trajectory_actions_"+environment.name+"_" + args.environment.variant +"_" +object_names.target+".pkl", (acts, action_chains))
    save_to_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl", (params, masks))
    save_to_pickle("Testing/testing_data/saved_buffer_inputs_"+environment.name+"_" + args.environment.variant +"_" +object_names.target+".pkl", (train_buffer, train_collector.her_buffer))
    return states

def generate_option(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant + "_" +object_names.target + ".pkl")
    params, masks = load_from_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
    def save_results(random=False):
        results = list()
        term_chains = list()
        state_chain = None
        term_chain = [False for i in range(option.get_depth())]
        mask_chain = [masks[0] for i in range(option.get_depth())]
        for last_full_state, full_state, next_full_state, param, mask in zip(full_states, full_states[1:], full_states[2:], params[1:], masks[1:]):
            param, mask = np.array(param), np.array(mask)
            batch = create_option_batch(option, last_full_state, full_state, next_full_state, param, mask)
            # run forward step
            act, chain, policy_batch, state, mask_val, needs_sample = option.extended_action_sample(batch, state_chain, term_chain, term_chain[1:], random=random, use_model=False, force=None)
            result = (act, chain, policy_batch, state, mask_val)
            # run terminate reward chain
            done, rewards, term_chain, inter, time_cutoff = option.terminate_reward_chain(full_state, next_full_state, param, chain, mask, mask_chain, true_done= full_state['factored_state']['Done'], true_reward=full_state['factored_state']['Reward'])
            term_chain = (done, rewards, term_chain, inter, time_cutoff)
            # run updater
            option.update(act, chain, term_chain, masks, update_policy=True)
            # compare outputs
            results.append(result)
            term_chains.append(term_chain)
        if random: save_to_pickle("Testing/testing_data/rand_option_outputs_"+environment.name+"_" + args.environment.variant + "_" + object_names.target+"_" + os.path.split(args.config)[-1] + ".pkl", (results, term_chains))
        else: save_to_pickle("Testing/testing_data/option_outputs_"+environment.name+"_" + args.environment.variant + "_" + object_names.target+"_" + os.path.split(args.config)[-1] + ".pkl", (results, term_chains))
        return results, term_chains
    return save_results(False), save_results(True)

def compare_list(l1, l2):
    if l1 is None:
        return l1 == l2
    results = list()
    for lv1, lv2 in zip(l1, l2):
        if type(lv1) == Batch or type(lv1) == dict:
            results.append({k: compare_list(lv1[k], lv2[k]) for k in lv1.keys()})
        elif type(lv1) == np.ndarray:
            if lv1.dtype == bool:
                results.append(lv1 == lv2)
            else:
                results.append( np.linalg.norm(lv1 - lv2))
        elif type(lv1) == list or type(lv1) == tuple:
            results.append(compare_list(lv1, lv2) for (lv1, lv2) in zip(l1, l2))
        else:
            if type(lv1) == bool or lv1 is None:
                results.append(lv1 == lv2)
            else:
                results.append(np.linalg.norm(pytorch_model.unwrap(lv1)-pytorch_model.unwrap(lv2)))
    return results

def create_option_batch(option, last_full_state, full_state, next_full_state, param, mask):
    last_full_state["factored_state"], full_state["factored_state"], next_full_state["factored_state"] = numpy_factored(last_full_state["factored_state"]), numpy_factored(full_state["factored_state"]), numpy_factored(next_full_state["factored_state"])
    batch = create_batch(option.interaction_model, full_state, next_full_state)
    obs = option.state_extractor.get_obs(last_full_state, full_state, param, mask) # one environment reliance
    obs_next = option.state_extractor.get_obs(full_state, next_full_state, param, mask) # one environment reliance
    batch.update(obs=[obs], obs_next=[obs_next], param=param, mask=mask, info={}, last_full_state = last_full_state, full_state = full_state, next_full_state=next_full_state)
    return batch

def check_option_outputs(option, inp_path, param_path, out_path, random=True):
    full_states = load_from_pickle(inp_path)
    params, masks = load_from_pickle(param_path)
    true_results, term_chain_groups = load_from_pickle(out_path)

    # initialize input values
    term_chain = option.reset(full_states[0])
    state_chain = None
    mask_chain = [masks[0] for i in range(option.get_depth())]
    results = list()
    for last_full_state, full_state, next_full_state, param, mask, true_result, term_chain_group in zip(full_states, full_states[1:], full_states[2:], params[1:], masks[1:], true_results, term_chain_groups):
        param, mask = np.array(param), np.array(mask)
        batch = create_option_batch(option, last_full_state, full_state, next_full_state, param, mask)
        # run forward step
        act, chain, policy_batch, state, masks, needs_sample = option.extended_action_sample(batch, state_chain, term_chain, term_chain[1:], random=random, use_model=False, force=None)
        result = (act, chain, policy_batch, state, masks)
        # run terminate reward chain
        done, rewards, term_chain, inter, time_cutoff = option.terminate_reward_chain(full_state, next_full_state, param, chain, mask, mask_chain, true_done= full_state['factored_state']['Done'], true_reward=full_state['factored_state']['Reward'])
        term_chain = (done, rewards, term_chain, inter, time_cutoff)
        # run updater
        option.update(act, chain, term_chain, masks, update_policy=True)
        # compare outputs
        results.append((compare_list(result, true_result), compare_list(term_chain, term_chain_group)))
    return results

def test_option(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    args.config = os.path.split(args.config)[1][:len(args.config) - 5] # removes the  path and .yaml
    inp_path = "Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant + "_" +object_names.target + ".pkl"
    param_path = "Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl"
    out_path = "Testing/testing_data/option_outputs_"+environment.name+"_" + args.environment.variant + "_" + object_names.target+"_" + os.path.split(args.config)[-1] + ".pkl"
    rand_out_path = "Testing/testing_data/rand_option_outputs_"+environment.name+"_" + args.environment.variant + "_" +object_names.target+"_" + os.path.split(args.config)[-1] +  ".pkl"
    return check_option_outputs(option, inp_path, param_path, rand_out_path, random=True), check_option_outputs(option, inp_path, param_path, out_path, random=False)