from train_option import init_option, init_buffer
import numpy as np
import copy, os
import gym
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps, numpy_factored
from Testing.interaction_test import create_batch
from Graph.graph import load_graph
    # Collection
    #     Aggregation alignment (next_state-current state match)
    #     Aggregation values (done, reward, terminate, act)
    #     Termination and reset effects: (reset correctly adjusts obs, target)
    #     Full data order and time counts (check time, full data sequential)
    #     Average return (check per-episode and mean)
    #     Observation/other feature selection (normalization, inter state, target, obs match, reconstruct from obs)
    # Replay buffer
    #     Temporal extension recording
    #     Reward/term/done/param
    #     Target obs inter state matching
    #     Action mapping
    # HER:
    #     Final parameter assignment (param and obs)
    #     Final parameter termination reward
    #     Partial value storage
    #     Interleaved sampling
    #     Correct next-state in temporal extension
    #     HER resampling properly activated
    #     early stopping

def generate_collect(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    option.policy = load_graph(args.record.load_dir, args.torch.gpu).nodes[object_names.target].option.policy
    save_to_pickle("Testing/testing_data/collect_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1] +"_policy.pkl", option.policy)
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)
    full_states = load_from_pickle("Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant + "_" +object_names.target + ".pkl")
    params, masks = load_from_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
    acts, action_chains = load_from_pickle("Testing/testing_data/trajectory_actions_"+environment.name+"_" + args.environment.variant +"_" +object_names.target+".pkl")
    debug_action_list = list(zip(acts, action_chains))
    # create debug actions set
    step = 0
    debug_actions, collect_results, start_states = list(), list(), list()
    for i in range(args.train.num_iters):
        start_states.append(train_collector.environment.get_state())
        collect_result = train_collector.collect(n_step=args.train.num_steps, debug=True, debug_actions = debug_action_list[step:], debug_states=(params[step:], masks[step:], full_states[step:]))
        debug_actions += [(cr[1].act, cr[1].action_chain) for cr in collect_result['debug']]
        collect_results.append(collect_result)
        step += collect_result["n/st"]
    save_to_pickle("Testing/testing_data/collect_buffers_fixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1]+".pkl", (train_collector.buffer, train_collector.her_buffer, train_collector.full_buffer))
    save_to_pickle("Testing/testing_data/collect_inputs_fixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target+".pkl", (debug_actions, start_states))

    # create set without debug actions
    np.random.seed(1)
    torch.manual_seed(1)
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)
    train_collector.environment.set_from_factored_state(start_states[0]['factored_state'])
    unfixed_collect_results = list()
    for i in range(args.train.num_iters):
        collect_result = train_collector.collect(n_step=args.train.num_steps)
        unfixed_collect_results.append(collect_result)
    save_to_pickle("Testing/testing_data/collect_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1]+".pkl", (collect_results, unfixed_collect_results))
    save_to_pickle("Testing/testing_data/collect_buffers_unfixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1]+".pkl", (train_collector.buffer, train_collector.her_buffer, train_collector.full_buffer))
    return (debug_actions, start_states), (collect_result, unfixed_collect_results)

def check_hindsight(her1, her2):
	hindsight_comp = list()
	for b1, b2 in zip(her1, her2):
		hindsight_comp.append(compare_batch(b1, b2))
	return hindsight_comp

def collect_set(args, train_collector, debug_actions, true_batch):
    single_res, aggregate_res, added_res, her_res = list(), list(), list(), list()
    collect_result = train_collector.collect(n_step=args.train.num_steps,debug=True, debug_actions = debug_actions) # TODO: make n-episode a usable parameter for collect
    for (added, single_batch, aggregate_batch, her_batch), (true_added, true_single, true_aggregate, true_her) in zip(collect_result["debug"], true_batch["debug"]):
        single_res.append(compare_batch(single_batch, true_single))
        aggregate_res.append(compare_batch(aggregate_batch, true_aggregate))
        if added: added_res.append(added == true_added)
        if len(her_batch) > 0: her_res.append(check_hindsight(her_batch, true_her))
    if "debug" in true_batch: del true_batch["debug"]
    if "debug" in collect_result: del collect_result["debug"]
    batch_res = compare_batch(true_batch, collect_result)
    return single_res, aggregate_res, added_res, her_res, batch_res


def collect_environment(args, train_collector, test_collector, start_state, debug_actions, true_batches):
    np.random.seed(1)
    torch.manual_seed(1)
    train_collector.environment.set_from_factored_state(start_state)
    single_reses, aggregate_reses, added_reses, her_reses, batch_reses = list(), list(), list(), list(), list()
    for i in range(args.train.num_iters):
        debug_acts = debug_actions[i * args.train.num_steps:] if debug_actions is not None else None
        true_batch = true_batches[i]
        single_res, aggregate_res, added_res, her_res, batch_res = collect_set(args, train_collector, debug_acts, true_batch)
        test_collector.collect(n_step=10) # gather a few test collections
        single_reses.append(single_res),aggregate_reses.append(aggregate_res),added_reses.append(added_res), her_reses.append(her_res), batch_reses.append(batch_res)
    return single_reses, aggregate_reses, added_reses, batch_reses

def compare_batch(b1, b2):
    final_batch = Batch()
    for k in b1.keys():
        if k not in b2: continue
        if k == "info": final_batch[k] = compare_batch(b1[k][0], b2[k][0])
        elif type(b1[k]) == Batch or type(b1[k]) == dict and k in b2: final_batch[k] = compare_batch(b1[k], b2[k])
        elif type(b1[k]) == np.ndarray:
            if b1[k].dtype != bool: final_batch[k] = np.linalg.norm(b1[k] - b2[k])
            else: final_batch[k] = b1[k] == b2[k]
        elif type(b1[k]) == bool or type(b1[k]) == np.bool_: final_batch[k] = b1[k] == b2[k]
        else: final_batch[k] = np.abs(b1[k] - b2[k])
    return final_batch

def test_collector(args): # TODO: test hindsight with and without early stopping, 
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    option.policy = load_from_pickle("Testing/testing_data/collect_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1]+"_policy.pkl")
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)

    inp_path = "Testing/testing_data/collect_inputs_fixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target+".pkl"
    out_path = "Testing/testing_data/collect_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target + "_" + os.path.split(args.config)[-1]+".pkl"
    debug_actions, start_states = load_from_pickle(inp_path)
    true_batches_fixed, true_batches_policy = load_from_pickle(out_path) # true batches has batch_res, 
    fixed_action_results = collect_environment(args, train_collector, test_collector, start_states[0], debug_actions, true_batches_fixed)
    mixed_action_results = collect_environment(args, train_collector, test_collector, start_states[0], None, true_batches_policy)
    return fixed_action_results, mixed_action_results