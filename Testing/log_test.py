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
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.utils.RL_logging import collect_test_trials
  # HER:
  #   Final parameter assignment (param and obs)
  #   Final parameter termination reward
  #   Partial value storage
  #   Interleaved sampling
  #   Correct next-state in temporal extension
  #   HER resampling properly activated
  #   early stopping

def generate_test_logging(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)
    debug_actions, start_states = load_from_pickle("Testing/testing_data/collect_inputs_fixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target+".pkl")
    
    # generate the losses decoupled from the action policy outputs using trainRL
    train_losses = list()
    trainRL(args, train_collector, test_collector, option, graph, (train_logger, test_logger, initial_logger), keep_losses = train_losses)
    
    test_results = collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, len(train_losses), args.policy.logging.initial_trials, False)
    for i, losses in enumerate(train_losses):
        collect_result = train_collector.collect(n_step=args.train.num_steps, debug_actions = debug_actions[args.train.num_steps * i:args.train.num_steps * (i+1)])
        train_logger.log_results(collect_result)
        train_logger.log_losses(losses)
        test_results = collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, len(train_losses), args.policy.logging.initial_trials, False)
    avg_reward, avg_success, train_log_string =  train_logger.print_log(10, force=True)
    avg_reward, avg_success, test_log_string =  test_logger.print_log(10, force=True)
    save_to_pickle("Testing/testing_data/log_losses_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl", train_losses)
    save_to_pickle("Testing/testing_data/log_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl", (train_log_string, test_log_string))
    return losses, train_log_string, test_log_string

def test_logging(args):
    environment, test_environment, record, test_record, option, models, policy, graph, object_names = init_option(args)
    train_buffer, test_buffer, hindsight, train_collector, test_collector, train_logger, test_logger, initial_logger = init_buffer(args, option, policy, graph, environment, test_environment, record, test_record, models)
    inp_path = "Testing/testing_data/collect_inputs_fixed_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  +".pkl"
    losses_path = "Testing/testing_data/log_losses_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl"
    out_path = "Testing/testing_data/log_outputs_"+environment.name+"_"+ args.environment.variant + "_" +object_names.target  + "_" + os.path.split(args.config)[-1]+".pkl"

    debug_actions, start_states = load_from_pickle(inp_path)
    train_losses = load_from_pickle(losses_path)
    environment.set_from_factored_state(start_states[0])

    test_results = collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, 0, args.policy.logging.initial_trials, False)
    for i, losses in enumerate(train_losses):
        collect_result = train_collector.collect(n_step=args.train.num_steps, debug_actions = debug_actions[args.train.num_steps * i:args.train.num_steps * (i+1)])
        train_logger.log_results(collect_result)
        train_logger.log_losses(losses)
        test_results = collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, len(train_losses), args.policy.logging.initial_trials, False)

    train_string, test_string = load_from_pickle(out_path)
    avg_reward, avg_success, train_log_string =  train_logger.print_log(10, force=True)
    avg_reward, avg_success, test_log_string =  test_logger.print_log(10, force=True)
    return train_string == train_log_string, test_string == test_log_string
