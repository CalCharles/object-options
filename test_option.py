import numpy as np
import os, torch
from arguments import get_args
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Graph.graph import load_graph
from train_interaction import init_names
from Buffer.buffer import ParamReplayBuffer
from Collect.collector import OptionCollector
from Environment.Normalization.norm import NormalizationModule
from ReinforcementLearning.test_RL import testRL
from ReinforcementLearning.utils.RL_logger import RLLogger
from Option.instantiation import instantiate_buffers, instantiate_action_map, instantiate_rtd, instantiate_interaction, instantiate_sampler, instantiate_extractor

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)
    test_environment, test_record = initialize_environment(args.environment, args.record)
    object_names = init_names(args.train.train_edge)
    args.object_names = object_names

    # initializes the graph or loads it from args.record.load_dir
    graph = load_graph(args.record.load_dir, args.torch.gpu) # device is needed to load options properly

    target_name = object_names.target
    if len(args.train.override_name) > 0:
        target_name = args.train.override_name
    # gets the interaction model for the current control object
    interaction_model = graph.nodes[target_name].interaction
    if interaction_model is None: parent_option, parent_interaction, interaction_model = instantiate_interaction(args, graph, environment, object_names)
    interaction_model.regenerate(environment)
    
    option = graph.nodes[target_name].option
    if option.interaction_model is None: option.assign_interaction_model(interaction_model)
    test_buffer = ParamReplayBuffer(args.collect.buffer_len, stack_num=1)

    args.collect.env_reset = environment.self_reset
    collector = OptionCollector(option.policy, environment, test_buffer, False, option, None, True, interaction_model.multi_instanced, None, args)

    test_logger = RLLogger(target_name + "_test", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.test_log_maxlen)
    full_logger = RLLogger(target_name + "_full", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.test_log_maxlen * args.train.num_iters)
    testRL(args, collector, option, graph, test_logger, full_logger)

