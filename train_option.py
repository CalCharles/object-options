import numpy as np
import os, torch
import logging
from arguments import get_args
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from train_interaction import generate_buffers
from Graph.graph import Graph, load_graph
from State.object_dict import ObjDict
from train_interaction import init_names
from Option.option import Option
from Option.instantiation import instantiate_buffers, instantiate_action_map, instantiate_rtd, instantiate_interaction, instantiate_sampler, instantiate_extractor
from ReinforcementLearning.policy import Policy
from Collect.collector import OptionCollector
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.utils.RL_logger import RLLogger
from Hindsight.hindsight import Hindsight
from Option.temporal_extension_manager import TemporalExtensionManager
from Causal.Training.inline_trainer import InlineTrainer

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)
    test_environment, test_record = initialize_environment(args.environment, args.record)
    object_names = init_names(args)

    # initializes the graph or loads it from args.record.load_dir
    graph = load_graph(args.record.load_dir, args.torch.gpu) # device is needed to load options properly
    if len(args.record.load_checkpoint) > 0: graph = load_graph(args.record.load_checkpoint, args.torch.gpu)

    # models contains the non-policy models for the current option
    models = ObjDict()
    
    # interaction might be dummy or real
    parent_option, parent_interaction, interaction_model = instantiate_interaction(args, graph, environment, object_names)
    models.interaction_model = interaction_model

    # the sampler samples goal states for the option to move to
    models.sampler = instantiate_sampler(args, interaction_model)
    models.test_sampler = instantiate_sampler(args, interaction_model)

    # the state extractor converts a factored state into the appropriate state for the option
    models.state_extractor = instantiate_extractor(args, interaction_model, environment, object_names)

    # manages termination of episodes at goals modulated by interactions
    models.terminate_reward = instantiate_rtd(args, interaction_model)

    # the inpolicy trainer trains the interaction model with the values in the replay buffer
    models.inline_trainer = InlineTrainer(args.inline, interaction_model, models.terminate_reward)

    # converts continuous actions to goals for the next level option
    models.action_map = instantiate_action_map(args, parent_option, parent_interaction, environment, object_names)

    models.temporal_extension_manager = TemporalExtensionManager(args.option)

    args.actor_net.pair.first_obj_dim, args.actor_net.pair.object_dim, args.actor_net.pair.aggregate_final = models.state_extractor.first_obj_dim, models.state_extractor.obj_dim, True
    args.critic_net.pair.first_obj_dim, args.critic_net.pair.object_dim, args.critic_net.pair.aggregate_final = models.state_extractor.first_obj_dim, models.state_extractor.obj_dim, True
    args.actor_net.pair.post_dim = -1 if args.actor_net.pair.post_dim == -1 else models.state_extractor.post_dim
    args.critic_net.pair.post_dim = -1 if args.critic_net.pair.post_dim == -1 else models.state_extractor.post_dim
    policy = Policy(models.action_map.discrete_actions, models.state_extractor.total_size, models.action_map.policy_action_space, args)
    
    option = Option(args, policy, models, parent_option) if len(args.record.load_checkpoint) == 0 else graph.nodes[object_names.target].option
    if args.torch.cuda: option.cuda(device=args.torch.gpu)
    train_buffer, test_buffer = instantiate_buffers(args, models)

    args.collect.env_reset = environment.self_reset
    hindsight = Hindsight(args, option, interaction_model) if args.hindsight.use_her else None
    train_collector = OptionCollector(policy, environment, train_buffer, args.policy.epsilon_random > 0, option, hindsight, False, interaction_model.multi_instanced, record, args)
    test_collector = OptionCollector(policy, environment, test_buffer, False, option, None, True, interaction_model.multi_instanced, None, args)

    graph.nodes[object_names.target].option = option
    train_logger = RLLogger(object_names.target + "_train", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.train_log_maxlen, args.record.log_filename)
    test_logger = RLLogger(object_names.target + "_test", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.test_log_maxlen)
    initial_logger = RLLogger(object_names.target + "_initial", args.record.record_rollouts, args.policy.logging.log_interval, 2)
    logging.info("config: " + str(args))
    trainRL(args, train_collector, test_collector, option, graph, (train_logger, test_logger, initial_logger))

    graph.save(args.record.save_dir)


