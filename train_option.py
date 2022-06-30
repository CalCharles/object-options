import numpy as np
import os, torch
from arguments import get_args
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from train_interaction import generate_buffers
from Graph.graph import Graph, load_graph
from State.object_dict import ObjDict
from train_interaction import init_names
from Causal.Sampling.sampling import samplers
from State.state_extractor import StateExtractor
from State.feature_selector import construct_object_selector
from Option.General.rew_term_fns import terminate_reward
from Option.action_map import ActionMap
from Option.temporal_extension_manager import TemporalExtensionManager
from Option.option import Option
from ReinforcementLearning.policy import Policy
from Environment.Normalization.norm import MappedNorm
from Buffer.buffer import ParamPriorityReplayBuffer, ParamReplayBuffer
from Collect.collector import OptionCollector
from Environment.Normalization.norm import NormalizationModule
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.utils.RL_logger import RLLogger
from Hindsight.hindsight import Hindsight

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

    # create a primitive option if the parent object is Action
    print(object_names.primary_parent, graph.nodes)
    parent_option = graph.nodes[object_names.primary_parent].option
    parent_interaction = graph.nodes[object_names.primary_parent].interaction

    # gets the interaction model for the current control object
    interaction_model = graph.nodes[object_names.target].interaction
    interaction_model.norm = NormalizationModule(environment.object_range, environment.object_dynamics, args.object_names)
    
    # models contains the non-policy models for the current option
    models = ObjDict()

    # the sampler samples goal states for the option to move to
    args.sample.mask = interaction_model.mask
    args.sample.target_select = interaction_model.target_select
    models.sampler = samplers[args.sample.sample_type](**args.sample)
    args.sample.test_sampler = True
    models.test_sampler = samplers[args.sample.sample_type](**args.sample)

    # the state extractor converts a factored state into the appropriate state for the option
    args.extract.inter_select = interaction_model.inter_select
    args.extract.target_select = interaction_model.target_select
    args.extract.parent_selectors = interaction_model.parent_selectors
    args.extract.parent_select = interaction_model.parent_select
    args.extract.additional_select = interaction_model.additional_select
    args.extract.object_names = object_names
    args.extract.norm = interaction_model.norm
    args.extract.max_target_objects = environment.object_instanced[object_names.target]
    args.extract.max_parent_objects = environment.object_instanced[object_names.primary_parent]
    models.state_extractor = StateExtractor(args.extract)

    # manages termination of episodes at goals modulated by interactions
    args.option.target_select = interaction_model.target_select
    args.option.inter_select = interaction_model.inter_select
    args.option.interaction_model = interaction_model
    models.terminate_reward = terminate_reward[args.option.term_form](**args.option)

    # converts continuous actions to goals for the next level option
    args.action.discrete_params = parent_option.action_map.filtered_active_set
    mapped_norm = MappedNorm(environment.object_range, environment.object_dynamics, object_names.primary_parent, parent_interaction.active_mask)
    mapped_select = construct_object_selector([object_names.primary_parent], environment, masks=[parent_interaction.active_mask])
    models.action_map = ActionMap(args.action, parent_interaction.mask.filtered_active_set,
     mapped_norm, mapped_select, environment.num_actions if object_names.primary_parent == "Action" and environment.discrete_actions else 0)
    

    models.temporal_extension_manager = TemporalExtensionManager(args.option)

    policy = Policy(models.action_map.discrete_actions, models.state_extractor.total_size, models.action_map.policy_action_space, args)
    
    option = Option(args, policy, models, parent_option) if len(args.record.load_checkpoint) == 0 else load_option(args.record.load_checkpoint, object_names.target, device=args.cuda.gpu)
    if len(args.collect.prioritized_replay) > 0: train_buffer = ParamPriorityReplayBuffer(args.collect.buffer_len, stack_num=1, alpha=args.collect.prioritized_replay[0], beta=args.collect.prioritized_replay[1])
    else: train_buffer = ParamReplayBuffer(args.collect.buffer_len, stack_num=1)
    test_buffer = ParamReplayBuffer(args.collect.buffer_len, stack_num=1)

    args.collect.env_reset = environment.self_reset
    hindsight = Hindsight(args, option, interaction_model) if args.hindsight.use_her else None
    train_collector = OptionCollector(policy, environment, train_buffer, args.policy.epsilon_random > 0, option, hindsight, False, interaction_model.multi_instanced, record, args)
    test_collector = OptionCollector(policy, environment, test_buffer, False, option, None, True, interaction_model.multi_instanced, None, args)

    graph.nodes[object_names.target].option = option
    train_logger = RLLogger(object_names.target + "_train", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.train_log_maxlen, args.record.log_filename)
    test_logger = RLLogger(object_names.target + "_test", args.record.record_rollouts, args.policy.logging.log_interval, args.policy.logging.test_log_maxlen)
    initial_logger = RLLogger(object_names.target + "_initial", args.record.record_rollouts, args.policy.logging.log_interval, 2)
    trainRL(args, train_collector, test_collector, option, graph, (train_logger, test_logger, initial_logger))

    graph.save(args.record.save_dir)


