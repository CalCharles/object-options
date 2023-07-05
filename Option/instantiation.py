import numpy as np
from Causal.Sampling.sampling import samplers
from State.state_extractor import StateExtractor
from State.feature_selector import construct_object_selector
from Option.General.rew_term_fns import terminate_reward
from Option.action_map import ActionMap
from Environment.Normalization.norm import MappedNorm
from Buffer.buffer import ParamPriorityReplayBuffer, ParamReplayBuffer, ParamPrioWeightedReplayBuffer, ParamWeightedReplayBuffer
from Causal.dummy_interactions import dummy_interactions

def instantiate_buffers(args, models):
    print(args.collect.prioritized_replay)
    if len(args.collect.prioritized_replay) > 0:
        if models.inline_trainer.train: train_buffer = ParamPrioWeightedReplayBuffer(args.collect.buffer_len, stack_num=1, alpha=args.collect.prioritized_replay[0], beta=args.collect.prioritized_replay[1])
        else: train_buffer = ParamPriorityReplayBuffer(args.collect.buffer_len, stack_num=1, alpha=args.collect.prioritized_replay[0], beta=args.collect.prioritized_replay[1])
    else:
        if models.inline_trainer.train: train_buffer = ParamWeightedReplayBuffer(args.collect.buffer_len, stack_num=1)
        else: train_buffer = ParamReplayBuffer(args.collect.buffer_len, stack_num=1)
    test_buffer = ParamReplayBuffer(args.collect.buffer_len, stack_num=1)
    return train_buffer, test_buffer


def instantiate_action_map(args, parent_option, parent_interaction, environment, object_names):
    args.action.discrete_params = parent_option.action_map.filtered_active_set
    discrete_primitive = environment.num_actions if object_names.primary_parent == "Action" and environment.discrete_actions else 0
    if object_names.primary_parent == "Action" and not environment.discrete_actions: discrete_primitive = -1
    mapped_norm = MappedNorm(environment.object_range, environment.object_dynamics, object_names.primary_parent, parent_interaction.active_mask)
    mapped_select = construct_object_selector([object_names.primary_parent], environment, masks=[parent_interaction.active_mask])
    filtered_active_set = [fas[parent_interaction.active_mask.astype(bool)] for fas in parent_interaction.mask.filtered_active_set]
    diff = mapped_norm.mapped_lim[1] - mapped_norm.mapped_lim[0]
    round_values = None if not args.action.round_values else [np.linspace(mapped_norm.mapped_lim[0][i], mapped_norm.mapped_lim[1][i], int(diff[i] // mapped_norm.mapped_dynamics[i]) + 1) for i in range(len(mapped_norm.mapped_dynamics))]
    no_scale_last = True if environment.name == "Asteroids" and object_names.target == "Laser" else False
    sample_angle = True if environment.name == "Asteroids" and object_names.target == "Laser" else False
    action_map = ActionMap(args.action, filtered_active_set, mapped_norm, mapped_select, 
                    discrete_primitive, round_values, no_scale_last, sample_angle)
    return action_map


def instantiate_rtd(args, interaction_model):
    args.option.interaction_model = interaction_model
    return terminate_reward[args.option.term_form](**args.option)

def instantiate_interaction(args, graph, environment, object_names):
    # get the parent option, the masking function should create the option
    parent_option = graph.nodes[object_names.primary_parent].option
    parent_interaction = graph.nodes[object_names.primary_parent].interaction

    # gets the interaction model for the current control object
    if len(args.train.dummy) != 0: # train in dummy mode
        print(args.train.dummy)
        interaction_model = dummy_interactions[args.train.dummy](args, object_names, environment, environment.object_sizes[object_names.target])
    else:
        interaction_model = graph.nodes[object_names.target].interaction
        interaction_model.regenerate(environment)
        if hasattr(interaction_model, "test") and interaction_model.test is not None: interaction_model.test.set_test_binaries(args.inter.interaction_testing)
    return parent_option, parent_interaction, interaction_model

def instantiate_sampler(args, interaction_model, environment):
    args.sample.mask = interaction_model.mask
    args.sample.target_select = interaction_model.target_select
    args.sample.parent_select = interaction_model.parent_select
    args.sample.additional_select = interaction_model.additional_selectors[-1] if len(interaction_model.additional_selectors) > 0 else interaction_model.additional_select
    args.sample.obj_dim = interaction_model.obj_dim
    args.sample.test_sampler = True
    args.sample.num_angles = 0 if environment.name != "Asteroids" else int(2 * np.pi / environment.ship_speed[1] if environment.ship_speed[1] > 0 else 0)
    args.sample.positive = True if environment.name == "Asteroids" and interaction_model.name == "Laser" else False
    args.sample.epsilon_close = args.option.epsilon_close
    return samplers[args.sample.sample_type](**args.sample)

def instantiate_extractor(args, interaction_model, environment, object_names):
    args.extract.inter_extractor = interaction_model.extractor
    args.extract.object_names = object_names
    args.extract.norm = interaction_model.norm
    args.extract.max_target_objects = environment.object_instanced[object_names.target]
    args.extract.max_parent_objects = environment.object_instanced[object_names.primary_parent]
    return StateExtractor(args.extract)
