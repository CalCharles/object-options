from Graph.graph import Graph, load_graph
from arguments import get_args
import torch
import os
import numpy as np
from train_interaction import init_names
from Testing.small_test import small_test
from Testing.environment_test import test_environment_sequence, test_collect_environment, test_environment_setting
from Testing.normalization_test import generate_norm_inputs, generate_mapped_dicts, generate_random_dicts, test_normalization
from Testing.network_test import generate_networks, test_network_construction
from Testing.extractor_test import load_trajectory, generate_extractor, test_state_extraction_environment, test_state_extraction_single
from Testing.interaction_test import generate_interaction, test_interaction_env
from Testing.action_map_test import generate_action_map, test_action_map_env
from Testing.terminate_test import generate_terms, test_reward_terminate_done
from Testing.option_test import generate_option, test_option, generate_option_data
from Testing.policy_test import generate_policy, test_policy
from Testing.collect_test import generate_collect, test_collector
from Testing.log_test import generate_test_logging, test_logging

NUM_ITERS = 100
LOG_INTERVAL = 10
BATCH_SIZE = 64

if __name__ == '__main__':
    args = get_args()
    print(args)
    torch.manual_seed(args.torch.torch_seed)
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if args.debug.run_test == "small_test":    
        small_test(args, file_path, discrete_actions, total_size, policy_action_space, 100, 10, 32)
    elif args.debug.run_test == "environment":
        if args.debug.collect_mode:
            test_collect_environment(args)
        else:
            sequence = test_environment_sequence(args)
            setting = test_environment_setting(args)
    elif args.debug.run_test == "construct_trajectory":
        load_trajectory(args)
    elif args.debug.run_test == "construct_option":
        generate_option_data(args)
    elif args.debug.run_test == "normalization":
        if args.environment.env is not None:
            if args.debug.collect_mode:
                full_inputs, targets, mapped_inputs, mapped_normed, norm_selectors = generate_norm_env_inputs(args)
            else:
                normed_diffs, denormed_diffs, mapped_norm_diffs, mapped_denorm_diffs = test_normalization_env(args)
        if args.debug.collect_mode:
            print("collect mode")
            for i in range(10):
                names, dicts, dims = generate_random_dicts(i)
                target,mask = generate_mapped_dicts(names, dims, i)
                full_inputs, targets, mapped_inputs, mapped_normed, selectors = generate_norm_inputs(names, dicts, dims, target,mask, i)
        else:
            for i in range(10):
                test_normalization(i)
    elif args.debug.run_test == "extractor":
        obs_settings = [
            [[1,1,1,1,1,1], [0,0,0,0]],
            [[1,1,1,1,1,1], [1,1,1,1]],
            [[0,0,0,0,0,0], [1,1,1,1]],
        ]
        if args.environment.env is not None:
            if args.debug.collect_mode:
                    for setting, (single_obs_setting, relative_obs_setting) in enumerate(obs_settings):
                        values = generate_extractor(args, single_obs_setting, relative_obs_setting, setting, 0, normed=False)
            else:
                    for setting, (single_obs_setting, relative_obs_setting) in enumerate(obs_settings):
                        test_state_extraction_environment(args, single_obs_setting, relative_obs_setting, setting)

        else:
            if args.debug.collect_mode:
                for n_setting in range(10):
                    for setting, (single_obs_setting, relative_obs_setting) in enumerate(obs_settings):
                        values = generate_extractor(args, single_obs_setting, relative_obs_setting, setting, n_setting, normed=True)
                        print("VALUES", values)
            else:
                for n_setting in range(10):
                    for setting, (single_obs_setting, relative_obs_setting) in enumerate(obs_settings):
                        test_state_extraction_single(args, single_obs_setting, relative_obs_setting, setting, n_setting)
    elif args.debug.run_test == "network":
        if args.environment.env is not None:
            if args.debug.collect_mode:
                for i in range(10):
                    generated_nets = generate_networks(args, i, normed=False)
            else:
                for i in range(10):
                    results = test_network_construction(args)
        else:
            if args.debug.collect_mode:
                for i in range(10):
                    generated_nets = generate_networks(args, i, normed=True)
            else:
                for i in range(10):
                    results = test_network_construction(args)
    elif args.debug.run_test == "sample":
        if args.debug.collect_mode:
            generated_samples = create_samples(args)
        else:
            results = test_masking_env(args)
    elif args.debug.run_test == "interaction":
        if args.debug.collect_mode:
            generated_interaction = generate_interaction(args)
        else:
            results = test_interaction_env(args)
    elif args.debug.run_test == "action":
        # TODO: NEEDS SEPARATE TESTS FOR ACTION AND NON-ACTION ROOTED, PARAM VERSION, differentiated using configs
        if args.debug.collect_mode:
            generated_samples = generate_action_map(args)
        else:
            results = test_action_map_env(args)
    elif args.debug.run_test == "terminate":
        if args.debug.collect_mode:
            generated_terms = generate_terms(args)
        else:
            results = test_reward_terminate_done(args)
    elif args.debug.run_test == "option":
        if args.debug.collect_mode:
            option, random_option = generate_option(args)
        else:
            option, random_option = test_option(args)
    elif args.debug.run_test == "collect":
        if args.debug.collect_mode:
            losses, vals = generate_collect(args)
        else:
            fixed,unfixed= test_collector(args)
    elif args.debug.run_test == "policy":
        if args.debug.collect_mode:
            losses, vals = generate_policy(args)
        else:
            learn,forward= test_policy(args)
    elif args.debug.run_test == "logging":
        if args.debug.collect_mode:
            losses, train_log_string, test_log_string = generate_test_logging(args)
        else:
            train_match,test_match= test_logging(args)