import os, torch
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from Environments.environment_initializer import initialize_environment
from EnvironmentModels.environment_model import ModelRollouts, FeatureSelector, ControllableFeature

from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen

from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D

from EnvironmentModels.Pushing.pushing_environment_model import PushingEnvironmentModel
from Environments.Pushing.screen import Pushing

from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel

from Counterfactual.counterfactual_dataset import CounterfactualStateDataset
from Counterfactual.passive_active_dataset import HackedPassiveActiveDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph, OptionEdge, graph_construct_load
from Options.option import Option, PrimitiveOption
from DistributionalModels.InteractionModels.interaction_model import default_model_args, load_hypothesis_model
from DistributionalModels.InteractionModels.InteractionTraining.active_determination import determine_active_set, determine_range
from DistributionalModels.InteractionModels.InteractionTraining.assessment_functions import assess_error
from DistributionalModels.InteractionModels.InteractionTraining.sampling import collect_samples
from DistributionalModels.InteractionModels.InteractionTraining.traces import generate_interaction_trace
from DistributionalModels.InteractionModels.feature_explorer import FeatureExplorer
from Networks.network import pytorch_model
import numpy as np
import sys
import psutil

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment = initialize_environment(args, set_save=False)

    # create a dictionary for the names
    object_names = ObjDict()
    object_names.target = args.train_edge[-1]
    object_names.parents = args.train_edge[:-1]
    object_names.inter_parents = args.train_edge[:-1]

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.target_selector = construct_object_selector([object_names.target], environment)
    args.parent_selectors = construct_object_selector(object_names.parents, environment)
    args.inter_selector = construct_object_selector(object_names.inter_parents + [object_names.target], environment)

    # set epsilons (TODO: find out why that matters)
    last_state = None
    for cfs in controllable_feature_selectors:
        if cfs.object() != "Action":
            graph.nodes[cfs.object()].option.policy.set_eps(0)
    
    # commented section BELOW
    if not args.load_intermediate:
        print("generating data")
        print("pre loaded", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
        data = read_obj_dumps(args.record_rollouts, i=-1, rng = args.num_frames, filename='object_dumps.txt')
        print("read", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
        rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
        print("rollouts", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
        i=0
        for data_dict, next_data_dict in zip(data, data[1:]):
            # print(data_dict)
            insert_dict, last_state, skip = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
            if not skip:
                # print(np.array(data_dict["Gripper"] + data_dict["Block"]+ next_data_dict["Block"] + (np.array(next_data_dict["Block"]) - np.array(data_dict["Block"])).tolist() ) )
                # print(np.array(data_dict["Paddle"] + data_dict["Ball"]+ next_data_dict["Ball"]))
                rollouts.append(**insert_dict)
            i += 1
            if i % 10000 == 0:
                print("loaded", i, psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
        if args.save_intermediate:
            # save_to_pickle("data/temp/rollouts.pkl", rollouts)
            save_to_pickle("/hdd/datasets/counterfactual_data/temp/rollouts.pkl", rollouts)
        del data
        print("del", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

    else:
        print("loading intermediate")
        # rollouts = load_from_pickle("data/temp/rollouts.pkl")
        rollouts = load_from_pickle("/hdd/datasets/counterfactual_data/temp/rollouts.pkl")
        if args.cuda:
            rollouts.cuda()

    # Just run selection binary

    if args.train:
        success = False
        if args.load_weights: # retraining an existing model
            hypothesis_model = load_hypothesis_model(args.dataset_dir) 
            hypothesis_model.cpu()
            hypothesis_model.cuda()
            train, test = rollouts.split_train_test(args.ratio)
            if args.cuda:
                hypothesis_model.to("cuda:"+str(args.gpu))
            hypothesis_model.train(train, args, control=hypothesis_model.control_feature, target_name=hypothesis_model.name.split("->")[1])
            success = True
        else:
            model_args = default_model_args(args.predict_dynamics, args.policy_type) # input and output sizes should not be needed
            model_args.hidden_sizes, model_args.interaction_binary, model_args.interaction_prediction, model_args.init_form, model_args.activation, model_args.interaction_distance = args.hidden_sizes, args.interaction_binary, args.interaction_prediction, args.init_form, args.activation, args.interaction_distance
            model_args['controllable'], model_args['environment_model'] = controllable_feature_selectors, environment_model
            print(model_args)
            feature_explorer = FeatureExplorer(graph, controllable_feature_selectors, environment_model, model_args) # args should contain the model args, might want subspaces for arguments or something since args is now gigantic
            print("rollouts filled", rollouts.filled)

            print("pre train test", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
            if not args.load_intermediate:
                train, test = rollouts.split_train_test(args.ratio)
                train.cpu(), test.cpu()
            else:
                train = load_from_pickle("data/temp/train.pkl")
                test = load_from_pickle("data/temp/test.pkl")
            if args.save_intermediate:
                save_to_pickle("data/temp/train.pkl", train)
                save_to_pickle("data/temp/test.pkl", test)
            print("train test", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
            rollouts.cpu()
            torch.cuda.empty_cache()
            del rollouts
            print("del rols", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))


            exploration = feature_explorer.search(train, test, args) # again, args contains the training parameters, but we might want subsets since this is a ton of parameters more 
            if exploration is None:
                print("FAILED TO LOCATE RELATED FEATURES")
                success = False
            else:
                hypothesis_model, delta, gamma = exploration
                success = True
            # save the cfs
        if success:
            hypothesis_model.cpu()
            hypothesis_model.save(args.save_dir)
    else: # if not training, determine and save the active set selection binary
        hypothesis_model = load_hypothesis_model(args.dataset_dir)
        controllable_feature_used = [cfs for cfs in controllable_feature_selectors if cfs.object() == hypothesis_model.name.split("->")[0]]
        hypothesis_model.control_feature = controllable_feature_used[0] if len(controllable_feature_used) == 1 else controllable_feature_used
        hypothesis_model.active_epsilon = args.active_epsilon
        hypothesis_model.environment_model = environment_model
        hypothesis_model.cpu()
        hypothesis_model.cuda()

        trace = None
        # if args.load_intermediate: trace = load_from_pickle("data/temp/trace.pkl").cpu().cuda()
        # else: trace = generate_interaction_trace(hypothesis_model, rollouts, [hypothesis_model.control_feature.object()], [hypothesis_model.target_name])
        if args.save_intermediate:
            save_to_pickle("data/temp/trace.pkl", trace)
        forward_error, passive_error = assess_error(hypothesis_model, rollouts, passive_error_cutoff=args.passive_error_cutoff, trace=trace)
        passed = forward_error < (passive_error - args.model_error_significance)
        print("comparison", forward_error, passive_error, args.model_error_significance, passed)

        # delta, gamma = hypothesis_model.delta, hypothesis_model.gamma
        # rollouts.cuda()
        # passive_error = hypothesis_model.get_prediction_error(rollouts)
        # weights, use_weights, total_live, total_dead, ratio_lambda = hypothesis_model.get_weights(passive_error, passive_error_cutoff=args.passive_error_cutoff)     
        # # trace = load_from_pickle("data/trace.pkl").cpu().cuda()
        # if args.env != "RoboPushing":
        #     hypothesis_model.compute_interaction_stats(rollouts, passive_error_cutoff=args.passive_error_cutoff)
        # afs = environment_model.construct_action_selector() 
        # controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions],1)]
        # if args.hardcode_norm[0] == "RoboPushing":
        #     hardcode_norm = (np.array([-.31, -.31, .83]), np.array([.10, .21, .915])) 
        determine_active_set(hypothesis_model, rollouts, feature_step=args.feature_step)
        collect_samples(hypothesis_model, rollouts, use_trace=args.interaction_iters > 0)
        hypothesis_model.cpu()
        hypothesis_model.save(args.save_dir)
        print(hypothesis_model.selection_binary)

    # interaction model statistics:
    # False positive (interaction at non-interaction), false negative (non-interaction at interaction), low value (predicting interaction with low confidence)
    # high error forward modeling not working properly, no variance: forward model not sensitive to changes 9or interaction model

    # print("forward", interaction_model.forward_model)
    # print("state", interaction_model.state_model)
    # print("state counts", interaction_model.difference_counts)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)