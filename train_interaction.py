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

def generate_buffers(args):
    # load data
    data = read_obj_dumps(args.record_rollouts, i=-1, rng = args.num_frames, filename='object_dumps.txt')

    # get the buffers
    buffer = fill_buffer(data, args)

    # fill the train/test buffer
    train_indices = np.random.choice(int(len(buffer) * args.train_ratio), replace=False)
    test_indices = [i for i in range(len((buffer))) if i not in train_indices]
    train_buffer = buffer[train_indices]
    test_buffer = buffer[test_indices]
    del buffer
    return train_buffer, test_buffer


if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment = initialize_environment(args, set_save=False)

    # create a dictionary for the names
    object_names = ObjDict()
    object_names.target = args.train_edge[-1]
    object_names.primary_parent = args.train_edge[0]
    object_names.parents = args.train_edge[:-1]

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.target_select = construct_object_selector([object_names.target], environment)
    args.parent_selectors = {p: construct_object_selector(p, environment) for p in object_names.parents}
    args.parent_select = construct_object_selector(object_names.parents, environment)
    args.inter_select = construct_object_selector(object_names.parents + [object_names.target], environment)
    args.controllable = None # this is filled in with controllable features of the target

    # get the train and test buffers
    if args.load_intermediate: train_buffer, test_buffer = load_from_pickle("/hdd/datasets/counterfactual_data/temp/rollouts.pkl")
    else: train_buffer, test_buffer = generate_buffers(args)

    # initialize the full model      
    full_model = NeuralInteractionForwardModel(**args)

    train_full(full_model, rollouts, test_rollout, args, object_names, environment)
    test_full(full_model, rollouts, test_rollout, args, object_names, environment)