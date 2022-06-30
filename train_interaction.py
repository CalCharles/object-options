import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from State.object_dict import ObjDict
from State.feature_selector import construct_object_selector
from Buffer.train_test_buffers import generate_buffers
from Causal.interaction_model import NeuralInteractionForwardModel
from Causal.Training.train_full import train_full
from Causal.Training.test_full import test_full, test_full_train

from Environment.Environments.initialize_environment import initialize_environment

from Network.network_utils import pytorch_model
import numpy as np
import sys
import psutil

def init_names(args):
    object_names = ObjDict()
    object_names.target = args.train.train_edge[-1]
    object_names.primary_parent = args.train.train_edge[0]
    object_names.parents = args.train.train_edge[:-1]
    object_names.additional = [p for p in object_names.parents if p != object_names.primary_parent]
    args.object_names = object_names
    return object_names

if __name__ == '__main__':
    args = get_args()
    print(args) # print out args for records
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)

    # create a dictionary for the names
    object_names = init_names(args)

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.target_select = construct_object_selector([object_names.target], environment)
    args.parent_selectors = {p: construct_object_selector([p], environment) for p in object_names.parents}
    args.additional_select = construct_object_selector(object_names.additional, environment) if len(object_names.additional) > 0 else None
    args.parent_select = construct_object_selector(object_names.parents, environment)
    args.inter_select = construct_object_selector(object_names.parents + [object_names.target], environment)
    args.controllable = None # this is filled in with controllable features of the target

    # initialize the full model
    full_model = NeuralInteractionForwardModel(args, environment)
    
    # get the train and test buffers
    if args.inter.load_intermediate: train_buffer, test_buffer = load_from_pickle("/hdd/datasets/object_data/temp/traintest.pkl")
    else: train_buffer, test_buffer = generate_buffers(environment, args, object_names, full_model)
    if args.inter.save_intermediate: save_to_pickle("/hdd/datasets/object_data/temp/traintest.pkl", (train_buffer, test_buffer))

    if args.train.train: train_full(full_model, train_buffer, test_buffer, args, object_names, environment)
    elif len(args.record.load_dir) > 0: full_model = torch.load(os.path.join(args.record.load_dir, full_model.name + "_inter_model.pt"))
    test_full_train(full_model, train_buffer, args, object_names, environment)
    test_full(full_model, test_buffer, args, object_names, environment)