import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle, create_directory
from State.object_dict import ObjDict
from Buffer.train_test_buffers import generate_buffers
from Causal.FullInteraction.full_interaction_model import FullNeuralInteractionForwardModel, regenerate
from Causal.FullInteraction.Training.full_train import train_full, run_train_passive
from Causal.FullInteraction.Training.full_test import test_full
# from Causal.Training.full_test import test_full, test_full_train

from Environment.Environments.initialize_environment import initialize_environment

from Network.network_utils import pytorch_model
import numpy as np
import sys
import psutil

if __name__ == '__main__':
    args = get_args()
    print(args) # print out args for records
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.controllable = None # this is filled in with controllable features of the target

    # initialize the full model
    extractor, normalization = regenerate(args.full_inter.object_id, environment)
    full_models = {n: FullNeuralInteractionForwardModel(args, n, environment, extractor, normalization) for n in environment.object_names if n != "Action"}
    args.target_select, args.inter_select = extractor.get_selectors()
    args.pad_size = extractor.object_size

    # get the train and test buffers
    if len(args.inter.load_intermediate) > 0: train_full_buffer, train_object_buffers, test_full_buffer, test_object_buffers = load_from_pickle(os.path.join(args.inter.load_intermediate,environment.name + "_traintest.pkl"))
    else: train_full_buffer, train_object_buffers, test_full_buffer, test_object_buffers = generate_buffers(environment, args, environment.object_names, list(full_models.values())[0], full=True)
    if len(args.inter.save_intermediate) > 0: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_traintest.pkl"), (train_full_buffer, train_object_buffers, test_full_buffer, test_object_buffers))

    if len(args.inter.load_intermediate) > 0: 
        print("loaded model")
        full_models = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_inter_model.pkl"))
    for name in environment.object_names:
        if name not in ["Action"]:
            print("TRAINING", name)
            if args.train.train and args.inter.passive.passive_iters > 0: run_train_passive(full_models[name], train_full_buffer, train_object_buffers[name], test_full_buffer, test_object_buffers[name], args, environment)
    if len(args.inter.save_intermediate) > 0: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_inter_model.pkl"), full_models)
    for name in environment.object_names:
        if name not in ["Action"]:
            full_models[name].regenerate(extractor, normalization, environment)
    for name in environment.object_names:
        if name not in ["Action", "oqueofr"]:
            print("TRAINING", name)
            if args.train.train: train_full(full_models[name], train_full_buffer, train_object_buffers[name], test_full_buffer, test_object_buffers[name], args, environment)
            test_full(full_models[name], test_full_buffer, test_object_buffers[name], args, environment)
