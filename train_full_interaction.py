import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from State.object_dict import ObjDict
from Buffer.train_test_buffers import generate_buffers
from Causal.FullInteraction.full_interaction_model import NeuralFullInteractionForwardModel, regenerate
from Causal.FullInteraction.Training.train_full import train_full
from Causal.Training.test_full import test_full, test_full_train

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
    extractor, normalization = regenerate()
    full_model = {n: FullNeuralInteractionForwardModel(args, n, environment, extractor, normalization) for n in environment.object_names}
    args.target_select, args.inter_select = extractor.get_selectors()

    # get the train and test buffers
    if len(args.inter.load_intermediate) > 0: train_full_buffer, test_full_buffer, train_object_buffers, test_object_buffers = load_from_pickle(os.path.join(args.inter.load_intermediate,environment.name + "_" + full_model.name + "_traintest.pkl"))
    else: train_full_buffer, test_full_buffer, train_object_buffers, test_object_buffers = generate_buffers(environment, args, args.object_names, full_model, full=True)
    if len(args.inter.save_intermediate) > 0: save_to_pickle(os.path.join(args.inter.save_intermediate, environment.name + "_" + full_model.name + "_traintest.pkl"), (train_full_buffer, test_full_buffer, train_object_buffers, test_object_buffers))

    for name in environment.object_names:
        if args.train.train: train_full(full_models[name], train_buffer, train_object_buffers[name], test_buffer, test_object_buffers[name], args, object_names, environment)
        elif len(args.record.load_dir) > 0: full_models[name] = torch.load(os.path.join(args.record.load_dir, full_model.name + "_inter_model.pt"))
        test_full_train(full_model, train_buffer, args, args.object_names, environment)
        test_full(full_model, test_buffer, args, args.object_names, environment)