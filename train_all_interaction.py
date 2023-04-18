import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle, create_directory
from State.object_dict import ObjDict
from Buffer.train_test_buffers import generate_buffers
from Causal.AllInteraction.all_interaction_model import AllNeuralInteractionForwardModel, regenerate
from Causal.FullInteraction.Training.full_train import train_full, run_train_passive, run_train_interaction
from Causal.FullInteraction.Training.full_test import test_full # TODO: all interaction might want a separate one
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

    # initialize the all model
    extractor, normalization = regenerate(args.full_inter.object_id, environment, all=True)
    all_model = AllNeuralInteractionForwardModel(args, "", environment, extractor, normalization)
    args.pad_size = extractor.object_size
    args.target_select, args.inter_select = extractor.get_selectors(all=True)

    # get the train and test buffers
    if len(args.inter.load_intermediate) > 0: train_all_buffer, test_all_buffer = load_from_pickle(os.path.join(args.inter.load_intermediate,environment.name + "_traintest.pkl"))
    else: train_all_buffer, test_all_buffer = generate_buffers(environment, args, environment.object_names, all_model, full=2)
    if len(args.inter.save_intermediate) > 0: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_traintest.pkl"), (train_all_buffer, test_all_buffer))

    passive_weights = dict()
    if len(args.inter.load_intermediate) > 0: 
        print("loaded model")
        all_model = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_inter_model.pkl"))
        for all_model in all_models.values():
            all_model.cpu().cuda(device = args.torch.gpu)
        passive_weights = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_passive_weights.pkl"))
    # training the passive models
    if args.train.train and args.inter.passive.passive_iters > 0: outputs, passive_weights = run_train_passive(all_model, train_all_buffer, None, test_all_buffer, None, args, environment)
    # saving the passive models and weights
    if len(args.inter.save_intermediate) > 0:
        save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_inter_model.pkl"), all_model)
        save_to_pickle(os.path.join(args.inter.save_intermediate, environment.name +  "_passive_weights.pkl"), passive_weights)
    # pretraining with the true traces, not used for the main algorithm
    if args.train.train and args.inter.interaction.interaction_pretrain > 0: run_train_interaction(all_model, train_all_buffer, None, test_all_buffer, None, args, environment)
    
    # training the active and interaction models
    extractor, normalization = regenerate(args.full_inter.object_id, environment, all=True)
    all_model.regenerate(extractor, normalization, environment)
    
    if args.train.train: train_full(all_model, train_all_buffer, None, test_all_buffer, None, args, environment)
    test_full(all_model, test_all_buffer, args, environment)
