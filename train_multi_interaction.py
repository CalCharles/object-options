import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle, create_directory
from State.object_dict import ObjDict
from Buffer.train_test_buffers import generate_buffers
from Causal.AllInteraction.all_interaction_model import AllNeuralInteractionForwardModel, regenerate
regenerate_all = regenerate
from Causal.FullInteraction.full_interaction_model import FullNeuralInteractionForwardModel, regenerate
regenerate_full = regenerate
from Causal.FullInteraction.Training.full_train import train_full, run_train_passive, run_train_interaction
from Causal.FullInteraction.Training.full_test import test_full # TODO: all interaction might want a separate one
from Causal.EMFAC.train_EMFAC import train_EMFAC
from Causal.MultiInteraction.evaluate_interactions import evaluate_null_interaction
# from Causal.Training.full_test import test_full, test_full_train

from Environment.Environments.initialize_environment import initialize_environment

from Network.network_utils import pytorch_model
import numpy as np
import sys
import psutil

if __name__ == '__main__': # TODO: combine with the train_all/train_full code
    args = get_args()
    print(args) # print out args for records
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.controllable = None # this is filled in with controllable features of the target
    args.EMFAC.is_emfac = False
    args.full_inter.object_id = args.EMFAC.full_train
    all_train = len(args.EMFAC.full_train) == 0 # only supports full model training for one name at a time, TODO: implement full model training
    if len(args.EMFAC.full_train) > 0: args.full_inter.train_names = [args.EMFAC.full_train]
    extractor, normalization = regenerate_all(True, environment, all=all_train) if all_train else regenerate_full(True, environment, all=all_train)
    args.pad_size = extractor.object_size
    args.target_select, args.inter_select = extractor.get_selectors(all=all_train)

    # initialize the all/full model
    if args.multi_inter.evaluate == 1: # TODO: change this to load from the actual saved, not intermediate
        model = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_inter_model.pkl"))
        model.cpu().cuda(device=args.torch.gpu)         
    else:
        model = AllNeuralInteractionForwardModel(args, "", environment, extractor, normalization) if all_train else FullNeuralInteractionForwardModel(args, args.EMFAC.full_train, environment, extractor, normalization)

    # get the train and test buffers
    if len(args.inter.load_intermediate) > 0: train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout = load_from_pickle(os.path.join(args.inter.load_intermediate,environment.name + "_traintest.pkl"))
    else: 
        if all_train:
            train_all_buffer, test_all_buffer = generate_buffers(environment, args, environment.object_names, model, full=2)
            train_object_rollout, test_object_rollout = None, None  
        else:
            train_all_buffer, train_object_rollouts, test_all_buffer, test_object_rollouts = generate_buffers(environment, args, environment.object_names, model, full=1)
            train_object_rollout, test_object_rollout = train_object_rollouts[args.EMFAC.full_train], test_object_rollouts[args.EMFAC.full_train]
    if len(args.inter.save_intermediate) > 0: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_traintest.pkl"), (train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout))

    if args.multi_inter.evaluate == 1:
        evaluate_null_interaction(model, train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout, args, environment)
    else:
        passive_weights = dict()
        if len(args.inter.load_intermediate) > 0: 
            print("loaded model")
            model = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_inter_model.pkl"))
            model.cpu().cuda(device=args.torch.gpu)
            # for model in model.values():
            #     model.cpu().cuda(device = args.torch.gpu)
            passive_weights = load_from_pickle(os.path.join(args.inter.load_intermediate, environment.name + "_passive_weights.pkl"))
        # training the passive/active models
        if args.train.train and args.inter.passive.passive_iters > 0: outputs, passive_weights = run_train_passive(model, train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout, args, environment)
        if args.multi_inter.evaluate == 2:
            evaluate_null_interaction(model, train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout, args, environment)
        if len(args.inter.save_intermediate) > 0:
            save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name +  "_inter_model.pkl"), model)

