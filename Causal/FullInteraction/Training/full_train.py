# primary train operator
import numpy as np
import os, cv2, time, copy
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Utils.get_error import get_error, error_types
from Causal.Utils.weighting import separate_weights, get_weights
from Causal.Training.train_passive import train_passive
from Causal.Training.train_trace import train_interaction
from Causal.Training.train_combined import train_combined
from Causal.Training.train_full import load_intermediate
from Record.file_management import create_directory

def initialize_optimizer(model, args, lr):
    return optim.Adam(model.parameters(), lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)

def train_full(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args, object_names, environment):
    '''
    Train the passive model, interaction model and active model
    @param control is the name of the object that we have control over
    @param controllers is the list of corresponding controllable feature selectors for object @param control 
    @param target_name is the name of the object that we want to control using @param control
    '''    
    # initialize the optimizers
    active_optimizer = initialize_optimizer(full_model.active_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
    passive_optimizer = initialize_optimizer(full_model.passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
    interaction_optimizer = initialize_optimizer(full_model.interaction_model, args.interaction_net.optimizer, args.interaction_net.optimizer.alt_lr)

    train_passive(full_model, rollouts, object_rollout, args, active_optimizer, passive_optimizer)

    # saving the intermediate model in the case of crashing during subsequent phases
    if len(args.inter.save_intermediate) > 0 and args.inter.passive.passive_iters > 0:
        full_model.cpu()
        torch.save(full_model.passive_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_passive_model.pt"))
        torch.save(full_model.active_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_active_model.pt"))
        full_model.cuda()

    # train the interaction model with true interaction "trace" values
    train_interaction(full_model, rollouts, args, interaction_optimizer)
    if args.inter.save_intermediate and args.inter.interaction.interaction_pretrain > 0:
        torch.save(full_model.interaction_model, os.path.join(args.inter.save_intermediate, environment.name + "_" + full_model.name + "_interaction_model.pt"))

    if len(args.inter.load_intermediate) > 0: full_model.passive_model, full_model.active_model, full_model.interaction_model, active_optimizer, passive_optimizer, interaction_optimizer = load_intermediate(args, full_model, environment)
    # sampling weights, either wit hthe passive error or if we can upweight the true interactions
    passive_error, active_weights, binaries = separate_weights(args.inter.active.weighting, full_model, rollouts, proximal, trace if args.inter.interaction.interaction_pretrain > 0 else None)
    interaction_weights = get_weights(args.inter.active.weighting[2], rollouts.weight_binary)

    train_combined(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args,
                        active_weights, interaction_weights, proximal, proximal_inst,
                        active_optimizer, passive_optimizer, interaction_optimizer)
    if len(args.record.save_dir) > 0: full_model.save(args.record.save_dir)