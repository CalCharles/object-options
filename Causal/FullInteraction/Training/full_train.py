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
from Causal.FullInteraction.Training.full_train_passive import train_passive, get_passive_weights
from Causal.FullInteraction.Training.full_train_trace import train_interaction
from Causal.FullInteraction.Training.full_train_combined import train_combined
from Causal.Training.train_full import load_intermediate
from Record.file_management import create_directory

def initialize_optimizer(model, args, lr):
    return optim.Adam(model.parameters(), lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)

def run_train_passive(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args, environment):
    # initialize the optimizers
    if full_model.cluster_mode: # the active optimizer does NOT optimize the interaction weights
        active_optimizer = optim.Adam(list(full_model.active_model.key_encoding.parameters()) + list(full_model.active_model.query_encoding.parameters()) 
                                        + list(full_model.active_model.means.parameters()) + list(full_model.active_model.stds.parameters()), 
                                        args.interaction_net.optimizer.lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
    else:
        active_optimizer = initialize_optimizer(full_model.active_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
    passive_optimizer = None if full_model.use_active_as_passive else initialize_optimizer(full_model.passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)

    passive_weights = get_passive_weights(args, full_model, object_rollout)
    outputs, passive_weights = train_passive(full_model, args, rollouts, object_rollout, passive_weights, active_optimizer, passive_optimizer)

    # saving the intermediate model in the case of crashing during subsequent phases
    if len(args.inter.save_intermediate) > 0 and args.inter.passive.passive_iters > 0:
        full_model.cpu()
        torch.save(full_model.passive_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_passive_model.pt"))
        torch.save(full_model.active_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_active_model.pt"))
        full_model.cuda()
    del active_optimizer
    del passive_optimizer
    return outputs, passive_weights

def run_train_interaction(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args, environment):
    # initialize the optimizers
    if full_model.cluster_mode:
        interaction_optimizer = optim.Adam(list(full_model.active_model.inter_models.parameters()) + list(full_model.interaction_model.parameters()),
                                            args.interaction_net.optimizer.alt_lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
    else:
        interaction_optimizer = initialize_optimizer(full_model.interaction_model, args.interaction_net.optimizer, args.interaction_net.optimizer.alt_lr)
    if args.full_inter.selection_train == "separate": # only works with the selection-interaction model right now
        if not full_model.cluster_mode:
            interaction_optimizer = optim.Adam(full_model.interaction_model.selection_network.parameters(),
                                                args.interaction_net.optimizer.alt_lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)

    # get weights based on interacting states
    passive_error, active_weights, binaries = separate_weights(args.inter.active.weighting, full_model, rollouts, None, object_rollouts=object_rollout) # trace=trace, object_rollouts=object_rollout)

    outputs, inter_weights = train_interaction(full_model, rollouts, object_rollout, args, interaction_optimizer)

    # saving the intermediate model in the case of crashing during subsequent phases
    if len(args.inter.save_intermediate) > 0 and args.inter.interaction.interaction_pretrain > 0:
        full_model.cpu()
        torch.save(full_model.interaction_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_interaction_model.pt"))
        full_model.cuda()
    del interaction_optimizer
    return outputs, inter_weights

def train_full(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args, environment):
    '''
    Train the passive model, interaction model and active model
    @param control is the name of the object that we have control over
    @param controllers is the list of corresponding controllable feature selectors for object @param control 
    @param target_name is the name of the object that we want to control using @param control
    '''    
    # train the interaction model with true interaction "trace" values
    # train_interaction(full_model, rollouts, args, interaction_optimizer)
    # if args.inter.save_intermediate and args.inter.interaction.interaction_pretrain > 0:
    #     torch.save(full_model.interaction_model, os.path.join(args.inter.save_intermediate, environment.name + "_" + full_model.name + "_interaction_model.pt"))
    if full_model.cluster_mode:
        active_optimizer = optim.Adam(list(full_model.active_model.key_encoding.parameters()) + list(full_model.active_model.query_encoding.parameters()) 
                                        + list(full_model.active_model.means.parameters()) + list(full_model.active_model.stds.parameters()), 
                                        args.interaction_net.optimizer.lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
        passive_optimizer = None if full_model.use_active_as_passive else initialize_optimizer(full_model.passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
        interaction_optimizer = optim.Adam(list(full_model.active_model.inter_models.parameters()) + list(full_model.interaction_model.parameters()),
                                            args.interaction_net.optimizer.alt_lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
    else:
        active_optimizer = initialize_optimizer(full_model.active_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
        passive_optimizer = None if full_model.use_active_as_passive else initialize_optimizer(full_model.passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
        interaction_optimizer = active_optimizer if full_model.attention_mode else initialize_optimizer(full_model.interaction_model, args.interaction_net.optimizer, args.interaction_net.optimizer.alt_lr)
    if args.full_inter.selection_train == "separate": # only works with the selection-interaction model right now
        if not full_model.cluster_mode:
            active_optimizer = optim.Adam(list(full_model.active_model.parameters()) + list(full_model.interaction_model.inter_models.parameters()), 
                                            args.interaction_net.optimizer.lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
            interaction_optimizer = optim.Adam(full_model.interaction_model.selection_model,
                                                args.interaction_net.optimizer.alt_lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)

    # if len(args.inter.load_intermediate) > 0: full_model.passive_model, full_model.active_model, full_model.interaction_model, active_optimizer, passive_optimizer, interaction_optimizer = load_intermediate(args, full_model, environment)
    # sampling weights, either wit hthe passive error or if we can upweight the true interactions
    passive_weights = get_passive_weights(args, full_model, object_rollout)
    proximal = get_error(full_model, rollouts, object_rollout, error_type=error_types.PROXIMITY_FULL, normalized = True)
    passive_error, active_weights, binaries = separate_weights(args.inter.active.weighting, full_model, rollouts, proximal, None, object_rollouts=object_rollout)
    # print(passive_error, binaries)
    # error
    interaction_weights = get_weights(args.inter.active.weighting[2], object_rollout.weight_binary)
    if args.inter.interaction.interaction_pretrain > 0: train_interaction(full_model, rollouts, object_rollout, args, interaction_optimizer)


    train_combined(full_model, rollouts, object_rollout, test_rollout, test_object_rollout, args,
                        passive_weights, active_weights, interaction_weights, proximal, active_optimizer,
                         passive_optimizer, interaction_optimizer)
    if len(args.record.save_dir) > 0: full_model.save(args.record.save_dir)