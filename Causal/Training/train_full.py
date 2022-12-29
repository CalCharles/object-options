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
from Record.file_management import create_directory

def initialize_optimizer(model, args, lr):
    return optim.Adam(model.parameters(), lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)


def load_intermediate(args, full_model, environment):
    # move the model onto the right device, and cuda
    def load_model(pth):
        model = torch.load(pth)
        model.cpu()
        model.cuda()
        return model

    # load models, skip if not using a certain model
    passive_model = load_model(os.path.join(args.inter.load_intermediate, environment.name + "_" + full_model.name + "_passive_model.pt"))
    interaction_model = load_model(os.path.join(args.inter.load_intermediate, environment.name + "_" + full_model.name + "_interaction_model.pt")) if args.inter.interaction.interaction_pretrain > 0 else full_model.interaction_model
    active_model = load_model(os.path.join(args.inter.load_intermediate, environment.name + "_" + full_model.name + "_active_model.pt")) if args.inter.passive.pretrain_active else full_model.active_model

    active_optimizer = initialize_optimizer(active_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr)
    passive_optimizer = initialize_optimizer(passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr * 0.1)
    interaction_optimizer = initialize_optimizer(interaction_model, args.interaction_net.optimizer, args.interaction_net.optimizer.alt_lr)
    return passive_model, active_model, interaction_model, active_optimizer, passive_optimizer, interaction_optimizer

def train_full(full_model, rollouts, test_rollout, args, object_names, environment):
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

    # construct proximity batches if necessary
    proximal = get_error(full_model, rollouts, error_type=error_types.PROXIMITY, normalized=True).astype(int)
    proximal_inst = get_error(full_model, rollouts, error_type=error_types.PROXIMITY, reduced=False, normalized=True).astype(int) # the same as above if not multiinstanced
    non_proximal = (proximal != True).astype(int)
    non_proximal_inst = (proximal_inst != True).astype(int)
    non_proximal_weights = non_proximal.squeeze() / np.sum(non_proximal) if np.sum(non_proximal) != 0 else np.ones(non_proximal.shape) / len(non_proximal)

    # Proximity for Test rollouts
    test_proximal = get_error(full_model, test_rollout, error_type=error_types.PROXIMITY, normalized=True).astype(int)
    test_proximal_inst = get_error(full_model, test_rollout, error_type=error_types.PROXIMITY, reduced=False, normalized=True).astype(int) # the same as above if not multiinstanced

    train_passive(full_model, rollouts, args, active_optimizer, passive_optimizer, weights=non_proximal_weights if full_model.proximity_epsilon > 0 else None)

    del passive_optimizer
    passive_optimizer = initialize_optimizer(full_model.passive_model, args.interaction_net.optimizer, args.interaction_net.optimizer.lr * 0.05)

    # saving the intermediate model in the case of crashing during subsequent phases
    if len(args.inter.save_intermediate) > 0 and args.inter.passive.passive_iters > 0:
        full_model.cpu()
        torch.save(full_model.passive_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_passive_model.pt"))
        torch.save(full_model.active_model, os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_active_model.pt"))
        full_model.cuda()
    # generate the trace
    trace = rollouts.trace if args.inter.interaction.interaction_pretrain > 0 or args.inter.compare_trace else None

    # train the interaction model with true interaction "trace" values
    if len(args.inter.load_intermediate) > 0: full_model.passive_model, full_model.active_model, full_model.interaction_model, active_optimizer, passive_optimizer, interaction_optimizer = load_intermediate(args, full_model, environment)
    passive_error, active_weights, binaries = separate_weights(args.inter.active.weighting, full_model, rollouts, proximal, trace if args.inter.interaction.interaction_pretrain > 0 else None)
    train_interaction(full_model, rollouts, args, trace, interaction_optimizer, weights = active_weights)
    if args.inter.save_intermediate and args.inter.interaction.interaction_pretrain > 0:
        torch.save(full_model.interaction_model, os.path.join(args.inter.save_intermediate, environment.name + "_" + full_model.name + "_interaction_model.pt"))

    # sampling weights, either wit hthe passive error or if we can upweight the true interactions
    interaction_weights = get_weights(args.inter.active.weighting[2], rollouts.weight_binary)
    # handling boosting the passive operator to work with upweighted states
    # boosted_passive_operator = copy.deepcopy(full_model.passive_model)
    # true_passive = full_model.passive_model
    # full_model.passive_model = boosted_passive_operator
    # passive_optimizer = optim.Adam(full_model.passive_model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
    # print("combined", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

    train_combined(full_model, rollouts, test_rollout, args,
                        trace, active_weights, interaction_weights, proximal, proximal_inst,
                        active_optimizer, passive_optimizer, interaction_optimizer)
    if len(args.record.save_dir) > 0: full_model.save(args.record.save_dir)