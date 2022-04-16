# primary train operator
import numpy as np
import os, cv2, time, copy
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from file_management import save_to_pickle, load_from_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer
from DistributionalModels.InteractionModels.InteractionTraining.traces import get_proximal_indexes, generate_interaction_trace, adjust_interaction_trace
from DistributionalModels.InteractionModels.InteractionTraining.train_passive import train_passive
from DistributionalModels.InteractionModels.InteractionTraining.train_interaction import train_interaction
from DistributionalModels.InteractionModels.InteractionTraining.train_combined import train_combined
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import get_target_magnitude, get_prediction_error, get_error
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import get_weights, run_optimizer, get_targets
from Networks.input_norm import InterInputNorm, PointwiseNorm
from EnvironmentModels.environment_normalization import hardcode_norm, position_mask

from Rollouts.rollouts import ObjDict, merge_rollouts

def initialize_optimizer(model, train_args, lr):
    return optim.Adam(model.parameters(), lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)


def load_intermediate(train_args, full_model)
    if train_args.compare_trace or train_args.interaction_iters > 0:
        trace = load_from_pickle("data/temp/trace.pkl").cpu().cuda()
        if train_args.max_distance_epsilon > 0:
            nonproximal_trace = pytorch_model.unwrap(trace)-proximal
            nonproximal_trace[nonproximal_trace < 0 ] = 0
    else:
        trace, nonproximal_trace = None, None

    # move the model onto the right device, and cuda
    def load_model(pth):
        model = torch.load(pth)
        model.cpu()
        model.cuda()

    # load models, skip if not using a certain model
    passive_model = load_model("data/temp/passive_model.pt")
    interaction_model = load_model("data/temp/interaction_model.pt") if train_args.interaction_iters > 0 else full_model.interaction_model
    forward_model = load_model("data/temp/active_model.pt") if train_args.pretrain_active else full_model.forward_model
    return trace, nonproximal_trace, passive_model, forward_model, interaction_model

def train_full(full_model, rollouts, test_rollout, train_args, object_names, environment):
    '''
    Train the passive model, interaction model and active model
    @param control is the name of the object that we have control over
    @param controllers is the list of corresponding controllable feature selectors for object @param control 
    @param target_name is the name of the object that we want to control using @param control
    '''
    # define names
    full_model.target = train_args.target_selector
    full_model.parents = train_args.parent_selector
    full_model.inter = train_args.inter_selector
    
    # initialize the optimizers
    active_optimizer = initialize_optimizer(full_model.forward_model, train_args, train_args.lr)
    passive_optimizer = initialize_optimizer(full_model.passive_model, train_args, train_args.lr)
    interaction_optimizer = initialize_optimizer(full_model.interaction_model, train_args, train_args.critic_lr)

    # construct proximity batches if necessary
    proximal, non_proximal = get_proximal_indexes(full_model, train_args.position_mask, rollouts, train_args.proximity_epsilon)
    non_proximal_weights = non_proximal / np.sum(non_proximal)

    train_passive(full_model, rollouts, train_args, active_optimizer, passive_optimizer, weights=non_proximal_weights)

    # saving the intermediate model in the case of crashing during subsequent phases
    if train_args.save_intermediate and train_args.pretrain_iters > 0:
        torch.save(full_model.passive_model.cpu(), "data/temp/passive_model.pt")
        torch.save(full_model.forward_model.cpu(), "data/temp/active_model.pt")

    # generate the trace
    trace = None
    if train_args.interaction_iters > 0 or train_args.compare_trace:
        if train_args.env != "RoboPushing":
            if train_args.load_intermediate: trace = load_from_pickle("data/temp/trace.pkl").cpu().cuda()
            else: trace = generate_interaction_trace(full_model, rollouts, [control_name], [target_name]) # if target is multi-instanced, trace is per-instance
            if train_args.save_intermediate:
                save_to_pickle("data/temp/trace.pkl", trace)

    # train the interaction model with true interaction "trace" values
    train_interaction(full_model, rollouts, train_args, trace, interaction_optimizer)

    if train_args.save_intermediate and train_args.interaction_iters > 0:
        torch.save(full_model.interaction_model, "data/temp/interaction_model.pt")

    if args.load_intermediate: trace, nonproximal_trace, passive_model, forward_model, interaction_model = load_intermediate(train_args, full_model)

    # sampling weights, either wit hthe passive error or if we can upweight the true interactions
    if train_args.passive_weighting > 0:
        passive_error_all = get_prediction_error(full_model, rollouts)
        # passive_error_all = full_model.interaction_model(full_model.gamma(rollouts.get_values("state")))
        # passive_error = pytorch_model.wrap(trace)
        weights, use_weights, total_live, total_dead, ratio_lambda = get_weights(passive_error_all, ratio_lambda = train_args.passive_weighting, 
            passive_error_cutoff=train_args.passive_error_cutoff, temporal_local=train_args.interaction_local, use_proximity=proximal)
        print(use_weights[:100])
        print(use_weights[100:200])
        print(use_weights[200:300])
    elif train_args.interaction_iters > 0:
        print("compute values")
        passive_error_all = trace.clone()
        trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        print(trw.sum())
        weights, use_weights, total_live, total_dead, ratio_lambda = get_weights(ratio_lambda=train_args.interaction_weight, weights=trw, temporal_local=train_args.interaction_local, use_proximity=proximal)
        use_weights =  copy.deepcopy(use_weights)
        print(use_weights.shape)
    elif train_args.change_weighting > 0:
        target_mag = get_target_mag(get_targets, full_model.predict_dynamics, rollouts)
        weights, use_weights, total_live, total_dead, ratio_lambda = get_weights(target_mag, ratio_lambda = train_args.change_weighting, passive_error_cutoff=train_args.passive_error_cutoff)

    else: # no weighting o nthe samples
        passive_error_all = torch.ones(rollouts.filled)
        weights, use_weights = np.ones(rollouts.filled) / rollouts.filled, np.ones(rollouts.filled) / rollouts.filled
        print(rollouts.filled, weights.shape, weights)
        total_live, total_dead = 0, 0
        ratio_lambda = 1

    # handling boosting the passive operator to work with upweighted states
    # boosted_passive_operator = copy.deepcopy(full_model.passive_model)
    # true_passive = full_model.passive_model
    # full_model.passive_model = boosted_passive_operator
    passive_optimizer = optim.Adam(full_model.passive_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
    print("combined", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

    train_combined(full_model, rollouts, test_rollout, train_args, batchvals, half_batchvals, 
        trace, weights, use_weights, passive_error_all, interaction_schedule, ratio_lambda,
        active_optimizer, passive_optimizer, interaction_optimizer, proximal=proximal)        # if args.save_intermediate:
    full_model.save(train_args.save_dir)
    if train_args.interaction_iters > 0:
        compute_interaction_stats(rollouts, trace = trace, passive_error_cutoff=train_args.passive_error_cutoff)
    del full_model.nf, full_model.rv