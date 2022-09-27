# train combined
import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Causal.Utils.weighting import get_weights, passive_binary
from Causal.Utils.get_error import error_types, get_error, compute_error, check_proximity
from Causal.Training.train_combined import train_combined
from Causal.Training.train_full import initialize_optimizer
from Causal.dummy_interaction import DummyInteraction
from Network.network_utils import pytorch_model, run_optimizer
from Hyperparam.read_config import read_config
from State.object_dict import ObjDict
from tianshou.data import Batch

class InlineTrainer():
    def __init__(self, args, interaction_model, terminate_reward):
        self.train = len(args.interaction_config) > 0
        self.interaction_model = interaction_model
        if len(args.interaction_config) > 0: 
            self.interaction_args = read_config(args.interaction_config)  
            self.interaction_args.train.num_iters = args.inpolicy_iters
            self.interaction_args.inter.intrain_passive = args.policy_intrain_passive
            self.interaction_args.inter.weighting = args.intrain_weighting
            self.interaction_args.inter.active.inline_iters = args.policy_inline_iters
        else: self.interaction_args = ObjDict()
        self.inline_iters = args.policy_inline_iters
        self.weighting = args.intrain_weighting
        self.schedule = args.inpolicy_schedule
        self.train_passive = args.policy_intrain_passive
        self.terminate_reward = terminate_reward
        self.reset_weights = args.reset_weights
        self.inpolicy_times = args.inpolicy_times
        self.init_optimizers()

    def init_optimizers(self):
        if self.schedule > 0:
            self.active_optimizer = initialize_optimizer(self.interaction_model.active_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.lr)
            self.passive_optimizer = initialize_optimizer(self.interaction_model.passive_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.lr)
            self.interaction_optimizer = initialize_optimizer(self.interaction_model.interaction_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.alt_lr)

    def set_values(self, data):
        tc_start = time.time()
        proximity = check_proximity(self.interaction_model, data.parent_state, data.target)#compute_error(self.interaction_model, error_types.PROXIMITY, data, prenormalize=True)
        proximity_inst = compute_error(self.interaction_model, error_types.PROXIMITY, data, prenormalize=True, reduced=False) if self.interaction_model.multi_instanced else copy.deepcopy(proximity) # the same as above if not multiinstanced
        tc_proximity = time.time()
        done = data.done#compute_error(self.interaction_model, error_types.DONE, data) # the same as above if not multiinstanced
        tc_done = time.time()
        if type(self.interaction_model) == DummyInteraction or not self.train: # since these values are expensive, don't use them if not training
            binaries = np.ones(proximity.shape)
        else:
            new_data = Batch(target = self.interaction_model.norm(data.target), target_diff = self.interaction_model.norm(data.target_diff, form="diff"), next_target = self.interaction_model.norm(data.next_target))
            tc_create = time.time()
            passive_error = - pytorch_model.unwrap(self.interaction_model.passive_likelihoods(new_data)[-1].sum().unsqueeze(-1).unsqueeze(-1)) # TODO: fails for multi-instanced
            tc_error = time.time()
            # passive_error = -compute_error(self.interaction_model, error_types.PASSIVE_LIKELIHOOD, data, prenormalize=True)
            binaries = passive_binary(passive_error, self.weighting, proximity, done)
            tc_bin = time.time()
            # print(f"bin calc create {tc_create - tc_done} error {tc_error - tc_create} bin {tc_bin -tc_error}")
        tc_binaries = time.time()
        # print(f"inline: prox: {tc_proximity - tc_start}, done: {tc_done - tc_proximity}, binaries: {tc_binaries - tc_done}, total: {tc_binaries -tc_start}")
        # print(data.full_state.factored_state.Paddle, data.full_state.factored_state.Ball,data.next_full_state.factored_state.Ball, proximity, binaries)
        return proximity, proximity_inst, binaries

    def run_train(self, i, rollouts):
        if self.schedule > 0 and i % self.schedule == 0 and i != 0 and self.inpolicy_times > 0 and i // self.schedule <= self.inpolicy_times:
            # resets the network
            if self.reset_weights[0] == 1:
                self.interaction_model.reset_network("interaction")
                self.interaction_optimizer = initialize_optimizer(self.interaction_model.interaction_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.alt_lr)
            if self.reset_weights[1] == 1:
                self.interaction_model.reset_network("active")
                self.active_optimizer = initialize_optimizer(self.interaction_model.active_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.lr)
            if self.reset_weights[2] == 1:
                self.interaction_model.reset_network("passive")
                self.passive_optimizer = initialize_optimizer(self.interaction_model.passive_model, self.interaction_args.interaction_net.optimizer, self.interaction_args.interaction_net.optimizer.lr)
            # change interaction model values
            cut_rols = rollouts[:len(rollouts)]
            # print(cut_rols.weight_binary[-200:], cut_rols.proximity[-200:])
            train_combined(self.interaction_model, rollouts, None, self.interaction_args,
                rollouts.trace, get_weights(self.weighting[2], cut_rols.weight_binary).squeeze(),
                get_weights(self.weighting[1], cut_rols.weight_binary).squeeze(), cut_rols.proximity, cut_rols.proximity_inst,
                self.active_optimizer, self.passive_optimizer, self.interaction_optimizer,
                normalize=True)

            # update inter, rew, terminate, done values
            term, rew, done, inter, cutoffs = self.terminate_reward(cut_rols.inter_state, cut_rols.target, cut_rols.next_target, cut_rols.target_diff, cut_rols.param, cut_rols.mask, cut_rols.true_done, cut_rols.true_reward, reset=False)
            print(term.shape, rew.shape, done.shape, inter.shape)
            # print(term.shape, rew.shape, done.shape, inter.shape)
            rollouts.terminate[:len(rollouts)], rollouts.rew[:len(rollouts)], rollouts.done[:len(rollouts)], rollouts.inter[:len(rollouts)] = \
                np.expand_dims(term, 1), np.expand_dims(rew, 1), np.expand_dims(done, 1), np.expand_dims(inter, 1)
            if self.train_passive: # also need to update the passive weights if we changed the passive model
                passive_error = get_error(self.interaction_model, rollouts, error_types.PASSIVE, prenormalize=True)
                rollouts.weight_binary = passive_binary(passive_error, self.weighting, cut_rols.proximity)
