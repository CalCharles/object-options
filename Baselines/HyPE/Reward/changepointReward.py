import numpy as np
import copy, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Record.file_management import load_from_pickle


def load_reward(load_dir, name):
    return load_from_pickle(os.path.join(load_dir, name + "_reward.pkl"))

class ChangepointDetectionReward():
    def __init__(self, object_names, model, desired_modes, reward_base, param_reward, changepoint_reward, extractor, norm):
        self.name = object_names.target
        self.model = model
        self.names = object_names
        self.true_num_modes = self.model.num_modes
        self.num_modes = self.model.num_modes
        self.parameter_minmax = [np.array([0]), np.array([84])] # TODO: where does this come from?
        self.desired_modes = desired_modes
        self.changepoint_reward = changepoint_reward
        self.reward_base = reward_base
        self.param_reward = param_reward
        self.extractor = extractor
        self.norm = norm
        self.one_mode = False

    def toggle_one_mode(self, one_mode):
        # one mode uses only a single mode for reward assignment---every nonnegative assignment of mode gets reward
        if one_mode:
            self.one_mode = True
            self.num_modes = 1
        else:
            self.one_mode = False
            self.num_modes = self.true_num_modes

    def set_extractor_norm(self, extractor, norm):
        self.extractor = extractor
        self.norm = norm

    def set_params(self, reward_base, param_reward, changepoint_reward, one_mode):
        self.reward_base = reward_base
        self.param_reward = param_reward
        self.changepoint_reward = changepoint_reward
        self.toggle_one_mode(one_mode)


    def compute_reward(self, target_diff, target_states, parent_states, dones, cached_rewards=None, cached_terminations=None):
        if cached_rewards is not None:
            return cached_rewards, cached_terminations
        assignments, terminations = self.model.get_mode(target_diff, target_states, parent_states, dones)
        
        # print(assignments[:100], self.desired_modes)
        rewards = list()

        if self.one_mode: reward_choice = np.ones(target_states.shape[:-1]) * self.reward_base # if one mode, will just reassign the same reward vector for each desired mode
        for desired_mode in self.desired_modes:
            if not self.one_mode: reward_choice = np.ones(target_states.shape[:-1]) * self.reward_base
            if hasattr(self, "changepoint_reward") and self.changepoint_reward != 0: # TODO: dangerous line because CP reward could be assigned to 0 
                for m in self.desired_modes:
                    reward_choice[assignments == m] = self.changepoint_reward
            reward_choice[assignments == desired_mode] = self.param_reward
            rewards.append(reward_choice)
        if self.one_mode: rewards = [reward_choice]
        return rewards, terminations