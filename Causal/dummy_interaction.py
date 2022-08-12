import os
import torch
import numpy as np
from Record.file_management import create_directory
from State.feature_selector import construct_object_selector
from Environment.Normalization.norm import NormalizationModule
from Causal.Utils.interaction_selectors import CausalExtractor


class ActionMask():
    def __init__(self, discrete_actions, num_actions, action_shape):
        self.filtered_active_set = list()
        self.active_mask = np.ones(action_shape)

class ActionDummyInteraction():
    def __init__(self, action_shape, discrete_actions, num_actions):
        self.name = "Action"
        self.active_mask = np.ones(action_shape)
        self.active_set = [1 for i in range(num_actions)] if discrete_actions else list()
        self.mask = ActionMask(discrete_actions, num_actions, action_shape)

    def save(self, pth):
        torch.save(self, os.path.join(create_directory(pth), self.name + "_inter_model.pt"))

    # no cuda needed for this class, but it might be called
    def cuda(self, device=-1):
        return self

    def cpu(self):
        return self

class DummyMask():
    def __init__(self, obj_dim, object_names):
        self.filtered_active_set = list()
        self.active_mask = np.ones(obj_dim)
        self.tar_name = object_names.target

    def regenerate_norm(self, norm):
        self.limits = norm.lim_dict[self.tar_name]
        self.range = norm.lim_dict[self.tar_name][1] - norm.lim_dict[self.tar_name][0]

class DummyInteraction(): # general dummy interaction
    def __init__(self, args, object_names, environment, obj_dim, mask=None):
        self.name = object_names.target
        self.names = object_names
        self.extractor = CausalExtractor(object_names, environment)
        self.target_selector, self.full_parent_selector, self.additional_select, \
            self.additional_selectors, self.padi_selector, self.parent_select, self.inter_selector = self.extractor.get_selectors()
        self.active_mask = np.ones(obj_dim) if mask is None else mask
        self.obj_dim = obj_dim
        self.active_set = list()
        self.mask = DummyMask(obj_dim, object_names)
        self.norm, self.extractor = self.regenerate(environment)
        self.multi_instanced = environment.object_instanced[object_names.target] > 1
        self.predict_dynamics = False
        self.position_masks = environment.position_masks
        self.proximity_epsilon = args.inter.proximity_epsilon

    def regenerate(self, environment):
        self.extractor = CausalExtractor(self.names, environment)
        self.target_select, self.full_parent_select, self.additional_select, self.additional_selectors, \
            self.padi_selector, self.parent_select, self.inter_select = self.extractor.get_selectors()
        self.norm = NormalizationModule(environment.object_range, environment.object_dynamics, self.names, environment.object_instanced, self.extractor.active)
        if hasattr(self, "mask") and self.mask is not None: self.mask.regenerate_norm(self.norm)
        return self.norm, self.extractor

    def save(self, pth):
        torch.save(self, os.path.join(create_directory(pth), self.name + "_inter_model.pt"))

    # no cuda needed for this class, but it might be called
    def cuda(self, device=-1):
        return self

    def cpu(self):
        return self

    def interaction(self, val, target, next_target): 
        if type(val) != np.ndarray: # if batches, use a value difference
            inter = np.linalg.norm(val.next_target - val.target) > 0.01
            return inter
        return np.ones(val.shape).astype(bool)

    def normalize_batch(self, batch): # copied from interaction_model.py
        batch.inter_state = self.norm(batch.inter_state, form="inter")
        batch.target = self.norm(batch.target)
        batch.next_target = self.norm(batch.next_target)
        batch.target_diff = self.norm(batch.target_diff, form="dyn")
        return batch
