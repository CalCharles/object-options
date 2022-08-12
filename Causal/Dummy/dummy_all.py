import os
import torch
import numpy as np
from Record.file_management import create_directory
from State.feature_selector import construct_object_selector
from Environment.Normalization.norm import NormalizationModule
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.dummy_interaction import DummyInteraction

class DummyAllInteraction(DummyInteraction): # general dummy interaction
    def __init__(self, args, object_names, environment, obj_dim, mask=None):
        super().__init__(args, object_names, environment, obj_dim, mask)

    def interaction(self, val, target=None, next_target=None, target_diff=None): 
        if type(val) != np.ndarray: # if batches, use a value difference
            return np.ones(val.inter_state.shape)[...,-1]
        else:
            return np.ones(inter_state.shape)[...,-1]