import os
import torch
import numpy as np
from Record.file_management import create_directory
from State.feature_selector import construct_object_selector
from Environment.Normalization.norm import NormalizationModule
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.dummy_interaction import DummyInteraction

class DummyLaserInteraction(DummyInteraction): # general dummy interaction
    def __init__(self, args, object_names, environment, obj_dim, mask=None):
        super().__init__(args, object_names, environment, obj_dim, mask)

    def interaction(self, val, target, next_target, target_diff): 
        if type(val) != np.ndarray: # if batches, use a value difference
            inter = val.next_target[...,-1] - val.target[...,-1] > 0.01
            return inter
        else:
            inter = next_target[...,-1] - target[...,-1] > 0.01
            return inter