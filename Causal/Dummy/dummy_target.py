import os
import torch
import numpy as np
from Record.file_management import create_directory
from State.feature_selector import construct_object_selector
from Environment.Normalization.norm import NormalizationModule
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.dummy_interaction import DummyInteraction

class DummyTargetInteraction(DummyInteraction): # general dummy interaction
    def __init__(self, args, object_names, environment, obj_dim, mask=None):
        super().__init__(args, object_names, environment, obj_dim, mask)
        self.mask.active_mask[-1] = 0
        self.active_mask = self.mask.active_mask