import numpy as np
import torch
import copy
import collections
import torch.nn as nn
import torch.optim as optim
from Networks.network import pytorch_model
from Networks.DistributionalNetworks.forward_network import forward_nets
from Options.state_extractor import array_state
from tianshou.data import Batch
from Rollouts.param_buffer import SamplerBuffer
from file_management import numpy_factored
from Environments.SelfBreakout.breakout_screen import AnglePolicy

class Sampler():
    def __init__(self, **kwargs):
        self.masking = kwargs["masking"]
        self.target_selector = kwargs["target_selector"]

    def update(self):
        # if there is a schedule for the sampler values, updates the schedule
        return

    def sample(self, full_state):
        '''
        samples a new value according to the sampling function
        only returns a single sample. Full state is used to bring in relevant information to the sampling
        '''
        return 

# samplers aggregated here
from Causal.Sampling.General.uniform import UniformSampler
mask_samplers = {"rans": RandomSubsetSampling, "pris": PrioritizedSubsetSampling} # must be 4 characters
samplers = {"uni": LinearUniformSampling, "cuni": LinearUniformCenteredSampling, 'cuuni': LinearUniformCenteredUnclipSampling,
            "gau": GaussianCenteredSampling, "hst": HistorySampling, 'inst': InstanceSampling,
            "hstinst": HistoryInstanceSampling, 'tar': TargetSampler, 'path': PathSampler, "block": BreakoutTargetSampler,
            'randblock': BreakoutRandomSampler}  