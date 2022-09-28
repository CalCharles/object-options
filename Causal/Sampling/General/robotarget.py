import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Utils.instance_handling import split_instances

class RoboTargetSampler(Sampler): # samples the target object
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = full_state["factored_state"]["Target"]
        return target, self.mask.active_mask
