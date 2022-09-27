import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Utils.instance_handling import split_instances

class TargetSampler(Sampler): # samples the target object
    def __init__(self, **kwargs):
        self.obj_dim = kwargs["obj_dim"]
        super().__init__(**kwargs)

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = self.target_selector(full_state["factored_state"])
        return target, self.mask.active_mask
