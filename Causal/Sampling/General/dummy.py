import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Utils.instance_handling import split_instances

class DummySampler(Sampler): # samples only from existing objects (last element, with )
    def __init__(self, **kwargs):
        self.obj_dim = kwargs["obj_dim"]
        super().__init__(**kwargs)

    def sample_single(self, target):
        return np.where(target[...,-1] == 1) 

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = self.target_selector(full_state["factored_state"])
        if len(target.shape) > 1:
            return np.ones((target.shape[0], self.obj_dim)), np.ones((target.shape[0], self.obj_dim))
        return np.ones(self.obj_dim), np.ones(target.shape[0])
