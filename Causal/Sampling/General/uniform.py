import numpy as np
from Causal.Sampling.sampler import Sampler

class UniformSampler(Sampler):
    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        weights = np.random.rand(size = self.mask.limits[0].shape)
        return (self.mask.limits[0] + self.mask.range * weights), self.mask.active_mask