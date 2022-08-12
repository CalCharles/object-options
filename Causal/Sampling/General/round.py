import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Sampling.General.centered import CenteredSampler
from Causal.Sampling.General.uniform import UniformSampler

def closest(val, arr):
    idx = (np.abs(arr - val)).argmin()
    return idx, arr[idx]

class RoundSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampler = CenteredSampler(**kwargs) if kwargs["round_base"] == "centered" else UniformSampler(**kwargs)
        self.round_step = kwargs["round_step"]
        self.round_vals = np.linspace(self.mask.limits[0], self.mask.limits[1], int((self.mask.limits[1] - self.mask.limits[0]) // self.rount_step))

    def update(self):
        self.sampler.update()

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        sample, mask = self.sampler(full_state)
        if len(sample.shape) == 1:
            idx, sample = closest(sample, self.round_vals)
        else:
            samples = list()
            for s in sample:
                idx, rs = closest(s, self.round_vals)
                samples.append(rs)
            sample = np.stack(samples, axis=0)
        return sample, mask
