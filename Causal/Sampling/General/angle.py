import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Sampling.General.centered import CenteredSampler

class AngleSampler(Sampler):
    def __init__(self, **kwargs):
        # assumes the last 2 dimensions are sin, cos of a single angle
        super().__init__(**kwargs)
        self.centered_sampler = CenteredSampler(**kwargs)
        self.availiable_angles = np.array([2 * np.pi * i/kwargs["num_angles"] for i in range(kwargs["num_angles"])])
        self.num_angles = kwargs["num_angles"]
        self.exist = kwargs["positive"]

    def update(self):
        self.centered_sampler.update()

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        sample, mask = self.centered_sampler.sample(full_state)
        angle = self.availiable_angles[np.random.randint(self.num_angles)].copy()
        sample[...,sample.shape[-1]-3:sample.shape[-1]-1] = np.array([np.sin(angle), np.cos(angle)])
        sample[...,-1] = np.random.rand() if not self.exist else 1
        print("sample", angle, sample)
        return sample, mask