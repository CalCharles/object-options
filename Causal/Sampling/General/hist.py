import numpy as np
from Causal.Sampling.sampler import Sampler

class HistorySampler(Sampler):
    def __init__(self, **kwargs):
        self.sample_raw = kwargs["sample_raw"]
        super().__init__(**kwargs)
        self.epsilon = 1e-5
        self.active_filtered = self.masking.filtered_active_set.copy()


    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        use_set = self.masking.active_set if self.sample_raw else self.active_filtered
        target = use_set[np.random.randint(len(use_set))]
        return use_set