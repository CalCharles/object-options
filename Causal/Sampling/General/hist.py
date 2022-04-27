import numpy as np
from Causal.Sampling.sampler import Sampler

class HistorySampler(Sampler):
    def __init__(self, **kwargs):
        self.sample_raw = kwargs["sample_raw"]
        super().__init__(**kwargs)
        self.epsilon = 1e-5
        self.active_filtered = self.filter_active()

    def filter_active(self):
        '''
        filters self.masking.active_set based on the active mask
        if states are the same after masking, they are only counted once
        '''
        active_filtered = list()
        for state in self.masking.active_set:
            masked_state = state * self.masking.active_mask
            failed = False
            for val in active_filtered:
                if np.linalg.norm(masked_state - val, ord=1) > self.epsilon:
                    failed = True
            if not failed:
                active_filtered.append(masked_state)
        return active_filtered

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        use_set = self.masking.active_set if self.sample_raw else self.active_filtered
        target = use_set[np.random.randint(len(use_set))]
        return use_set