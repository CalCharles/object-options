import numpy as np
from Causal.Sampling.sampler import Sampler

class CenteredSampler(Sampler):
    def __init__(self, **kwargs):
        self.distance = kwargs["sample_distance"]
        self.schedule_counter = 0
        self.schedule = kwargs["sample_schedule"]
        super().__init__(**kwargs)
        self.current_distance = .1 if self.schedule > 0 or self.test_sampler else kwargs["sample_distance"]

    def update(self):
        if self.schedule > 0:
            self.schedule_counter += 1
            self.current_distance = self.distance if self.test_sampler else self.distance - (self.distance - self.current_distance) * np.exp(-(self.schedule_counter + 1)/self.schedule)

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = self.target_selector(full_state["factored_state"])
        axis = 0 if len(target.shape) == 1 else 1
        upper_range = np.min([self.mask.limits[1], target + self.current_distance * self.mask.range], axis=axis)
        lower_range = np.max([target - self.current_distance * self.mask.range, self.mask.limits[0]], axis=axis)
        limit_ranges = (upper_range - lower_range) / 2
        new_center = (lower_range + upper_range) / 2
        weights = (np.random.rand(*target.shape) - .5) * 2
        return (new_center + limit_ranges * weights), self.mask.active_mask
