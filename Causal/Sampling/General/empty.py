import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Sampling.General.centered import CenteredSampler
from Causal.Utils.instance_handling import split_instances
from Option.General.param import check_close

class EmptySampler(Sampler):
    def __init__(self, **kwargs):
        # Sampler only for Sokoban
        super().__init__(**kwargs)
        self.centered_sampler = CenteredSampler(**kwargs)
        self.epsilon_close = np.array(kwargs["epsilon_close"])
        self.additional_dim = 2#kwargs["additional_dim"]

    def update(self):
        self.centered_sampler.update()

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        while True:
            failed = False
            sample, mask = self.centered_sampler.sample(full_state)
            if len(self.additional_selector.names) == 0:
                break
            obstacles = split_instances(self.additional_selector(full_state["factored_state"]).squeeze(), self.additional_dim)
            block = np.expand_dims(full_state["factored_state"]["Block"], 0)
            for additional in np.concatenate([obstacles, block], axis=0):
                if check_close(self.epsilon_close, 1, sample,additional, np.ones(sample.shape)):
                    failed = True
            if not failed: break
        return sample, mask