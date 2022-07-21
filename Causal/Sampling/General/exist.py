import numpy as np
from Causal.Sampling.sampler import Sampler
from Causal.Utils.instance_handling import split_instances

class ExistSampler(Sampler): # samples only from existing objects (last element, with )
    def __init__(self, **kwargs):
        self.obj_dim = kwargs["obj_dim"]
        super().__init__(**kwargs)

    def sample_single(self, target):
        return np.where(target[...,-1] == 1)[0]

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = self.target_selector(full_state["factored_state"])
        target = split_instances(target, self.obj_dim)
        if len(target.shape) > 2: # we have batches
            samples, masks = list(), list()
            for tar in target:
                sample = tar[np.random.choice(self.sample_single(tar).squeeze())]
                samples.append(sample)
                mask = np.zeros(sample.shape)
                mask[-1] = 1
                masks.append(mask)
        else:
            samples = target[np.random.choice(self.sample_single(target).squeeze())]
            masks = np.zeros(samples.shape)
            masks[-1] = 1
        return samples, masks
