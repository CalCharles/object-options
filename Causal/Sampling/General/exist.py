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
                sidxes = self.sample_single(tar).squeeze()
                if len(sidxes.shape) == 0:
                    sidxes = np.array([sidxes])
                if len(sidxes) == 0:
                    sample = np.zeros(target.shape)
                else:
                    sample = tar[np.random.choice(sidxes)]
                    samples.append(sample)
                mask = np.zeros(sample.shape)
                mask[-1] = 1
                masks.append(mask)
        else:
            sidxes = self.sample_single(target).squeeze()
            if len(sidxes.shape) == 0:
                sidxes = np.array([sidxes])
            if len(sidxes) == 0:
                samples = np.zeros(target.shape)
            else:
                samples = target[np.random.choice(sidxes)]
            masks = np.zeros(samples.shape)
            masks[-1] = 1
            self.param = samples
        return samples, masks
