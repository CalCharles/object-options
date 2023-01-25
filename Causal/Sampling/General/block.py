import numpy as np
import copy

class BreakoutBlockSampler(): # samples only from existing blocks (last element, with )
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.param = np.zeros(5)
        self.param_idx = 0

    def sample_single(self, target):
        return np.where(target[...,-1] == 1)[0]

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        if self.num_blocks == 1:
            self.param = copy.deepcopy(np.array(full_state["factored_state"]["Block"]))
            return self.param
        blocks = [np.array(full_state["factored_state"]["Block" + str(i)]) for i in range(self.num_blocks)]
        valid = [i for i in range(self.num_blocks) if blocks[i][-1] == 1]
        if len(valid) == 0:
            self.param_idx = np.random.randint(self.num_blocks)
            self.param = copy.deepcopy(np.array(full_state["factored_state"]["Block" + str(self.param_idx)]))
            return self.param
        sample = np.random.choice(valid)
        self.param  = copy.deepcopy(blocks[sample].squeeze())
        self.param[-1] = 0
        self.param_idx = sample
        return self.param
