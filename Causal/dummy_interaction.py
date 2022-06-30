import os
import torch
import numpy as np
from Record.file_management import create_directory


class ActionMask():
    def __init__(self, discrete_actions, num_actions, action_shape):
        self.filtered_active_set = list()
        self.active_mask = np.ones(action_shape)

class ActionDummyInteraction():
    def __init__(self, action_shape, discrete_actions, num_actions):
        self.name = "Action"
        self.active_mask = np.ones(action_shape)
        self.active_set = [1 for i in range(num_actions)] if discrete_actions else list()
        self.mask = ActionMask(discrete_actions, num_actions, action_shape)

    def save(self, pth):
        torch.save(self, os.path.join(create_directory(pth), self.name + "_inter_model.pt"))

    # no cuda needed for this class, but it might be called
    def cuda(self, device=-1):
        return self

    def cpu(self):
        return self