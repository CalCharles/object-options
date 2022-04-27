from Networks.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BasicResNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.residual_masks = kwargs["residual_masks"] # TODO: not yet implemented
        last_layer = nn.Linear(self.hs[-1], self.num_outputs) if len(self.hs) % 2 == 1 else nn.Linear(self.hs[-1] + self.hs[-2], self.num_outputs)
        even_layer = [nn.Linear(self.hs[-2], self.hs[-1])] if len(self.hs) % 2 == 0 else list()
        layers = ([nn.Linear(self.num_inputs, self.hs[0])] + 
              sum([[nn.Linear(self.hs[i-2], self.hs[i-1]), nn.Linear(self.hs[i-1] + self.hs[i-2], self.hs[i])] for i in [j * 2 for j in range(1, len(self.hs) // 2 + int(len(self.hs) % 2))]], list()) + 
            even_layer + [last_layer])
        self.model = nn.ModuleList(layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
            if i != len(self.model) - 1:
                x = torch.relu(x)
            if i % 2 == 0:
                lx = x
            elif i != len(self.model) - 1:
                x = torch.cat([lx, x], dim=-1)
        return x