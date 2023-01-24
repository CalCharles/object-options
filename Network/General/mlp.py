from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MLPNetwork(Network):    
    def __init__(self, args):
        super().__init__(args)
        self.scale_final = args.scale_final
        if len(self.hs) == 0:
            if self.use_layer_norm:
                layers = [nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.num_outputs)]
            else:
                layers = [nn.Linear(self.num_inputs, self.num_outputs)]
        elif self.use_layer_norm:
            layers = ([nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
        else:
            layers = ([nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True)] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True)] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        x = self.model(x)
        x = self.activation_final(x)
        if hasattr(self, "scale_final"): x = x * self.scale_final
        return x