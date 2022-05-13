from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ConvNetwork(Network): # basic 1d conv network 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        include_last = kwargs['include_last']

        if len(self.hs) == 0:
            layers = [nn.Conv1d(self.object_dim, self.output_dim, 1)]
        else:
            if len(self.hs) == 1:
                layers = [nn.Conv1d(self.object_dim, self.hs[0], 1)]
            elif self.use_layer_norm:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(1, len(self.hs) - 1)], list())
                    + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            else:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True)] + 
                      sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True)] for i in range(1, len(self.hs) - 1)], list())
                      + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv1d(self.hs[-1], self.output_dim, 1)]
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x
