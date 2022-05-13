from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Basic2DConvNetwork(Network): # basic 1d conv network 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kwargs["kernel"] # square kernel of a single value
        self.input_dims = kwargs["input_dims"]
        self.channels = self.input_dims[-1] if len(self.input_dims) > 2 else 1
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.output_dim = kwargs["output_dim"]
        self.reduce = kwargs["reduce"]
        include_last = kwargs['include_last']

        x,y = self.input_dims[:2]# technically switched
        for i in range(len(self.hs)):
            x = int(( x + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
            y = int(( y + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        last_num = self.hs[-1] if not include_last else self.output_dim
        self.reduce_size = x * y * last_num 

        if len(self.hs) == 0:
            layers = [nn.Conv2d(self.channels, self.output_dim, self.kernel, self.stride, self.padding)]
        else:
            if len(self.hs) == 1:
                layers = [nn.Conv2d(self.channels, self.hs[0], self.kernel, self.stride, self.padding)]
            else:
                layers = ([nn.Conv2d(self.channels, self.hs[0], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)] + 
                      sum([[nn.Conv2d(self.hs[i-1], self.hs[i], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)] for i in range(1, len(self.hs) - 1)], list())
                      + [nn.Conv2d(self.hs[-2], self.hs[-1], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv2d(self.hs[-1], self.output_dim, self.kernel, self.stride, self.padding)]
        self.conv = nn.Sequential(*layers)
        self.model = nn.ModuleList([self.conv])
        self.final = None
        if self.reduce:
            self.final = nn.Linear(self.reduce_size, self.output_dim)
            self.model = nn.ModuleList([self.conv] + [self.final])
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.reduce:
            x = x.reshape(-1, self.reduce_size)
            x = self.final(x)
        return x