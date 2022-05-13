from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# TODO: INCOMPLETE
class FactoredMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor = kwargs['factor']
        self.num_layers = kwargs['num_layers']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.MLP = BasicMLPNetwork(**kwargs)
        self.train()
        self.reset_parameters()

    def basic_operations(self, x):
        # add, subtract, outer product
        return

    def forward(self, x):
        x = self.basic_operations(x)
        x = self.MLP(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x