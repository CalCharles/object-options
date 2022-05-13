from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PointNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        self.object_dim = kwargs['object_dim']
        self.hs = kwargs["hidden_sizes"]
        self.aggregate_final = kwargs["aggregate_final"]
        self.output_dim = kwargs["output_dim"]
        self.reduce_fn = kwargs["reduce_function"]
        
        subnet_args = copy.deepcopy(kwargs)
        subnet_args["include_last"] = True
        self.conv = BasicConvNetwork(**subnet_args)
        if kwargs["aggregate_final"]:
            subnet_args["num_inputs"] = self.output_dim
            subnet_args["hidden_sizes"] = kwargs["post_process"] 
            self.MLP = BasicMLPNetwork(**subnet_args)
            self.model = nn.Sequential(self.conv, self.MLP)
        else:
            self.model = nn.ModuleList([self.conv])
        self.train()
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        nobj = x.shape[-1] // self.object_dim
        x = x.view(-1, nobj, self.object_dim).transpose(1,2)
        x = self.conv(x).transpose(2,1)
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x)[0]
            x = x.view(-1, self.output_dim)
            x = self.MLP(x)
        return x