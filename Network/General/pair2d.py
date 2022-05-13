from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class PairConvNetwork(Network):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.first_dim = kwargs["first_obj_dim"]
        self.input_dims = kwargs["input_dims"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        self.output_dim = kwargs["output_dim"]

        conv_args = copy.deepcopy(kwargs)
        conv_args["num_inputs"] = self.num_inputs - self.first_dim
        conv_args["output_dim"] = self.hidden_sizes[-1] # last hidden size is the output size of the conv network
        conv_args["hidden_sizes"] = self.hidden_sizes[:-1]
        redu = conv_args["reduce"] if "reduce" in conv_args else False
        conv_args["include_last"] = True
        conv_args["reduce"] = True
        self.conv = Basic2DConvNetwork(**conv_args)

        mlp_args = copy.deepcopy(kwargs)
        mlp_args["num_inputs"] = self.first_dim + self.hidden_sizes[-1]
        mlp_args["output_dim"] = self.output_dim
        mlp_args["hidden_sizes"] = kwargs["post_process"]
        self.final_layer = BasicMLPNetwork(**mlp_args)
        self.model = nn.ModuleList([self.conv, self.final_layer])

        self.train()
        self.reset_parameters()

    def forward(self, x):
        px = x[...,:self.first_dim]
        if len(px.shape) <= 1:
            px = px.reshape(-1, *px.shape)
        x = x[...,self.first_dim:]
        x = x.reshape(-1, *self.input_dims)
        x = x.transpose(2,3).transpose(1,2)
        x = self.conv(x)
        x = torch.cat([x, px], dim=-1)
        x = self.final_layer(x)
        return x