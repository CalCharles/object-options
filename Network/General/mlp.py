from Network.network import Network
from Network.network_utils import get_inplace_acti
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BIAS = True

class MLPNetwork(Network):    
    def __init__(self, args):
        super().__init__(args)
        self.scale_final = args.scale_final
        self.is_crelu = args.activation == "crelu"
        crelu_mul = 1
        if self.is_crelu: 
            self.hs = [int(hs / 2) for hs in self.hs]
            crelu_mul = 2
            if args.activation_final == "crelu": # this prevents there from being issues with the last layers of networks
                self.activation_final = get_inplace_acti("leakyrelu")

        if len(self.hs) == 0:
            if self.use_layer_norm:
                layers = [nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.num_outputs, bias=args.use_bias)]
            else:
                layers = [nn.Linear(self.num_inputs, self.num_outputs, bias=args.use_bias)]
        elif self.use_layer_norm:
            layers = ([nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.hs[0], bias=args.use_bias), get_inplace_acti(args.activation),nn.LayerNorm(self.hs[0] * crelu_mul)] + 
                  sum([[nn.Linear(self.hs[i-1] * crelu_mul, self.hs[i], bias=args.use_bias), get_inplace_acti(args.activation), nn.LayerNorm(self.hs[i] * crelu_mul)] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1] * crelu_mul, self.num_outputs)])
        else:
            layers = ([nn.Linear(self.num_inputs, self.hs[0], bias=args.use_bias), get_inplace_acti(args.activation)] + 
                  sum([[nn.Linear(self.hs[i-1] * crelu_mul, self.hs[i], bias=args.use_bias), get_inplace_acti(args.activation)] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1] * crelu_mul, self.num_outputs, bias=args.use_bias)])
        if "dropout" in args and args.dropout > 0:
            layers = [nn.Dropout(args.dropout)] + layers
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        x = self.model(x)
        x = self.activation_final(x)
        if hasattr(self, "scale_final"): x = x * self.scale_final
        return x