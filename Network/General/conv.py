from Network.network import Network
from Network.network_utils import get_inplace_acti
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

BIAS = True

class ConvNetwork(Network): # basic 1d conv network 
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.object_dim
        self.output_dim = args.output_dim # slightly different in meaning from num_outputs
        include_last = args.include_last
        self.is_crelu = args.activation == "crelu"
        crelu_mul = 1
        if self.is_crelu: 
            self.hs = [int(hs / 2) for hs in self.hs]
            crelu_mul = 2
            if args.activation_final == "crelu":
                self.activation_final = get_inplace_acti("leakyrelu")

        if len(self.hs) == 0:
            if self.use_layer_norm: # activation final handles activativation
                layers = [nn.GroupNorm(1, self.object_dim), nn.Conv1d(self.object_dim, self.output_dim, 1, bias=args.use_bias)]
            else:
                layers = [nn.Conv1d(self.object_dim, self.output_dim, 1, bias=args.use_bias)]
        else:
            if len(self.hs) == 1:
                if self.use_layer_norm:
                    layers = [nn.Conv1d(self.object_dim, self.hs[0], 1, bias=args.use_bias), get_inplace_acti(args.activation), nn.GroupNorm(1, self.hs[0] * crelu_mul)]
                else:
                    layers = [nn.Conv1d(self.object_dim, self.hs[0], 1, bias=args.use_bias), get_inplace_acti(args.activation)]
            else:
                if self.use_layer_norm:
                    layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1, bias=args.use_bias), get_inplace_acti(args.activation),nn.GroupNorm(1, self.hs[0] * crelu_mul)] + 
                    sum([[nn.Conv1d(self.hs[i-1] * crelu_mul, self.hs[i], 1, bias=args.use_bias), get_inplace_acti(args.activation), nn.GroupNorm(1, self.hs[i] * crelu_mul)] for i in range(1, len(self.hs) - 1)], list())
                        + [nn.Conv1d(self.hs[-2] * crelu_mul, self.hs[-1], 1, bias=args.use_bias), get_inplace_acti(args.activation)])
                else:
                    layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1, bias=args.use_bias), get_inplace_acti(args.activation)] + 
                        sum([[nn.Conv1d(self.hs[i-1] * crelu_mul, self.hs[i], 1, bias=args.use_bias), get_inplace_acti(args.activation)] for i in range(1, len(self.hs) - 1)], list())
                        + [nn.Conv1d(self.hs[-2] * crelu_mul, self.hs[-1], 1, bias=args.use_bias), get_inplace_acti(args.activation)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv1d(self.hs[-1] * crelu_mul, self.output_dim, 1, bias=args.use_bias)]
        if "dropout" in args and args.dropout > 0:
            layers = [nn.Dropout(args.dropout)] + layers
        self.model = nn.Sequential(*layers)
        self.train()
        # print(crelu_mul, self)
        # error
        self.reset_network_parameters()

    def forward(self, x):
        x = self.model(x)
        x = self.activation_final(x)
        return x
