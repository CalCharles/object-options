from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from Network.network_utils import reduce_function, get_acti
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork

class InputExpandNetwork(Network):
    def __init__(self, args):
        '''
        TODO: if we make only the input expansion part a separate network, we might be able to add it to pair-nets
        performs a series of operations between the parent and the target. This is not built to handle additional objects
        always assumes the parent is at the front and the target is at the back
        Operations to be performed:
            difference
            multiplication
            closeness
            farness
            angle
        Multi operations:
            1- value
            base value
        '''
        super().__init__(args)
        self.target_dim = args.pair.target_dim # the dimension of the target object
        self.single_object_dim = args.pair.single_obj_dim # the dimension used for the parent component, assumed to be at the front of first_obj_dim
        self.parent_dim = args.pair.parent_dim # 
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the first components, i.e. the single_object_dim + 
        self.include_relative = args.input_expand.include_relative # includes relative operations if True

        self.relative_size = min(self.parent_dim, self.target_dim)
        self.relative_size_total = int((self.relative_size * (4 + 1)) * float(self.include_relative))
        self.single_size_total = int(4 * self.parent_dim + 4 * self.target_dim) if self.include_relative else self.num_inputs * 4
        self.total_dim = self.relative_size_total + self.single_size_total
        self.total_operations = int((5 + 4) * float(self.include_relative) + 4) # 5 relative operations and 4 single operations (counted twice if include relative)

        layers = list()
        self.pre_embed = args.input_expand.pre_embed
        self.pad_dim = max(self.parent_dim, self.target_dim)
        self.mlp_preembed = list()
        if len(self.pre_embed) > 0:
            for i in range(self.total_operations):
                mlp_args = copy.deepcopy(args)
                mlp_args.num_inputs = max(self.parent_dim, self.target_dim) if self.include_relative else self.num_inputs
                mlp_args.num_outputs = self.pre_embed[-1]
                mlp_args.hidden_sizes = self.pre_embed[:-1]
                self.mlp_args = mlp_args
                self.mlp_preembed.append(MLPNetwork(mlp_args))
            layers += self.mlp_preembed
            self.mlp_preembed = nn.ModuleList(self.mlp_preembed)
            self.total_dim = self.pre_embed[-1] * TOTAL_OPERATIONS
            self.relative_size_total = int(self.pre_embed[-1] * 5 * float(self.include_relative))
            self.single_size_total = int(self.pre_embed[-1] * (4 + 4 * float(self.include_relative)))

        args.include_last = True
        args.num_inputs = self.total_dim
        args.num_outputs = self.num_outputs
        self.MLP = MLPNetwork(args)
        layers.append(self.MLP)

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def _split_input(self, x):
        return x[...,:self.parent_dim], x[...,self.parent_dim:self.first_obj_dim], x[...,self.first_obj_dim:self.first_obj_dim + self.target_dim]

    def apply_relative_operations(self, x1, x2):
        # difference
        # multiplication
        # closeness
        # farness
        # angle
        pad = lambda x: F.pad(x, (0,self.pad_dim - self.relative_size))
        diff = (x1[...,:self.relative_size] - x2[...,:self.relative_size]) / 2
        diff = self.mlp_preembed[0](pad(diff)) if self.pre_embed else diff
        mul = x1[...,:self.relative_size] * x2[...,:self.relative_size]
        mul = self.mlp_preembed[1](pad(mul)) if self.pre_embed else mul
        close = torch.exp(-(x1[...,:self.relative_size] - x2[...,:self.relative_size]).abs() * 3)
        close = self.mlp_preembed[2](pad(close)) if self.pre_embed else close
        far = (x1[...,:self.relative_size] - x2[...,:self.relative_size]).abs() / 2
        far = self.mlp_preembed[3](pad(far)) if self.pre_embed else far
        angle = (x1[...,:self.relative_size] * x2[...,:self.relative_size]).sum(axis=-1) / (torch.norm(x1[...,:self.relative_size]) * torch.norm(x2[...,:self.relative_size]))
        angle = self.mlp_preembed[4](F.pad(angle, (0,self.pad_dim - 1))) if self.pre_embed else angle
        return [diff, mul, close, far, angle]

    def apply_single_operations(self, x, start):
        # 1- value
        # base value
        pad = lambda x: F.pad(x, (0,self.pad_dim - x1.shape[-1]))
        opp = 1-x
        opp = self.mlp_preembed[start](pad(opp)) if self.pre_embed else opp
        base = x
        base = self.mlp_preembed[start + 1](pad(base)) if self.pre_embed else base
        close = torch.exp(-x.abs() * 5)
        close = self.mlp_preembed[start + 2](pad(close)) if self.pre_embed else close
        far = torch.exp((x.abs() - 1) * 5)
        far = self.mlp_preembed[start + 3](pad(far)) if self.pre_embed else far
        size_total = opp.shape[-1] + base.shape[-1] + close.shape[-1] + far.shape[-1]
        return [opp, base, close, far], size_total

    def forward(self, x):
        # iterate over each instance
        batch_size = x.shape[0]
        if self.include_relative:
            x, fir, y = self._split_input(x) # TODO: first object not parent not used
            inp = list()
            start = 0
            if self.include_relative:
                inp += self.apply_relative_operations(x, y)
                start += self.relative_size_total
            ninp, total = self.apply_single_operations(x, start)
            start += total
            inp += ninp
            ninp, total = self.apply_single_operations(y, start)
            start += total
            inp += ninp
        else:
            inp, total = self.apply_single_operations(x, 0)
        # print(torch.cat(inp, dim=-1).shape, self.total_dim)
        # print(torch.cat(inp, dim=-1)[0])
        x = self.MLP(torch.cat(inp, dim=-1))
        return x
