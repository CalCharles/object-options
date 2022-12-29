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

NUM_SINGLE = 4
NUM_RELATIVE = 5

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
        self.parent_dim = args.pair.parent_dim # the dimension of the parent object
        self.first_obj_dim = int(args.pair.first_obj_dim) # this should include all the first components, i.e. the single_object_dim + 
        self.include_relative = args.input_expand.include_relative # includes relative operations if True
        self.param_mode = args.input_expand.param_mode # assumes that the remainder is the param and compares it to BOTH the target and the parent
        self.param_dim = int(self.param_mode) * self.target_dim
        self.first_include = args.input_expand.first_include # includes the first n dimensions of the input not as an object state

        print(self.target_dim, self.single_object_dim, self.parent_dim, self.first_obj_dim, self.include_relative, self.param_mode, self.first_include)
        if self.include_relative:
            self.pad_dim = max(self.parent_dim, self.target_dim)
        else:
            self.pad_dim = max(self.first_include, self.parent_dim + self.target_dim)
        self.relative_size = min(self.parent_dim, self.target_dim)
        self.relative_size_total = int((self.pad_dim * NUM_SINGLE + self.pad_dim * NUM_RELATIVE) * float(self.include_relative))
        self.param_size_total = int((self.pad_dim * NUM_RELATIVE + self.pad_dim * NUM_SINGLE + float(self.include_relative) * NUM_RELATIVE * self.pad_dim) * float(self.param_mode))
        self.first_include_size_total = int(float(self.first_include > 0) * NUM_SINGLE * self.pad_dim)
        self.single_size_total = int(NUM_SINGLE * self.pad_dim)
        self.total_dim = self.relative_size_total + self.single_size_total + self.param_size_total + self.first_include_size_total
        print(self.relative_size_total, self.param_size_total, self.first_include_size_total, self.single_size_total, self.total_dim)
        self.total_operations = int((NUM_RELATIVE + NUM_SINGLE + NUM_RELATIVE * float(self.param_mode)) * float(self.include_relative) + NUM_SINGLE + (NUM_RELATIVE+NUM_SINGLE) * float(self.param_mode)) + int(self.first_include > 0) * NUM_SINGLE # NUM_RELATIVE relative operations and NUM_SINGLE single operations (counted twice if include relative) # 9 guaranteed param operations plus NUM_RELATIVE if relative
        layers = list()
        self.pre_embed = args.input_expand.pre_embed
        self.mlp_preembed = list()
        if len(self.pre_embed) > 0:
            # TODO: first_include cannot exceed the dimension of max(self.parent_dim, self.target_dim) for this to work
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
            self.relative_size_total = int(self.pre_embed[-1] * (NUM_RELATIVE + NUM_RELATIVE * float(self.param_mode)) * float(self.include_relative) + NUM_RELATIVE * float(self.param_mode))
            self.single_size_total = int(self.pre_embed[-1] * (NUM_SINGLE + NUM_SINGLE * float(self.include_relative) + NUM_SINGLE * float(self.param_mode) + NUM_SINGLE * float(self.first_include > 0)))
            self.first_include_size_total = int(self.pre_embed[-1] * float(self.first_include > 0) * NUM_SINGLE)

        args.include_last = True
        args.num_inputs = self.total_dim
        args.num_outputs = self.num_outputs
        self.MLP = MLPNetwork(args)
        layers.append(self.MLP)

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def _split_input(self, x):
        # print(self.first_obj_dim, self.parent_dim, self.target_dim, x.shape)
        if self.include_relative:
            return x[...,self.first_obj_dim-self.parent_dim:self.first_obj_dim], x[...,:self.first_include], x[...,self.first_include:self.first_include + self.param_dim], x[...,self.first_obj_dim:self.first_obj_dim + self.target_dim]
        else: # parent dim and target dim are now irrelevant
            return x[...,self.first_include + self.param_dim:], x[...,:self.first_include], x[...,self.first_include:self.first_include + self.param_dim], x[...,self.first_obj_dim:self.first_obj_dim + self.target_dim]
    
    def apply_relative_operations(self, x1, x2, start):
        # difference
        # multiplication
        # closeness
        # farness
        # angle
        pad = lambda x: F.pad(x, (0,self.pad_dim - x.shape[-1]))
        diff = (x1[...,:self.relative_size] - x2[...,:self.relative_size]) / 2
        diff = self.mlp_preembed[start](pad(diff)) if self.pre_embed else pad(diff)
        mul = x1[...,:self.relative_size] * x2[...,:self.relative_size]
        mul = self.mlp_preembed[start+1](pad(mul)) if self.pre_embed else pad(mul)
        close = torch.exp(-(x1[...,:self.relative_size] - x2[...,:self.relative_size]).abs() * 3)
        close = self.mlp_preembed[start+2](pad(close)) if self.pre_embed else pad(close)
        far = (x1[...,:self.relative_size] - x2[...,:self.relative_size]).abs() / 2
        far = self.mlp_preembed[start+3](pad(far)) if self.pre_embed else pad(far)
        angle = ((x1[...,:self.relative_size] * x2[...,:self.relative_size]).sum(axis=-1) / (torch.norm(x1[...,:self.relative_size]) * torch.norm(x2[...,:self.relative_size]) + 1e-5)).unsqueeze(-1)
        angle = self.mlp_preembed[start+4](F.pad(angle, (0,self.pad_dim - 1))) if self.pre_embed else pad(angle)
        size_total = diff.shape[0] + mul.shape[0] + close.shape[0] + far.shape[0] + angle.shape[0]
        return [diff, mul, close, far, angle], size_total

    def apply_single_operations(self, x, start):
        # 1- value
        # base value
        pad = lambda x: F.pad(x, (0,self.pad_dim - x.shape[-1]))
        opp = 1-x
        opp = self.mlp_preembed[start](pad(opp)) if self.pre_embed else pad(opp)
        base = x
        base = self.mlp_preembed[start + 1](pad(base)) if self.pre_embed else pad(base)
        close = torch.exp(-x.abs() * 5)
        close = self.mlp_preembed[start + 2](pad(close)) if self.pre_embed else pad(close)
        far = torch.exp((torch.sin(x.abs()) - 1) * 5)
        far = self.mlp_preembed[start + 3](pad(far)) if self.pre_embed else pad(far)
        size_total = opp.shape[-1] + base.shape[-1] + close.shape[-1] + far.shape[-1]
        return [opp, base, close, far], size_total

    def forward(self, x):
        # iterate over each instance
        batch_size = x.shape[0]
        # print(x[0])
        x, finc, fir, y = self._split_input(x) # TODO: first object not parent not used
        # print(x.shape, finc.shape, fir.shape, y.shape)
        # if len(fir) > 0:
        #     print(x[0], fir[0], y[0])
        inp = list()
        start = 0
        if self.first_include > 0:
            ninp, total = self.apply_single_operations(finc, start)
            # print('finc', torch.cat(ninp, dim=-1).shape)
            start += NUM_SINGLE
            inp += ninp
        if self.include_relative or self.param_mode:
            ninp, total = self.apply_single_operations(x, start)
            # print('x', torch.cat(ninp, dim=-1).shape)
            start += NUM_SINGLE # single adds NUM_SINGLE operations
            inp += ninp
            if self.include_relative:
                ninp, total = self.apply_single_operations(y, start)
                # print('y', torch.cat(ninp, dim=-1).shape)
                start += NUM_SINGLE
                inp += ninp
                ninp, total = self.apply_relative_operations(x, y, start)
                inp += ninp
                start += NUM_RELATIVE # relative adds NUM_RELATIVE operations
                # print('yrel', torch.cat(ninp, dim=-1).shape)
            if self.param_mode: # param mode always has some relative information
                ninp, total = self.apply_relative_operations(y, fir, start)
                inp += ninp
                start += NUM_RELATIVE
                # print("yprel", torch.cat(ninp, dim=-1).shape)
                if self.include_relative:
                    ninp, total = self.apply_relative_operations(x, fir, start)
                    inp += ninp
                    # print("xprel", torch.cat(ninp, dim=-1).shape)
                    start += NUM_RELATIVE
                ninp, total = self.apply_single_operations(fir, start)
                # print("param", torch.cat(ninp, dim=-1).shape)
                start += NUM_SINGLE
                inp += ninp
        else:
            ninp, total = self.apply_single_operations(x, start)
            inp += ninp
        full_inp = torch.cat(inp, dim=-1)
        # print(full_inp[0]) 
        # print([iv.shape for iv in inp])
        # print(full_inp.shape, self.total_dim)
        # print(full_inp[0])
        # print(full_inp[0])
        x = self.MLP(full_inp)
        return x
