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

class KeyPairNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # expects that object_dim is the same for the targets and the values
        self.single_object_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.reduce_fn = args.pair.reduce_function
        self.conv_dim = self.hs[-1] if self.aggregate_final else args.num_outputs
        self.query_pair = not args.pair.query_pair
        self.total_obj_dim = args.pair.total_obj_dim
        self.expand_dim = args.pair.expand_dim
        self.total_instances = args.pair.total_instances
        self.total_targets = args.pair.total_targets

        pair_args = copy.deepcopy(args)
        pair_args.pair.first_obj_dim = self.single_object_dim
        pair_args.pair.aggregate_final = self.query_pair # the pair net computes over all queries, so we only need a single output for any one target
        pair_args.num_outputs = self.conv_dim
        self.pair_net = PairNetwork(pair_args)
        layers = [self.pair_net]

        if args.pair.aggregate_final: # copied from pairnet
            args.include_last = True
            args.num_inputs = self.conv_dim
            args.num_outputs = self.num_outputs
            args.hidden_sizes = args.pair.final_layers # TODO: hardcoded final hidden sizes for now
            self.MLP = MLPNetwork(args)
            layers.append(self.MLP)

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def set_instancing(self, total_instances, total_obj_dim, total_targets):
        assert self.expand_dim * total_instances == total_obj_dim
        self.total_obj_dim = total_obj_dim
        self.total_instances = total_instances
        self.total_targets = total_targets

    def slice_mask_input(self, x, i, m):
        if m is None:
            xi = torch.cat([x[...,i * self.single_object_dim: (i+1) * self.single_object_dim], x[...,self.first_obj_dim:]], dim=-1)
        else:
            xi = torch.cat([x[...,i * self.single_object_dim: (i+1) * self.single_object_dim], x[...,self.first_obj_dim:] * m[...,i * self.total_obj_dim:(i+1) * self.total_obj_dim]], dim=-1)
        return xi

    def forward(self, x, m=None):
        # iterate over each instance
        batch_size = x.shape[0]
        value = list()
        for i in range(int(self.first_obj_dim // self.single_object_dim)):
            xi = self.slice_mask_input(x, i, m)
            # print(i, x.shape, xi.shape)
            # print(self.pair_net.aggregate_final, self.pair_net.num_outputs, self.pair_net.first_obj_dim)
            value.append(self.pair_net(xi))
        # print(value[0].shape)
        x = torch.stack(value, dim=2) # [batch size, pairnet output dim, num_instances]
        # print(x.shape)
        # print(x.shape, self.aggregate_final)
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.conv_dim)
            # print(x.shape)
            x = self.MLP(x)
        else:
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
        return x
