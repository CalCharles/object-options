import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, network_type
from Network.network_utils import pytorch_model, get_acti, assign_distribution
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork
from Network.Dists.mask_utils import expand_mask
import copy, time

class InteractionMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "none" if inter_args.softmax_output else "sigmoid"
        inter_args.mask_attn.needs_encoding = True
        self.softmax_output = inter_args.softmax_output
        self.num_outputs = inter_args.num_outputs
        self.inter = network_type[args.net_type](inter_args)
        self.softmax = nn.Softmax(dim=-1)
        self.model = [self.inter]

        self.train()
        self.reset_network_parameters()
        
    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        # print(v[:10])
        if self.softmax_output:
            # print(v.shape)
            v = self.softmax(v)

        return v

class InteractionSelectionMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        self.num_clusters = args.cluster.num_clusters
        self.first_obj_dim = args.pair.first_obj_dim
        self.single_obj_dim = args.pair.single_obj_dim
        self.selection_temperature = args.selection_temperature
        self.num_keys = self.first_obj_dim // self.single_obj_dim
        select_args = copy.deepcopy(args)
        select_args.pair.aggregate_final = False
        select_args.include_last = True
        select_args.num_outputs = self.num_clusters
        select_args.activation_final = "none"
        select_args.mask_attn.needs_encoding = True
        self.selection_network = network_type[select_args.net_type](select_args)

        self.soft_inter_dist = assign_distribution("RelaxedHot")
        self.hard_inter_dist = assign_distribution("CategoricalHot")

        # inter models must operate by pointnet principles to be instance invariant
        inter_args = copy.deepcopy(args)
        inter_args.net_type = "keypair"
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "sigmoid"
        inter_args.hidden_sizes = args.cluster.cluster_inter_hidden
        inter_args.pair.num_pair_layers = args.cluster.inter_pair_layers
        inter_args.pair.aggregate_final = False
        self.inter_models = nn.ModuleList([network_type[inter_args.net_type](inter_args) for i in range(self.num_clusters - 2)]) # two clusters reserved, one for passive and one for full

        self.softmax_output = inter_args.softmax_output
        self.num_outputs = inter_args.num_outputs
        self.softmax = nn.Softmax(dim=-1)
        self.model = [self.selection_network, self.inter_models]

        self.train()
        self.reset_network_parameters()
        print(self)
        error
        
    def forward(self, x, hard=False):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        s = self.selection_network.forward(x)
        s = self.softmax(s)
        print(s.shape, x.shape)
        s = s.reshape(s.shape[0], -1, self.num_clusters) # (batch, num_keys, num_clusters)
        if hard: s = self.hard_inter_dist(s).sample()
        else: s = self.soft_inter_dist(self.selection_temperature, probs=s).rsample()
        inters = torch.stack([self.inter_models[i].forward(x).reshape(x.shape[0], self.num_keys, -1) for i in range(self.num_clusters)], dim=2) # expects to get output of shape (batch, num_keys, num_clusters, num_instances)
        inters = inters * s.unsqueeze(-1)
        v = inters.sum(dim=2).reshape(x.shape[0], -1)
        return v

