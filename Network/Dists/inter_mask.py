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
from Network.Dists.mask_utils import expand_mask, get_hot_mask, get_active_mask, get_passive_mask, apply_symmetric
import copy, time

class InteractionMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = args.cluster.num_clusters if args.cluster.use_cluster else 1 
        inter_args.activation_final = "none" if inter_args.softmax_output else "sigmoid"
        inter_args.mask_attn.needs_encoding = True
        inter_args.aggregate_final = args.cluster.use_cluster # if in cluster mode, special behavior
        self.softmax_output = inter_args.softmax_output
        self.num_outputs = inter_args.num_outputs
        self.inter = network_type[args.net_type](inter_args)
        self.softmax = nn.Softmax(dim=-1)
        self.model = [self.inter]
        self.symmetric_key_query = args.symmetric_key_query # if these are the same, then we need to cat x to itself to fit key-query expectations

        self.train()
        self.reset_network_parameters()
        
    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.class_index = class_index
        self.num_objects = num_objects
        if hasattr(self.inter, "reset_environment"): 
            self.inter.reset_environment(class_index, num_objects, first_obj_dim)

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = apply_symmetric(self.symmetric_key_query, x)
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
        assert(self.num_clusters >= 3)
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
        self.symmetric_key_query = args.symmetric_key_query # if these are the same, then we need to cat x to itself to fit key-query expectations

        self.soft_inter_dist = assign_distribution("RelaxedHot")
        self.hard_inter_dist = assign_distribution("CategoricalHot")

        # inter models must operate by pointnet principles to be instance invariant, keypair, keyembed, mask_attn, raw_attn
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "sigmoid"
        inter_args.hidden_sizes = args.cluster.cluster_inter_hidden
        inter_args.pair.num_pair_layers = args.cluster.inter_pair_layers
        inter_args.pair.aggregate_final = False
        self.inter_models = nn.ModuleList([network_type[inter_args.net_type](inter_args) for i in range(self.num_clusters)]) # two clusters reserved, one for passive and one for full

        self.softmax_output = inter_args.softmax_output
        self.num_outputs = inter_args.num_outputs
        self.softmax = nn.Softmax(dim=-1)
        self.model = [self.selection_network, self.inter_models]

        self.train()
        self.reset_network_parameters()
    
    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.class_index = class_index
        self.num_objects = num_objects
        if hasattr(self.selection_network, "reset_environment"): 
            self.selection_network.reset_environment(class_index, num_objects, first_obj_dim)
        if hasattr(self.inter_models[0], "reset_environment"): 
            for im in self.inter_models:
                im.reset_environment(class_index, num_objects, first_obj_dim)

    def get_hot_passive_mask(self, batch_size, num_keys, num_queries): # if called with a numpy input, needs to be unwrapped
        return get_hot_mask(self.num_clusters, batch_size, num_keys, num_queries, 0, self.iscuda)

    def get_hot_full_mask(self, batch_size, num_keys, num_queries):
        return get_hot_mask(self.num_clusters, batch_size, num_keys, num_queries, 1, self.iscuda)

    def get_all_mask(self, batch_size, num_keys, num_queries):
        return get_hot_mask(self.num_clusters, batch_size, num_keys, num_queries, -1, self.iscuda)

    def get_passive_mask(self, batch_size, num_keys, num_queries):
        return get_passive_mask(batch_size, num_keys, num_queries, self.num_objects, self.class_index, self.iscuda)

    def get_active_mask(self, batch_size, num_keys, num_queries):
        return get_active_mask(batch_size, num_keys, num_queries, self.iscuda)

    def forward(self, x, hard=False, return_selection=False):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = apply_symmetric(self.symmetric_key_query, x)
        s = self.selection_network.forward(x)
        s = self.softmax(s)
        s = s.reshape(s.shape[0], -1, self.num_clusters) # (batch, num_keys, num_clusters)
        if hard: s = self.hard_inter_dist(s).sample()
        else: s = self.soft_inter_dist(self.selection_temperature, probs=s).rsample()
        inters = [self.inter_models[i].forward(x).reshape(x.shape[0], self.num_keys, -1) for i in range(self.num_clusters)] # expects to get output of shape (batch, num_keys, num_clusters, num_instances)
        batch_size, nk, nq = s.shape[0], s.shape[1], inters[0].shape[-1] # TODO: relies on having at num_clusters > 3
        # inters = torch.stack([self.get_passive_mask(batch_size, nk, nq), self.get_active_mask(batch_size, nk, nq)] + inters, dim=2)
        inters = torch.stack(inters, dim=2)
        inters = inters * s.unsqueeze(-1)
        v = inters.sum(dim=2).reshape(x.shape[0], -1)
        # print(inters[:3], v[:3], s[:3])
        if return_selection: return v, s.reshape(s.shape[0], -1)
        return v

