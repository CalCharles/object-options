import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, network_type
from Network.network_utils import pytorch_model, get_acti
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork
import copy

class DiagGaussianForwardNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        self.mean = network_type[args.net_type](mean_args)
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance

        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + self.base_variance

class InteractionNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = 1
        inter_args.activation_final = "sigmoid"
        self.inter = network_type[args.net_type](inter_args)
        self.model = [self.inter]

        self.train()
        self.reset_network_parameters()
        
    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        return v

class DiagGaussianForwardMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        self.mean = network_type[args.net_type](mean_args)
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance

        self.total_object_sizes = [args.total_object_sizes[n] for n in args.object_names]

        self.train()
        self.reset_network_parameters()

    def expand_mask(self, m):
        # m = batch x num_objects
        # TODO: make this not a for loop
        comb = list()
        for i in range(m.shape[-1]):
            comb.append(m[...,i] * pytorch_model.wrap(torch.ones(self.total_object_sizes[i]), cuda=self.iscuda))
        return torch.cat(comb, dim=-1)

    def forward(self, x, m):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        if not self.hot: x = x * self.expand_mask(m)
        mask, mean = self.mean(x)
        _, var = self.std(x)
        return torch.tanh(mean), torch.sigmoid(var) + self.base_variance, mask

class DiagGaussianForwardPadMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        self.mean = network_type[args.net_type](mean_args)
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance
        self.cluster_mode = args.mask_attn.cluster
        self.num_clusters = args.mask_attn.num_clusters
        self.maskattn = args.net_type in ["maskattn"] # currently only one kind of mask attention net
        self.mask_dim = args.pair.total_instances # does not handle arbitary number of instances

        self.object_dim = args.object_dim

        self.train()
        self.reset_network_parameters()

    def get_masks(self):
        if self.maskattn:
            return 
        else:
            return self

    def expand_mask(self, m):
        # m = batch x num_objects
        # TODO: make this not a for loop
        comb = list()
        for i in range(m.shape[-1]):
            comb.append(m[...,i].unsqueeze(-1) * pytorch_model.wrap(torch.ones(self.object_dim), cuda=self.iscuda))
        return torch.cat(comb, dim=-1)

    def forward(self, x, m):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        if not (self.cluster_mode or self.maskattn): m = self.expand_mask(m)
        mean = self.mean(x, m)
        var = self.std(x, m)
        return (torch.tanh(mean), torch.sigmoid(var) + self.base_variance), m

class InteractionMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "sigmoid"
        inter_args.mask_attn.needs_encoding = True
        self.inter = network_type[args.net_type](inter_args)
        self.model = [self.inter]

        self.train()
        self.reset_network_parameters()
        
    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        return v


class DiagGaussianForwardPadHotNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        self.cluster_mode = args.mask_attn.cluster
        self.num_clusters = args.mask_attn.num_clusters
        self.object_dim = args.pair.object_dim # the object dim is the dimension of the value input
        self.single_object_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.embed_dim = args.mask_attn.embed_dim
        self.model_dim = args.mask_attn.model_dim

        # COPIED FROM mask_attention.py
        args.include_last = True
        key_args = copy.deepcopy(args)
        key_args.object_dim = args.pair.single_obj_dim
        key_args.output_dim = self.embed_dim
        key_args.hidden_sizes = args.pair.final_layers
        key_args.use_layer_norm = False
        key_args.pair.aggregate_final = False
        self.key_encoding = ConvNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.object_dim = args.pair.object_dim
        query_args.output_dim = self.embed_dim
        query_args.hidden_sizes = args.pair.final_layers
        query_args.use_layer_norm = False
        query_args.pair.aggregate_final = False
        self.query_encoding = ConvNetwork(query_args)

        # inter models must operate by pointnet principles to be instance invariant
        inter_args = copy.deepcopy(args)
        inter_args.net_type = "keypair"
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "sigmoid"
        inter_args.hidden_sizes = args.mask_attn.cluster_inter_hidden
        inter_args.pair.num_pair_layers = 2
        inter_args.aggregate_final = False
        self.inter_models = nn.ModuleList([network_type[inter_args.net_type](inter_args) for i in range(self.num_clusters - 2)]) # two clusters reserved, one for passive and one for full
        print(self.inter_models)

        # forward networks
        forward_args = copy.deepcopy(args)
        forward_args.mask_attn.needs_encoding = False
        self.means = nn.ModuleList([network_type[args.net_type](forward_args) for i in range(self.num_clusters)])
        self.stds = nn.ModuleList([network_type[args.net_type](forward_args) for i in range(self.num_clusters)])

        self.passive_mask = args.mask_attn.passive_mask.astype(np.float32)



        layers = [self.key_encoding, self.query_encoding, self.means, self.stds]
        self.model = layers
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance
        self.maskattn = args.net_type in ["maskattn"] # currently only one kind of mask attention net
        self.mask_dim = args.pair.total_instances # does not handle arbitary number of instances

        self.object_dim = args.object_dim

        self.train()
        self.reset_network_parameters()

    def reset_environment(self, passive_mask, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.passive_mask = passive_mask.astype(np.float32)

    def slice_input(self, x):
        keys = torch.stack([x[...,i * self.single_object_dim: (i+1) * self.single_object_dim] for i in range(int(self.first_obj_dim // self.single_object_dim))], dim=-2) # [batch size, num keys, single object dim]
        queries = torch.stack([x[...,self.first_obj_dim + j * self.object_dim:self.first_obj_dim + (j+1) *self.object_dim] for j in range(int((x.shape[-1] - self.first_obj_dim) // self.object_dim))], dim=-2) # [batch size, num queries, object dim]
        # TODO: add relative value calculation
        keys, queries = keys.transpose(-2,-1), queries.transpose(-2,-1)
        return keys, queries

    def count_keys_queries(self, x):
        return self.first_obj_dim // self.single_object_dim, int((x.shape[-1] - self.first_obj_dim) // self.object_dim)


    def get_hot_passive_mask(self, batch_size, num_keys, num_queries): # if called with a numpy input, needs to be unwrapped
        if batch_size <= 0:
            hot_passive = pytorch_model.wrap(torch.zeros(num_keys, self.num_clusters), cuda=self.iscuda)
            hot_passive[...,0] = 1
            return hot_passive 
        hot_passive = pytorch_model.wrap(torch.zeros(batch_size, num_keys, self.num_clusters), cuda=self.iscuda)
        hot_passive[...,0] = 1
        return hot_passive 

    def get_hot_full_mask(self, batch_size, num_keys, num_queries):
        if batch_size <= 0:
            hot_active = pytorch_model.wrap(torch.zeros(num_keys, self.num_clusters), cuda=self.iscuda)
            hot_active[...,1] = 1
            return hot_active 
        hot_active = pytorch_model.wrap(torch.zeros(batch_size, num_keys, self.num_clusters), cuda=self.iscuda)
        hot_active[...,1] = 1
        # print("hot_active", hot_active.shape, hot_active)
        return hot_active

    def get_passive_mask(self, batch_size, num_keys, num_queries):
        if batch_size <= 0: return pytorch_model.wrap(torch.ones(num_keys, 1) * self.passive_mask, cuda=self.iscuda)
        return pytorch_model.wrap(torch.ones(batch_size, num_keys, 1) * self.passive_mask, cuda=self.iscuda)

    def get_active_mask(self, batch_size, num_keys, num_queries):
        if batch_size <= 0: return pytorch_model.wrap(torch.ones(num_keys, num_queries), cuda=self.iscuda)
        return pytorch_model.wrap(torch.ones(batch_size, num_keys, num_queries), cuda=self.iscuda)

    def compute_cluster_masks(self, x, m, num_keys, num_queries):
        passive_masks = [self.get_passive_mask(x.shape[0], num_keys, num_queries)] # broadcast to batch size
        active_masks = [self.get_active_mask(x.shape[0], num_keys, num_queries)]
        # print(passive_masks[0], active_masks[0], self.inter_models[0](x).reshape(x.shape[0], num_keys, -1))
        all_masks = torch.stack(passive_masks + active_masks + [self.inter_models[i](x).reshape(x.shape[0], num_keys, -1) for i in range(self.num_clusters - 2)], axis=0)
        # all masks shape: num_clusters, batch size, num_keys, num_queries 
        # print(all_masks.shape, m.shape, x.shape)
        m = m.reshape(x.shape[0], num_keys, self.num_clusters).transpose(0,2).transpose(1,2).unsqueeze(-1) # flip clusters to the front, flip keys and num_batch add a dimension for queries broadcasting
        # print("mask shapes", all_masks.shape, m.shape, x.shape)
        # print(m, all_masks, (all_masks * m).sum(0))
        return all_masks, (all_masks * m).sum(0).reshape(x.shape[0], -1)

    def compute_clusters(self, cluster_nets, keys, queries, masks, m):
        # keys of shape n_batch, n_keys, d_keys
        # queries of shape n_batch n_queries d_queries
        # masks pf shape n_batch n_keys n_queries
        # m of shape n_batch n_keys n_cluster
        total_out = list()
        for i in range(self.num_clusters): # we could probably do this in parallel
            # print("inp mask", masks[i], m, m.shape, keys.shape, queries.shape, masks.shape)
            total_out.append(cluster_nets[i]((keys, queries, masks[i])).reshape(keys.shape[0], keys.shape[-1],-1)) # [batch, num_keys, single_obj_dim] x num_clusters
        # print(torch.stack(total_out, dim=-1).shape, m.unsqueeze(-2).shape, (torch.stack(total_out, dim=-1) * m.unsqueeze(-2)).sum(-1).shape)
        return (torch.stack(total_out, dim=-1) * m.unsqueeze(-2)).sum(-1).reshape(keys.shape[0], -1) # batch size x n_keys * single_obj_dim

    def forward(self, x, m):
        # x: batch size, single_object_dim * num_keys (first_obj_dim) + object_dim * num_queries
        # m: batch size, num_keys * num_clusters
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        keys, queries = self.slice_input(x)
        keys = self.key_encoding(keys) # [batch size, num keys, single object dim]
        queries = self.query_encoding(queries) # [batch size, num queries, object dim]
        # print(keys.shape, queries.shape, m)
        inter_masks, total_mask = self.compute_cluster_masks(x, m, keys.shape[-1], queries.shape[-1])
        # print(m.shape)
        # print("masks", inter_masks, total_mask)
        # m = m.reshape(self.num_clusters, x.shape[0], keys.shape[-1], -1)
        queries = queries.transpose(-2,-1)
        mean = self.means[0]((keys, queries, inter_masks[0]))
        var = self.stds[0]((keys, queries, inter_masks[0]))
        # mean = self.compute_clusters(self.means, keys, queries, inter_masks, m)
        # var = self.compute_clusters(self.stds, keys, queries, inter_masks, m)
        # print(mean.shape, var.shape)
        return (torch.tanh(mean), torch.sigmoid(var) + self.base_variance), inter_masks