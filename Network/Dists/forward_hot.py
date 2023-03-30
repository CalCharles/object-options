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
import copy, time
from Network.Dists.mask_utils import expand_mask, apply_probabilistic_mask, count_keys_queries


MASK_ATTENTION_TYPES = ["maskattn"] # currently only one kind of mask attention net


class DiagGaussianForwardPadHotNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        
        self.maskattn = args.net_type in MASK_ATTENTION_TYPES # currently only one kind of mask attention net

        self.cluster_mode = args.cluster.cluster_mode
        self.num_clusters = args.cluster.num_clusters
        self.needs_key_query = args.net_type in MASK_ATTENTION_TYPES
        self.embed_dim = args.embed_inputs * max(1, int(self.maskattn) * args.mask_attn.num_heads )
        self.object_dim = args.pair.object_dim # the object dim is the dimension of the value input
        self.single_obj_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.single_obj_dim
        self.model_dim = args.mask_attn.model_dim
        self.symmetric_key_query = args.symmetric_key_query # if these are the same, then we need to cat x to itself to fit key-query expectations

        # distributional parameters
        self.inter_dist = args.mask_attn.inter_dist
        self.relaxed_inter_dist = args.mask_attn.relaxed_inter_dist
        self.dist_temperature = args.mask_attn.dist_temperature
        self.test = args.mask_attn.test

        # COPIED FROM mask_attention.py
        if self.embed_dim > 0:
            args.include_last = True
            key_args = copy.deepcopy(args)
            key_args.object_dim = args.pair.single_obj_dim
            key_args.output_dim = self.embed_dims
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
        inter_args.hidden_sizes = args.cluster.cluster_inter_hidden
        inter_args.pair.num_pair_layers = args.cluster.inter_pair_layers
        inter_args.aggregate_final = False
        # if self.embed_dim > 0:
        #     inter_args.pair.first_obj_dim = int(self.embed_dim * (self.first_obj_dim / self.single_obj_dim))
        #     inter_args.pair.single_obj_dim = self.embed_dim
        #     inter_args.pair.object_dim = self.embed_dim 
        self.inter_models = nn.ModuleList([network_type[inter_args.net_type](inter_args) for i in range(self.num_clusters - 2)]) # two clusters reserved, one for passive and one for full
        # print(self.inter_models)

        # forward networks
        forward_args = copy.deepcopy(args)
        forward_args.mask_attn.needs_encoding = False
        if self.embed_dim > 0:
            forward_args.pair.single_obj_dim = self.embed_dim
            forward_args.pair.aggregate_final = False
            forward_args.pair.object_dim = self.embed_dim 
            forward_args.pair.first_obj_dim = int(self.embed_dim * (self.first_obj_dim / self.single_obj_dim))
        self.means = nn.ModuleList([network_type[args.net_type](forward_args) for i in range(self.num_clusters)])
        self.stds = nn.ModuleList([network_type[args.net_type](forward_args) for i in range(self.num_clusters)])

        layers = [self.key_encoding, self.query_encoding, self.means, self.stds, self.inter_models]
        self.model = layers
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance
        self.mask_dim = args.pair.total_instances # does not handle arbitary number of instances

        self.object_dim = args.object_dim

        self.train()
        self.reset_network_parameters()

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.class_index = class_index
        self.num_objects = num_objects
        if hasattr(self.inter_models[0], "reset_environment"): 
            for im in self.inter_models:
                im.reset_environment(class_index, num_objects, first_obj_dim)
        if hasattr(self.means[0], "reset_environment"): 
            for m in self.means:
                m.reset_environment(class_index, num_objects, first_obj_dim)
            for s in self.stds:
                s.reset_environment(class_index, num_objects, first_obj_dim)

    def slice_input(self, x):
        keys = torch.stack([x[...,i * self.single_obj_dim: (i+1) * self.single_obj_dim] for i in range(int(self.first_obj_dim // self.single_obj_dim))], dim=-2) # [batch size, num keys, single object dim]
        queries = torch.stack([x[...,self.first_obj_dim + j * self.object_dim:self.first_obj_dim + (j+1) *self.object_dim] for j in range(int((x.shape[-1] - self.first_obj_dim) // self.object_dim))], dim=-2) # [batch size, num queries, object dim]
        # TODO: add relative value calculation
        keys, queries = keys.transpose(-2,-1), queries.transpose(-2,-1)
        return keys, queries

    def count_keys_queries(self, x):
        return count_keys_queries(self.first_obj_dim, self.single_obj_dim, self.object_dim, x)

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

    def get_inter_mask(self, i, x, num_keys, soft, mixed, flat):
        inter = self.inter_models[i](x).reshape(x.shape[0], num_keys, -1)
        # print(inter[:4], self.inter_dist, self.relaxed_inter_dist, ((not soft) or (soft and mixed)), (soft and not mixed), mixed, flat, self.dist_temperature)
        return apply_probabilistic_mask(inter, inter_dist=self.inter_dist if ((not soft) or (soft and mixed)) else None,
                                        relaxed_inter_dist=self.relaxed_inter_dist if (soft and not mixed) else None, 
                                        mixed=mixed, test=self.test if flat else None, dist_temperature=self.dist_temperature, 
                                        revert_mask=False)

    def compute_cluster_masks(self, x, m, num_keys, num_queries, soft=False, mixed=False, flat=False):
        # returns all the interactions masks computed by all the models, as well as the masks weighted by their true values
        passive_masks = [self.get_passive_mask(x.shape[0], num_keys, num_queries)] # broadcast to batch size
        active_masks = [self.get_active_mask(x.shape[0], num_keys, num_queries)]
        # print(passive_masks[0], active_masks[0], self.inter_models[0](x).reshape(x.shape[0], num_keys, -1))

        all_masks = torch.stack(passive_masks + active_masks + [self.get_inter_mask(i, x, num_keys, soft, mixed, flat) for i in range(self.num_clusters - 2)], axis=0)
        # all masks shape: num_clusters, batch size, num_keys, num_queries 
        # print(all_masks.shape, m.shape, x.shape, num_keys, self.num_clusters, m.reshape(x.shape[0], num_keys, self.num_clusters).shape)
        # print(torch.transpose(m.reshape(x.shape[0], num_keys, self.num_clusters), 0, 2).shape)
        m = m.reshape(x.shape[0], num_keys, self.num_clusters).transpose(0,2).transpose(1,2).unsqueeze(-1) # flip clusters to the front, flip keys and num_batch add a dimension for queries broadcasting
        # print("mask shapes", all_masks.shape, m.shape, x.shape)
        # print(soft, m[:,:4].squeeze(), all_masks[:,:4].squeeze(), (all_masks * m)[:,:4].sum(0).squeeze())
        # print(all_masks[2,0])
        return all_masks, (all_masks * m).sum(0).reshape(x.shape[0], -1)

    def compute_clusters(self, cluster_nets, keys, queries, masks, m, full=False):
        # keys of shape n_batch, n_keys, d_keys
        # queries of shape n_batch n_queries d_queries
        # masks pf shape n_batch n_keys n_queries
        # m of shape n_batch n_keys n_cluster
        total_out = list()
        # print("mask", m[:3])
        for i in range(self.num_clusters): # we could probably do this in parallel
            # print("inp mask", masks[i], m, m.shape, keys.shape, queries.shape, masks.shape)
            if self.needs_key_query:
                total_out.append(cluster_nets[i]((keys, queries, masks[i])).reshape(keys.shape[0], keys.shape[-1],-1)) # [batch, num_keys, single_obj_dim] x num_clusters
            else:
                mask_dim = self.embed_dim if self.embed_dim > 0 else self.object_dim
                # print("cluster in", keys.shape, masks[i].shape, masks[i].reshape(masks[i].shape[0],-1).shape, expand_mask(masks[i].reshape(masks[i].shape[0],-1), mask_dim,-1).shape)
                                # , cluster_nets[i](keys, 
                                # expand_mask(masks[i], mask_dim,-1), mask_dim, self.iscuda)).shape)
                exp_mask = expand_mask(masks[i].reshape(masks[i].shape[0],-1), masks[i].shape[0], mask_dim)
                # print("cluster in", exp_mask, keys)
                total_out.append(cluster_nets[i](keys, 
                                exp_mask)
                                .reshape(keys.shape[0], queries.shape[1],-1)) # queries is the keys
            # print(i, masks[i][:3], total_out[i][:3])
        # print(torch.stack(total_out, dim=-1).shape, m.unsqueeze(-2).shape, (torch.stack(total_out, dim=-1) * m.unsqueeze(-2)).sum(-1).shape)
        # print(torch.stack(total_out, dim=-1).shape, torch.stack(total_out, dim=-1).transpose(-2,-1)[:3], torch.stack(total_out, dim=-1).transpose(-2,-1).reshape(keys.shape[0], -1)[:3])
        # print("to", total_out[2][0], full, torch.stack(total_out, dim=1).reshape(keys.shape[0], -1)[0])
        if full: return torch.stack(total_out, dim=1).reshape(keys.shape[0], -1) # batch size x n_keys * single_obj_dim * num_clusters
        return (torch.stack(total_out, dim=-1) * m.unsqueeze(-2)).sum(-1).reshape(keys.shape[0], -1) # batch size x n_keys * single_obj_dim

    def forward(self, x, m, soft=False, mixed=False, flat=False, full=False):
        # x: batch size, single_obj_dim * num_keys (first_obj_dim) + object_dim * num_queries
        # m: batch size, num_keys * num_clusters
        # print("soft, mixed, flat, full", soft, mixed, flat, full)
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = apply_symmetric(self.symmetric_key_query, x)
        keys, queries = self.slice_input(x) # [batch size, object dim, num queries], [batch size, single object dim, num keys]
        keys = self.key_encoding(keys) # [batch size, embed dim, num keys]
        queries = self.query_encoding(queries) # [batch size, embed dim, num queries]
        # print(keys.shape, queries.shape, m)
        inter_masks, total_mask = self.compute_cluster_masks(x, m, keys.shape[-1], queries.shape[-1], soft=soft, mixed=mixed, flat=flat)
        # if full: print(inter_masks.mean(1).squeeze(), soft, mixed)
        if not self.needs_key_query:
            x = torch.cat([keys, queries], dim=-1).transpose(1,2).reshape(keys.shape[0], -1) # [batch size, embed dim * (nk + nq)]
            queries = keys
            keys = x # compute clusters only looks at keys
            # print("kqx", keys.shape, queries.shape, keys, queries, x)
        # print(m.shape)
        # print("masks", inter_masks, total_mask)
        # m = m.reshape(self.num_clusters, x.shape[0], keys.shape[-1], -1)
        queries = queries.transpose(-2,-1)
        # mean = self.means[0]((keys, queries, inter_masks[0]))
        # var = self.stds[0]((keys, queries, inter_masks[0]))
        # print(keys.shape, queries.shape)
        # in the full case, the cluster heads are established as the last layer
        mean = self.compute_clusters(self.means, keys, queries, inter_masks, m, full=full)
        # print("mean", mean[:3])
        var = self.compute_clusters(self.stds, keys, queries, inter_masks, m, full=full)
        # print(mean.shape, var.shape)
        return (torch.tanh(mean), torch.sigmoid(var) + self.base_variance), total_mask