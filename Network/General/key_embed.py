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

class EmbedPairNetwork(Network): 
    # like the keypair network, but applies masks very late, and does not append keys to queries
    # has the shape: [key -> key embedding mlp -> + queries -> pairnet] -> masking ->  reduce function -> mlp -> output
    # part in brackets can be precomputed
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # expects that object_dim is the same for the targets and the values
        self.single_obj_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.query_aggregate = args.embedpair.query_aggregate
        self.reduce_fn = args.pair.reduce_function
        self.conv_dim = self.hs[-1] if self.aggregate_final else args.num_outputs
        self.query_pair = not args.pair.query_pair
        self.total_obj_dim = args.pair.total_obj_dim # total = single + object 
        self.expand_dim = args.pair.expand_dim
        self.embed_dim = args.embed_inputs
        self.total_instances = args.pair.total_instances
        self.return_mask = args.mask_attn.return_mask

        if True: # args.pair.needs_embed:
            embed_query_layer_args = copy.deepcopy(args)
            embed_query_layer_args.object_dim = self.object_dim
            embed_query_layer_args.pair.first_obj_dim = self.single_obj_dim
            embed_query_layer_args.num_outputs = self.embed_dim
            embed_query_layer_args.activation_final = embed_query_layer_args.activation
            embed_query_layer_args.pair.aggregate_final = False
            embed_query_layer_args.include_last = True
            embed_query_layer_args.pair.num_pair_layers = 1 # TODO: multilayer possible if the mask is applied at EVERY layer
            self.embed_query_layer = PairNetwork(embed_query_layer_args)
        
        decode_layer_args = copy.deepcopy(args)
        decode_layer_args.hidden_sizes = args.pair.final_layers
        decode_layer_args.include_last = not args.pair.aggregate_final # False
        decode_layer_args.activation_final = decode_layer_args.activation if args.pair.aggregate_final else decode_layer_args.activation_final
        if self.query_aggregate: # an MLP using the resulting embedding
            decode_layer_args.num_inputs = self.embed_dim
            decode_layer_args.num_outputs = self.conv_dim
            self.decode_layer = MLPNetwork(decode_layer_args)
        else: # a pairnet using the embedding as the key
            decode_layer_args.object_dim = self.object_dim
            decode_layer_args.pair.preencode = True
            decode_layer_args.pair.first_obj_dim = self.embed_dim
            decode_layer_args.num_outputs = self.conv_dim
            decode_layer_args.pair.aggregate_final = False
            decode_layer_args.pair.num_pair_layers = 1 # TODO: multilayer possible if the mask is applied at EVERY layer
            self.decode_layer = PairNetwork(decode_layer_args)
        layers = [self.embed_query_layer, self.decode_layer]
        # layers = [self.conv_layers, self.decode_layer]

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
        print("agf", args.pair.aggregate_final)

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.total_instances = num_objects

    def slice_mask_input(self, x, i, m):
        key = x[...,i * self.single_obj_dim: (i+1) * self.single_obj_dim]
        # print(key.shape, self.embed_key_layer, x.shape)
        prequeries = x[...,self.first_obj_dim:]
        queries = torch.cat([key, prequeries], axis=-1) # attach the key to the start, pairnet handles slicing
        # print(self.first_obj_dim, x.shape, self.object_dim, self.single_obj_dim)
        queries = self.embed_query_layer(queries)
        # print(queries.transpose(2,1).reshape(x.shape[0], -1)[0])
        queries = queries.view(queries.shape[0], self.total_instances, -1)
        if m is not None: # unmasked if m is None
            # print(x[0], xi[0])
            queries = queries * m[:,i].unsqueeze(-1)
        return queries, prequeries
    
    def reappend_queries(self, x, xi):
        return torch.cat([xi, x[...,self.embed_dim:]], dim=-1)

    def forward(self, x, m=None):
        # iterate over each instance
        batch_size = x.shape[0]
        value = list()
        # print(self.first_obj_dim, self.single_obj_dim)
        num_keys = int(self.first_obj_dim // self.single_obj_dim)
        if m is not None: m = m.view(m.shape[0], num_keys, self.total_instances)
        for i in range(num_keys):
            xi, pq = self.slice_mask_input(x, i, m)
            # print(m, xi, pq)
            xi = reduce_function(self.reduce_fn, xi, dim=1)[:,0]
            if self.query_aggregate: xi = self.decode_layer(xi)
            else: xi = self.decode_layer(torch.cat([xi, pq], axis=-1))
            value.append(xi)
       # print(value[0].shape, xil.shape, xi.shape)
        x = torch.stack(value, dim=2) # [batch size, pairnet output dim, num_instances]
        # print(x.shape, self.aggregate_final)
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.conv_dim)
            # print(x.shape)
            x = self.MLP(x)
        else:
            x = x.transpose(2,1)
            # print(x.shape)
            x = x.reshape(batch_size, -1)
        # if m is not None: 
        #     print(x.shape, m.shape, )
        #     print(x[0:10], m[0])
        if self.return_mask: return m, x
        return x
