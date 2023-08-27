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
from Network.Dists.mask_utils import expand_mask

class KeyPairNetwork(Network):
    '''
    first_obj_dim defines a set of keys, and the rest of the input are the queries.
    Compares each of the keys with all of the queries by performing a pairnet computation
    with each of the keys separately, then possibly aggregating back together
    '''
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # expects that object_dim is the same for the targets and the values
        self.single_obj_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.single_object_dim
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
        self.num_layers = args.pair.num_pair_layers
        self.repeat_layers = args.pair.repeat_layers

        embed_key_layer_args = copy.deepcopy(args)
        embed_key_layer_args.hidden_sizes = list()
        embed_key_layer_args.num_inputs = self.single_obj_dim
        embed_key_layer_args.num_outputs = self.embed_dim # must have embed_inputs
        embed_key_layer_args.activation_final = embed_key_layer_args.activation
        embed_key_layer_args.pair.aggregate_final = self.query_pair # the pair net computes over all queries, so we only need a single output for any one target
        self.embed_key_layer = MLPNetwork(embed_key_layer_args)
            
        embed_query_layer_args = copy.deepcopy(args)
        embed_query_layer_args.hidden_sizes = list()
        embed_query_layer_args.object_dim = self.object_dim
        embed_query_layer_args.output_dim = self.embed_dim # must have embed_inputs
        embed_query_layer_args.activation_final = embed_query_layer_args.activation
        embed_query_layer_args.pair.aggregate_final = self.query_pair # the pair net computes over all queries, so we only need a single output for any one target
        embed_query_layer_args.include_last = True
        self.embed_query_layer = ConvNetwork(embed_query_layer_args)

        layers = list()
        self.conv_layers = list()
        pair_args = copy.deepcopy(args)
        pair_args.pair.first_obj_dim = self.embed_dim
        pair_args.pair.object_dim = self.embed_dim
        pair_args.pair.num_pair_layers = 1 # use multilayer at keypair level instead of pair level
        pair_args.include_last = False
        pair_args.pair.aggregate_final = True
        pair_args.activation_final = pair_args.activation
        pair_args.num_outputs = self.embed_dim
        if not self.repeat_layers:
            for i in range(self.num_layers): # we need num_layers pair networks
                self.conv_layers.append(PairNetwork(pair_args))
        else:
            self.conv_layers.append(PairNetwork(pair_args))
        
        decode_layer_args = copy.deepcopy(args)
        decode_layer_args.hidden_sizes = args.pair.final_layers
        decode_layer_args.include_last = not args.pair.aggregate_final # False
        decode_layer_args.activation_final = decode_layer_args.activation if args.pair.aggregate_final else decode_layer_args.activation_final
        if self.query_aggregate: # an MLP using the resulting embedding
            decode_layer_args.num_inputs = self.embed_dim
            decode_layer_args.num_outputs = self.conv_dim
            self.decode_layer = MLPNetwork(decode_layer_args)
        else: # a pairnet using the embedding as the key
            decode_layer_args.pair.object_dim = self.embed_dim
            decode_layer_args.pair.preencode = True
            decode_layer_args.pair.first_obj_dim = self.embed_dim
            decode_layer_args.num_outputs = self.conv_dim
            decode_layer_args.pair.aggregate_final = False
            decode_layer_args.pair.num_pair_layers = 1 # TODO: multilayer possible if the mask is applied at EVERY layer
            self.decode_layer = PairNetwork(decode_layer_args)

        self.conv_layers = nn.ModuleList(self.conv_layers)
        layers = [self.embed_key_layer, self.embed_query_layer, self.conv_layers, self.decode_layer]
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

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.total_instances = num_objects

    def slice_mask_input(self, x, i, m):
        # slices x into keys and queries, embeds them, masks the queries and then creates 
        # a vector of [batchssize, num_queries (self.total_instances), key_embed_dim + query_embed_dim]
        # assumes masks is already expanded to be the [batchsize, num_queries*self.embed_dim * self.total_instances]
        key = x[...,i * self.single_obj_dim: (i+1) * self.single_obj_dim]
        key = self.embed_key_layer(key)
        queries = x[...,self.first_obj_dim:]
        # print(key.shape, self.embed_query_layer, x.shape, queries.shape,queries.reshape(x.shape[0], -1, self.object_dim).transpose(1,2).shape, self.object_dim, self.first_obj_dim, self.single_obj_dim)
        queries = queries.reshape(x.shape[0], -1, self.object_dim).transpose(1,2)
        # print(queries.shape, self.first_obj_dim, x.shape, self.object_dim, self.single_obj_dim)
        queries = self.embed_query_layer(queries).transpose(2,1).reshape(x.shape[0], -1)
        # print(queries.transpose(2,1).reshape(x.shape[0], -1)[0])
        if m is None: # unmasked if m is None
            xi = torch.cat([key, queries], dim=-1)
            # print(x[0], xi[0])
        else:
            total_obj_dim  = self.embed_dim * self.total_instances
            # print(m.shape, self.embed_dim, self.total_instances, x.shape, self.first_obj_dim, queries.shape)
            # print(x.shape, total_obj_dim, self.embed_dim, self.total_instances, key.shape, queries.shape, m.shape, i)
            # print((queries * m[...,i * total_obj_dim:(i+1) * total_obj_dim])[0:10])
            xi = torch.cat([key, queries * m[...,i * total_obj_dim:(i+1) * total_obj_dim]], dim=-1) # [batch, embed_dim + embed_dim * total_instances]
        # print(key[0], queries[0], xi[0], x[0])
        # print(m[0][:512])
        # print(key.shape, queries.shape, xi.shape, self.first_obj_dim, m.shape, x.shape, i, self.single_obj_dim)
        return xi
    
    def reappend_queries(self, x, xi):
        return torch.cat([xi, x[...,self.embed_dim:]], dim=-1)

    def forward(self, x, m=None, valid=None):
        # iterate over each instance
        batch_size = x.shape[0]
        value = list()
        # print(self.first_obj_dim, self.single_obj_dim)
        if m is not None:
            if valid is not None:
                m = m * valid # invalidate through the mask
            m = expand_mask(m, batch_size, self.embed_dim)
        for i in range(int(self.first_obj_dim // self.single_obj_dim)):
            xi = self.slice_mask_input(x, i, m)
            # print(i, x.shape, xi.shape, self.query_aggregate, self.single_obj_dim, self.first_obj_dim, x.shape[-1] - self.first_obj_dim, self.first_obj_dim // self.single_obj_dim)
            # print(self.pair_net.aggregate_final, self.pair_net.num_outputs, self.pair_net.first_obj_dim)
            xil = xi
            # print(x.shape, xi.shape, self.num_layers)
            for j in range(self.num_layers):
                l = self.conv_layers[0] if self.repeat_layers else self.conv_layers[j]
                # print(self.first_obj_dim, self.object_dim, l.first_obj_dim, l.object_dim)
                xil = l(xil) # [batch, embed_dim]
                if j < self.num_layers - 1: xil = self.reappend_queries(xi, xil) # [batch, embed + embed_dim + embed_dim * total_instances]
                # print(j, self.num_layers - 1, xi.shape, xil.shape,self.embed_dim)
            # print(xil.shape, self.decode_layer)
            # print("xil", xil.shape, self.reappend_queries(xi, xil).shape)
            # xil = self.decode_layer(xil)
            if self.query_aggregate: xil = self.decode_layer(xil)
            else: xil = self.decode_layer(self.reappend_queries(xi, xil))
            # print(xil.shape)
            value.append(xil)
        # print(value[0].shape, xil.shape, xi.shape)
        x = torch.stack(value, dim=2) # [batch size, pairnet output dim, num_instances]
        # print(x.shape, self.query_aggregate)
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
