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

class LinearKeyPairNetwork(Network):
    '''
    first_obj_dim defines a set of keys, and the rest of the input are the queries.
    Compares each of the keys with all of the queries by performing a pairnet computation
    with each of the keys separately, then possibly aggregating back together
    no aggregation between queries, instead uses a linear model
    '''
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # expects that object_dim is the same for the targets and the values
        self.real_object_dim = args.pair.real_object_dim if "real_obj_dim" in args.pair else self.object_dim # if the object dim is an embedding, uses this dim for reconstruction
        self.single_obj_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.single_object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.query_aggregate = args.embedpair.query_aggregate
        self.reduce_fn = args.pair.reduce_function
        self.conv_dim = args.num_outputs
        self.query_pair = not args.pair.query_pair
        self.total_obj_dim = args.pair.total_obj_dim # total = single + object 
        self.expand_dim = args.pair.expand_dim
        self.embed_dim = args.embed_inputs
        self.total_instances = args.pair.total_instances
        self.return_mask = args.mask_attn.return_mask
        self.num_layers = args.pair.num_pair_layers
        self.repeat_layers = args.pair.repeat_layers
        self.new_embedding = args.embedpair.new_embedding

        layers = list()
        if self.new_embedding:
            
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
            embed_query_layer_args.pair.aggregate_final = False
            embed_query_layer_args.include_last = True
            self.embed_query_layer = ConvNetwork(embed_query_layer_args)

            layers = list()
            self.conv_layers = list()
            pair_args = copy.deepcopy(args)
            pair_args.pair.first_obj_dim = self.embed_dim
            pair_args.pair.object_dim = self.embed_dim
            pair_args.pair.num_pair_layers = 1 # use multilayer at keypair level instead of pair level
            pair_args.include_last = False
            pair_args.pair.aggregate_final = False
            pair_args.activation_final = pair_args.activation
            pair_args.num_outputs = self.embed_dim
            self.pair_network = PairNetwork(pair_args)
            layers = [self.embed_key_layer, self.embed_query_layer, self.pair_network]
        else:
            self.embed_dim = self.obj_dim # embedding is given as object_dim. TODO: alternatively single obj + object_dim
        
        decode_layer_args = copy.deepcopy(args)
        decode_layer_args.hidden_sizes = list()
        decode_layer_args.include_last = False
        decode_layer_args.activation_final = "none"
        decode_layer_args.num_inputs = self.embed_dim * self.total_instances # TODO: alternatively, we could have a reduce function
        decode_layer_args.num_outputs = self.conv_dim
        decode_layer_args.use_bias = False # TODO: we could do affine or linear
        self.decode_layer = MLPNetwork(decode_layer_args) # network is linear

        reconstruct_layer_args = copy.deepcopy(args)
        reconstruct_layer_args.hidden_sizes = [256]
        reconstruct_layer_args.object_dim = self.embed_dim
        reconstruct_layer_args.output_dim = self.real_object_dim # must have embed_inputs
        reconstruct_layer_args.activation_final = "none"
        reconstruct_layer_args.pair.aggregate_final = False
        reconstruct_layer_args.include_last = True
        self.reconstruction = ConvNetwork(reconstruct_layer_args)

        layers = layers + [self.reconstruction, self.decode_layer]
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

    def slice_input(self, x, i):
        # slices x into keys and queries, embeds them, masks the queries and then creates 
        # a vector of [batchssize, num_queries (self.total_instances), key_embed_dim + query_embed_dim]
        # assumes masks is already expanded to be the [batchsize, num_queries*self.embed_dim * self.total_instances]
        if self.new_embedding: # we need a new embedding for keys and queries, computed here and processed for a pointnet
            key = x[...,i * self.single_obj_dim: (i+1) * self.single_obj_dim]
            key = self.embed_key_layer(key)
            queries = x[...,self.first_obj_dim:]
            queries = queries.reshape(x.shape[0], -1, self.object_dim).transpose(1,2)
            queries = self.embed_query_layer(queries).transpose(2,1).reshape(x.shape[0], -1)
            xi = torch.cat([key, queries], dim=-1)
        else: # create a convnet 
            xi = x[...,self.first_obj_dim:]
            # # TODO: we could append keys, but we don't
            # key = x[...,i * self.single_obj_dim: (i+1) * self.single_obj_dim]
            # key = torch.broadcast_to(key.unsqueeze(-1), (x.shape[0], queries.shape[1], self.object_dim))
        return xi
    
    def reappend_queries(self, x, xi):
        return torch.cat([xi, x[...,self.embed_dim:]], dim=-1)

    def forward(self, x, m=None, return_embeddings=False, return_reconstruction=False, valid=None):
        # iterate over each instance
        batch_size = x.shape[0]
        value = list()
        embeddings = list()
        # print(self.first_obj_dim, self.single_obj_dim)
        # apply validity and expand mask to embed dim
        if m is not None:
            if valid is not None:
                m = m * valid # invalidate through the mask
            m = expand_mask(m, batch_size, self.embed_dim)
        
        for i in range(int(self.first_obj_dim // self.single_obj_dim)):
            xi = self.slice_input(x, i)
            if self.new_embedding:
                xi = self.pair_network(xi)
                embed = xi.reshape(batch_size, self.total_instances, self.embed_dim)
                embeddings.append(self.reconstruction(embed.transpose(1,2)).transpose(2,1) if return_reconstruction else embed)
            else:
                embeddings.append(self.reconstruction(xi.reshape(x.shape[0], -1, self.object_dim).transpose(1,2)).transpose(2,1).reshape(batch_size, -1) if return_reconstruction else embed)
            if m is not None: xi = xi * m[...,i * self.total_instances: (i+1)* self.total_instances].reshape(batch_size, self.total_instances, 1).expand(-1,  self.total_instances, self.embed_dim).reshape(-1, self.total_instances* self.embed_dim)
            xi = self.decode_layer(xi)
            # print(xil.shape)
            value.append(xi)
        if return_embeddings:
            return torch.stack(embeddings, dim=1), x[...,self.first_obj_dim:].clone().detach() # batch, keys, queries, embed_dim, batch, queries * object_dim
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
