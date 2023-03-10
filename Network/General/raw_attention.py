from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy, time
from Network.network_utils import reduce_function, get_acti, pytorch_model, cuda_string
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork

class MultiHeadAttentionInteractionParallelLayer(Network):
    def __init__(self, args):
        super().__init__(args)
        self.softmax =  nn.Softmax(-1)
        self.model_dim = args.mask_attn.model_dim # the dimension of the keys and queries after network
        self.key_dim = args.embed_inputs # the dimension of the key inputs, must equal args.mask_attn.model_dim * args.mask_attn.num_heads
        self.query_dim = args.embed_inputs
        self.num_heads = args.mask_attn.num_heads
        self.gumbel_temperature = args.mask_attn.gumbel_temperature
        self.merge_function = args.mask_attn.merge_function
        if self.model_dim < 0:
            self.model_dim = args.embed_inputs // self.num_heads
        assert args.embed_inputs % self.num_heads == 0, f"head and key not divisible, key: {args.key_dim}, head: {self.num_heads}"
        self.head_dim = int(args.embed_inputs // self.num_heads) # should be key_dim / num_heads, integer divisible

        # process one key at a time
        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.num_outputs = self.model_dim * self.num_heads
        key_args.hidden_sizes = [hs*self.num_heads for hs in key_args.hidden_sizes]
        # key_args.dropout = args.mask_attn.attention_dropout
        # key_args.use_layer_norm = True
        key_args.activation_final = "none"
        self.key_network = MLPNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim * self.num_heads
        query_args.object_dim = self.query_dim
        query_args.hidden_sizes = [hs*self.num_heads for hs in query_args.hidden_sizes]
        # query_args.dropout = args.mask_attn.attention_dropout
        # query_args.use_layer_norm = True
        query_args.aggregate_final = False
        query_args.activation_final = "none"
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim * self.num_heads
        value_args.pair.object_dim = self.query_dim
        value_args.pair.first_obj_dim = self.key_dim
        value_args.hidden_sizes = [hs*self.num_heads for hs in value_args.hidden_sizes]
        # value_args.dropout = args.mask_attn.attention_dropout
        # value_args.use_layer_norm = True
        value_args.pair.aggregate_final = False
        value_args.activation_final = value_args.activation
        self.value_network = PairNetwork(value_args)

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim * self.num_heads
        final_args.num_outputs = self.key_dim
        final_args.dropout = args.mask_attn.attention_dropout
        # final_args.use_layer_norm = True
        final_args.hidden_sizes = [hs*self.num_heads for hs in final_args.pair.final_layers]
        final_args.activation_final = final_args.activation
        self.final_network = MLPNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def forward(self, key, queries, mask=None, hard=False):
        # generates the mask from the softmax
        # alternatively, broadcasts the mask into the shape of 
        batch_size = key.shape[0]
        # uncomment below if queries = queries * mask.unsqueeze(-1) is commented, or we want safer operations that mask out the whole value
        # if mask is not None: queries = queries * torch.broadcast_to(mask, (batch_size, mask.shape[-1])).unsqueeze(-1)
        # value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1) # todo: relative state operations here
        # if mask is not None: value_inputs = torch.cat([key.unsqueeze(1)] + [queries * torch.broadcast_to(mask, (batch_size, mask.shape[-1])).unsqueeze(-1)], dim=1) # todo: relative state operations here
        # key = key * 0
        # queries[...,2,:] = 0
        value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1)
        # print(value_inputs.shape)
        # print(key[0], queries[0], value_inputs[0])
        # print(value_inputs.reshape(batch_size, -1)[0], value_inputs.shape)
        values = self.value_network(value_inputs.reshape(batch_size, -1)) # -1 dim is num_queries
        values = values.reshape(batch_size, queries.shape[1], self.num_heads, self.model_dim).transpose(1,2)

        # compute the attention component
        key = self.key_network(key)
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
        key, queries = key.reshape(batch_size, self.num_heads, self.model_dim, 1), queries.reshape(batch_size, self.num_heads, -1, self.model_dim)
        full_head_mask = self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim)) # Batch, heads, num queries


        if mask is not None: # if the mask is None,we replace the key-query computation
            # print(mask.shape, batch_size, self.num_heads, mask.shape[-1])
            if len(mask.shape) > 1: ret_mask = mask.unsqueeze(1)
            full_head_dist_mask = torch.broadcast_to(ret_mask, (batch_size, self.num_heads, mask.shape[-1])).unsqueeze(-2)
            # print(full_head_mask.shape, full_head_dist_mask.shape)
            full_head_dist_mask = full_head_dist_mask * full_head_mask.unsqueeze(-2)
            ret_mask = full_head_dist_mask.max(1)[0]
            # print(full_head_dist_mask[0])
            # print(full_head_dist_mask, mask)
            # # instead of replacing, we could multiply the existing mask
            # key = self.key_network(key)
            # queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
            # key, queries = key.reshape(batch_size, self.num_heads, self.model_dim, 1), queries.reshape(batch_size, self.num_heads, -1, self.model_dim)
            # # attention head computation
            # full_head_mask = self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim)) # Batch, heads, num queries
            # full_head_dist_mask = F.gumbel_softmax(full_head_mask, tau = self.gumbel_temperature, hard = hard).unsqueeze(-2)
            # full_head_dist_mask = full_head_dist_mask * torch.broadcast_to(mask, (batch_size, self.num_heads, mask.shape[-1])).unsqueeze(-2)
            # mask = full_head_mask.max(1)[0]

            # print("broadcast", full_head_dist_mask.shape)

        else:
            # print((torch.matmul(queries, key)[...,0])[0,0:2])
            full_head_dist_mask = full_head_mask.unsqueeze(-2) # Batch, heads, num queries
            # print("raw", full_head_dist_mask[0])
            full_head_dist_mask = F.gumbel_softmax(full_head_mask, tau = self.gumbel_temperature, hard = False).unsqueeze(-2)
            # print(pytorch_model.unwrap(queries[0]), pytorch_model.unwrap(key[0]))
            # print("softmax", (torch.matmul(queries, key)[...,0]  / np.sqrt(self.model_dim))[0,0])
            # print(self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim))[0,0])
            ret_mask = full_head_mask.max(1)[0]
            # print(full_head_dist_mask[0])
            # print("non-broadcast", full_head_dist_mask.shape, values.shape, queries.shape)

        # print(values[0], values.shape)
        # print("presoft", values[0], full_head_dist_mask.shape, values.shape)
        values = torch.matmul(full_head_dist_mask, values)
        # print("soft", full_head_dist_mask[0])
        # print("masks", (mask[0], mask.shape) if mask is not None else ret_mask[0], full_head_dist_mask[0][0][0])
        values = values.reshape(batch_size, -1)
        values = self.final_network(values)
        # print("final", values[0])
        return values, ret_mask

class MultiHeadInteractionAttentionBase(Network):
    def __init__(self, args):
        super().__init__(args)
        # just handles the final reasoning logic and multiple layers (on a single key, it reprocesses the key input for each layer)
        self.num_layers = args.mask_attn.num_layers
        self.repeat_layers = args.pair.repeat_layers
        if self.repeat_layers: 
            self.multi_head_attention = MultiHeadAttentionInteractionParallelLayer(args)
            layers = [self.multi_head_attention]
        else: 
            layers = [MultiHeadAttentionInteractionParallelLayer(args) for i in range(self.num_layers)]
            self.multi_head_attention = nn.ModuleList(layers)
        self.embed_dim = args.embed_inputs

        # final_args = copy.deepcopy(args)
        # final_args.include_last = True
        # final_args.num_inputs = final_args.embed_inputs
        # final_args.num_outputs = final_args.embed_inputs
        # final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
        # self.final_layer = MLPNetwork(final_args)

        self.model = layers#  + [self.final_layer]

    def forward(self, key, queries, mask=None, hard=False):
        batch_size = key.shape[0]
        cur_mask = None
        for i in range(self.num_layers):
            # start = time.time()
            if self.repeat_layers: key, out_mask = self.multi_head_attention(key, queries, mask, hard)
            else: key, out_mask = self.multi_head_attention[i](key, queries, mask, hard)
            # print(key.shape, out_mask.shape)
            if cur_mask is not None: cur_mask = torch.stack([cur_mask, out_mask], dim = -1).max(dim=-1)[0]
            else: cur_mask = out_mask
            # print("cur_mask", cur_mask.shape)
        # key = self.final_layer(key)
        return key, cur_mask

class RawAttentionNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # the object dim is the dimension of the value input
        self.single_object_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.reduce_fn = args.pair.reduce_function

        self.model_dim = args.mask_attn.model_dim
        self.embed_dim = args.embed_inputs
        if self.embed_dim <= 0:
            self.embed_dim = args.mask_attn.model_dim * args.mask_attn.num_heads
        args.embed_inputs = self.embed_dim

        self.return_mask = args.mask_attn.return_mask
        self.total_obj_dim = args.pair.total_obj_dim
        self.expand_dim = args.pair.expand_dim
        self.total_instances = args.pair.total_instances
        self.total_targets = args.pair.total_targets
        self.repeat_layers = args.pair.repeat_layers

        self.needs_encoding = args.mask_attn.needs_encoding # if mean variance network, will not need encoding layer
        if args.mask_attn.needs_encoding:
            args.include_last = True
            key_args = copy.deepcopy(args)
            key_args.object_dim = args.pair.single_obj_dim
            key_args.output_dim = self.embed_dim
            key_args.use_layer_norm = False
            key_args.pair.aggregate_final = False
            key_args.activation_final = "none"
            self.key_encoding = ConvNetwork(key_args)

            query_args = copy.deepcopy(args)
            query_args.object_dim = args.pair.object_dim
            query_args.output_dim = self.embed_dim
            query_args.use_layer_norm = False
            query_args.pair.aggregate_final = False
            query_args.activation_final = "none"
            self.query_encoding = ConvNetwork(query_args)
            layers = [self.key_encoding, self.query_encoding]
        else:
            layers = list()

        self.multi_head_attention = MultiHeadInteractionAttentionBase(args)
        
        layers =  layers + [self.multi_head_attention]
        if args.pair.aggregate_final:
            final_args = copy.deepcopy(args)
            final_args.include_last = True
            final_args.num_inputs = final_args.embed_inputs
            final_args.num_outputs = self.num_outputs
            final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
            self.final_layer = MLPNetwork(final_args)
            layers += [self.final_layer] 
        else:
            final_args = copy.deepcopy(args)
            final_args.output_dim = self.num_outputs
            final_args.object_dim = self.embed_dim
            final_args.use_layer_norm = False
            final_args.hidden_sizes = final_args.pair.final_layers
            self.final_layer = ConvNetwork(final_args)
            layers += [self.final_layer] 

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def set_instancing(self, total_instances, total_obj_dim, total_targets):
        assert self.expand_dim * total_instances == total_obj_dim
        self.total_obj_dim = total_obj_dim
        self.total_instances = total_instances
        self.total_targets = total_targets

    def slice_input(self, x):
        keys = torch.stack([x[...,i * self.single_object_dim: (i+1) * self.single_object_dim] for i in range(int(self.first_obj_dim // self.single_object_dim))], dim=-2) # [batch size, num keys, single object dim]
        queries = torch.stack([x[...,self.first_obj_dim + j * self.object_dim:self.first_obj_dim + (j+1) *self.object_dim] for j in range(int((x.shape[-1] - self.first_obj_dim) // self.object_dim))], dim=-2) # [batch size, num values, single object dim]
        # TODO: add relative value calculation
        keys, queries = keys.transpose(-2,-1), queries.transpose(-2,-1)
        return keys, queries

    def forward(self, x, m=None, hard=False):
        # x is an input of shape [batch, flattened dim of all target objects + flattened all query objects]
        # m is the batch, key, query mask
        # iterate over each instance
        # start = time.time()
        # print(x.shape)
        if self.needs_encoding:
            batch_size = x.shape[0]
            keys, queries = self.slice_input(x) # [batch, n_k, single_obj_dim], [batch, n_k, obj_dim]
            keys = self.key_encoding(keys) # [batch, d_k, n_k]
            queries = self.query_encoding(queries) # [batch, d_q, n_q]
            queries = queries.transpose(-2,-1)
        else:
            keys, queries = x # assumes the encodings are already done
            batch_size = keys.shape[0]
        # print(keys.shape, queries.shape)
        # slice = time.time()
        values, masks = list(), list() # the final output values
        for i in range(int(self.first_obj_dim // self.single_object_dim)):
            key = keys[...,i]
            value, mask = self.multi_head_attention(key, queries, m, hard=hard)
            values.append(value)
            masks.append(mask)
        m = torch.stack(masks, dim=2)
        # print("stacked", m.shape)
        x = torch.stack(values, dim=2) # [batch size, pairnet output dim, num_instances]
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.embed_dim)
            x = self.final_layer(x)
            m = m.max(-1)[0]
        else:
            x = self.final_layer(x)
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
            m = m.transpose(2,1)
            m = m.reshape(batch_size, -1)
        return x, m
