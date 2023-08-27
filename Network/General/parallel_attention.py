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
from Network.General.attn_utils import evaluate_key_query

class MultiHeadAttentionParallelLayer(Network):
    def __init__(self, args):
        super().__init__(args)
        self.softmax =  nn.Softmax(-1)
        self.model_dim = args.mask_attn.model_dim # the dimension of the keys and queries after network
        # assert(args.embed_inputs == args.mask_attn.model_dim * args.mask_attn.num_heads or args.mask_attn.merge_function != "cat")
        self.key_dim = args.embed_inputs # the dimension of the key inputs, must equal model_dim * num_heads
        self.query_dim = args.embed_inputs
        self.num_heads = args.mask_attn.num_heads
        self.merge_function = args.mask_attn.merge_function
        concatenated_values = self.merge_function == "cat"
        # assert args.embed_inputs % self.num_heads == 0 or (not concatenated_values), "head and key not divisible, key: {args.embed_inputs}, head: {self.num_heads}"
        self.head_dim = int(args.embed_inputs // self.num_heads) # should be key_dim / num_heads, integer divisible
        self.append_keys = args.mask_attn.append_keys
        self.no_hidden = args.mask_attn.no_hidden
        self.gumbel = args.mask_attn.gumbel_attention

        # process all keys at once
        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.object_dim = self.query_dim
        key_args.num_outputs = self.model_dim * self.num_heads
        key_args.output_dim = self.model_dim * self.num_heads
        key_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in key_args.hidden_sizes]
        key_args.aggregate_final = False
        key_args.activation_final = "none"
        self.key_network = ConvNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim * self.num_heads
        query_args.object_dim = self.query_dim
        query_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in query_args.hidden_sizes]
        query_args.aggregate_final = False
        query_args.activation_final = "none"
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim * self.num_heads
        value_args.object_dim = self.query_dim + int(self.append_keys) * self.key_dim
        value_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in value_args.hidden_sizes]
        value_args.activation_final = value_args.activation
        value_args.output_dim = self.model_dim * self.num_heads
        value_args.num_inputs = self.query_dim
        self.value_network = ConvNetwork(value_args) # values append keys internally

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim * self.num_heads if concatenated_values else self.model_dim
        final_args.object_dim = self.model_dim * self.num_heads if concatenated_values else self.model_dim
        final_args.num_outputs = self.key_dim
        final_args.output_dim = self.key_dim
        final_args.dropout = args.mask_attn.attention_dropout
        final_args.use_layer_norm = True
        final_args.hidden_sizes = [(hs*self.num_heads if concatenated_values else hs) for hs in final_args.pair.final_layers] # these should be wide enough to support model_dim * self.num_heads
        final_args.activation_final = final_args.activation
        self.final_network = ConvNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]
    
    def append_values(self, keys, queries):
        # appends the keys to the queries, expanding the values to batch x keys x queries x (key_dim + query_dim)
        if self.append_keys:
            values = list()
            for i in range(keys.shape[1]):
                key = torch.broadcast_to(keys[:,i].unsqueeze(1),  (keys.shape[0], queries.shape[1], keys.shape[-1]))
                values.append(torch.cat([key, queries], dim=-1))
            return torch.stack(values, dim = 1)
        else: return torch.stack([queries.clone() for _ in range(keys.shape[1])], dim=1) 

    def forward(self, keys, queries, mask, query_final=False, valid=None):
        # keys of shape: batch x num_keys x key_dim
        # queries of shape: batch x num_queries x query_dim
        # mask and valid shape: batch x num_keys x num_queries x 1
        # query final returns batch x num_keys x num_queries x model dim, otherwise batch x num_keys x model_dim

        start = time.time()
        batch_size = keys.shape[0]
        num_keys = keys.shape[1]
        num_queries = queries.shape[1]
        value_inputs = self.append_values(keys, queries)
        # print(keys.shape, queries.shape, value_inputs.shape)
        keys = self.key_network(keys.transpose(-2,-1)).transpose(-2,-1) # batch x num_keys x key_dim * num_heads
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1) # batch x num_queries x query_dim * num_heads
        keys, queries = keys.reshape(batch_size, -1, self.num_heads, self.model_dim).transpose(1,2).transpose(2,3), queries.reshape(batch_size, -1, self.num_heads, self.model_dim).transpose(1,2)

        # TODO: confirm that the reshapes below actually have the right format
        values = self.value_network(value_inputs.reshape(batch_size, num_keys * num_queries, -1).transpose(-1,-2)).transpose(-1,-2) # batch x num_keys * num_queries x num_heads * model_dim
        values = values.reshape(batch_size, num_keys * num_queries, self.num_heads, self.model_dim).transpose(1,2) # batch x num_heads x num_keys * num_queries x model_dim
        values = values.transpose(-1,-2).reshape(batch_size, self.num_heads, self.model_dim, num_keys, num_queries).transpose(2,3).transpose(3,4) # batch x num_heads x num_keys x num_queries x model_dim
        # print(values.shape, keys.shape, queries.shape, mask.shape, valid)

        if query_final: # uses a sigmoid for weights
            weights = evaluate_key_query(torch.sigmoid, keys, queries, mask, valid, single_key=False) # batch x heads x keys x queries
            # batch x heads x keys x queries x 1 * batch x heads x keys x queries x model_dim = batch x heads x keys x queries x model_dim
            values = (weights.unsqueeze(-1) * values )
            values = reduce_function(self.merge_function, values, dim=1) # batch x keys x queries x model_dim (merges the heads)
        else: # sum along the query dimension (already normalized by weights) and apply the final network
            weights = evaluate_key_query(self.softmax, keys, queries, mask, valid, single_key=False) # batch x heads x keys x queries
            # batch x heads x keys x queries x 1 * batch x heads x keys x queries x model_dim = batch x heads x keys x queries x model_dim
            values = (weights.unsqueeze(-1) * values )
            # print("after", values)
            values = values.sum(dim=-2) # batch x heads x keys x model_dim
            # cat requires the heads and value dimension to get concatenated
            if self.merge_function == "cat": values = values = reduce_function(self.merge_function, values.transpose(1,2), dim=2)
            else: values = reduce_function(self.merge_function, values, dim=1)
            values = self.final_network(values.transpose(-2,-1)).transpose(-2,-1) # batch x keys x model_dim
        return values, weights

class MultiHeadAttentionBase(Network):
    def __init__(self, args):
        super().__init__(args)
        # just handles the final reasoning logic and multiple layers (on a single key, it reprocesses the key input for each layer)
        self.num_layers = args.mask_attn.num_layers
        self.repeat_layers = args.pair.repeat_layers
        if self.repeat_layers: 
            self.multi_head_attention = MultiHeadAttentionParallelLayer(args)
            layers = [self.multi_head_attention]
        else: 
            layers = [MultiHeadAttentionParallelLayer(args) for i in range(self.num_layers)]
            self.multi_head_attention = nn.ModuleList(layers)
        self.embed_dim = args.embed_inputs * args.mask_attn.num_heads
        self.query_aggregate = args.embedpair.query_aggregate

        # final_args = copy.deepcopy(args)
        # final_args.include_last = True
        # final_args.num_inputs = final_args.embed_inputs * args.mask_attn.num_heads
        # final_args.num_outputs = final_args.embed_inputs * args.mask_attn.num_heads
        # final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
        # self.final_layer = MLPNetwork(final_args)

        self.model = layers # + [self.final_layer]

    def forward(self, keys, queries, m, valid=ModuleNotFoundError):
        weights = list()
        for i in range(self.num_layers):
            start = time.time()
            if self.repeat_layers: keys, weight = self.multi_head_attention(keys, queries, m, query_final=i==self.num_layers-1 and (not self.query_aggregate), valid=valid)
            else: keys, weight = self.multi_head_attention[i](keys, queries, m, query_final=i==self.num_layers-1 and (not self.query_aggregate), valid=valid)
            # print("layer compute", time.time() - start)
            weights.append(weight)
        # keys = self.final_layer(keys)
        return keys, weights

class ParallelMaskedAttentionNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # the object dim is the dimension of the value input
        self.single_object_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.reduce_fn = args.pair.reduce_function
        self.query_aggregate = args.embedpair.query_aggregate

        self.model_dim = args.mask_attn.model_dim
        self.embed_dim = args.embed_inputs
        concatenate_values = args.mask_attn.merge_function == "cat"
        if self.embed_dim <= 0:
            if concatenate_values:
                self.embed_dim = self.model_dim * args.mask_attn.num_heads
            else:
                self.embed_dim = self.model_dim
        args.embed_inputs = self.embed_dim

        self.final_dim = self.embed_dim if concatenate_values else self.model_dim
        
        
        self.return_mask = args.mask_attn.return_mask
        self.total_obj_dim = args.pair.total_obj_dim
        self.expand_dim = args.pair.expand_dim
        self.total_instances = args.pair.total_instances
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

        self.multi_head_attention = MultiHeadAttentionBase(args)
        
        layers =  layers + [self.multi_head_attention]
        if args.pair.aggregate_final:
            final_args = copy.deepcopy(args)
            final_args.include_last = True
            final_args.num_inputs = self.final_dim
            final_args.num_outputs = self.num_outputs
            final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
            self.final_layer = MLPNetwork(final_args)
            layers += [self.final_layer] 
        else:
            final_args = copy.deepcopy(args)
            final_args.output_dim = self.num_outputs
            final_args.object_dim = self.embed_dim if self.query_aggregate else self.final_dim
            final_args.use_layer_norm = False
            final_args.hidden_sizes = final_args.pair.final_layers
            self.final_layer = ConvNetwork(final_args)

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim # this is the only one that matters, single_object_dim and object_dim should not change
        self.total_instances = num_objects

    def slice_input(self, x):
        keys = torch.stack([x[...,i * self.single_object_dim: (i+1) * self.single_object_dim] for i in range(int(self.first_obj_dim // self.single_object_dim))], dim=-2) # [batch size, num keys, single object dim]
        queries = torch.stack([x[...,self.first_obj_dim + j * self.object_dim:self.first_obj_dim + (j+1) *self.object_dim] for j in range(int((x.shape[-1] - self.first_obj_dim) // self.object_dim))], dim=-2) # [batch size, num values, single object dim]
        # TODO: add relative value calculation
        keys, queries = keys.transpose(-2,-1), queries.transpose(-2,-1)
        return keys, queries

    def slice_masks(self, m, batch_size, num_keys):
        # assumes m of shape [batch, num_keys * num_queries]
        # print(m.shape,torch.ones(batch_size, 1).shape, cuda_string(self.gpu if self.iscuda else -1), pytorch_model.wrap(torch.ones(batch_size, 1), cuda=self.iscuda).device)
        # if len(m.shape) == 1: m = m * torch.ones((batch_size, 1), device = cuda_string(self.gpu if self.iscuda else -1))
        # start= time.time()
        if len(m.shape) == 1: 
            m = m * pytorch_model.wrap(torch.ones(batch_size, 1), cuda=self.iscuda)
        m = m.reshape(batch_size, num_keys, -1)
        # print("slice opt", time.time() - start)
        return m

    def forward(self, x, m=None, valid=None, return_weights=False):
        # x is an input of shape [batch, flattened dim of all target objects + flattened all query objects]
        # m is the batch, key, query mask
        # iterate over each instance
        # start = time.time()
        if self.needs_encoding:
            batch_size = x.shape[0]
            keys, queries = self.slice_input(x) # [batch, n_k, single_obj_dim], [batch, n_k, obj_dim]
            # print("kq", keys, queries)
            keys = self.key_encoding(keys).transpose(-2,-1) # [batch, n_k, d]
            queries = self.query_encoding(queries).transpose(-2,-1) # [batch, n_q, d]
            # print("kq encoded", keys, queries)
        else:
            keys, queries, m = x # assumes the encodings are already done
            batch_size = keys.shape[0]
        # slice = time.time()
        if m is None:
            # mgen = time.time()
            m = pytorch_model.wrap(torch.ones((batch_size, keys.shape[1], queries.shape[1])), cuda=self.iscuda) # create ones of [batch, n_k, n_q]
        else:
            m = self.slice_masks(m, batch_size, keys.shape[1])
        # print(self.needs_encoding, m.shape)
        x, weights = self.multi_head_attention(keys, queries, m, valid=valid) # batch, keys, (queries), final_dim
        if not self.query_aggregate:
            x = x.reshape(batch_size, -1, self.final_dim).transpose(1,2)
            x = self.final_layer(x).transpose(1,2)
            return x.view(batch_size, -1)
        if self.aggregate_final: # reduce all the keys
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.embed_dim)
            x = self.final_layer(x)
        else:
            x = self.final_layer(x.transpose(2,1))
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
            m = m.transpose(2,1)
            m = m.reshape(batch_size, -1)
        # print("total compute", time.time() - start)
        if return_weights: return x,weights
        return x