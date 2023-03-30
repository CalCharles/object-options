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

class MultiHeadAttentionParallelLayer(Network):
    def __init__(self, args):
        super().__init__(args)
        self.softmax =  nn.Softmax(-1)
        self.model_dim = args.mask_attn.model_dim # the dimension of the keys and queries after network
        assert(args.embed_inputs == args.mask_attn.model_dim * args.mask_attn.num_heads or args.mask_attn.merge_function != "cat")
        self.key_dim = args.embed_inputs # the dimension of the key inputs, must equal model_dim * num_heads
        self.query_dim = args.embed_inputs
        self.num_heads = args.mask_attn.num_heads
        assert args.embed_inputs % self.num_heads == 0, f"head and key not divisible, key: {args.key_dim}, head: {self.num_heads}"
        self.head_dim = int(args.embed_inputs // self.num_heads) # should be key_dim / num_heads, integer divisible
        self.merge_function = args.mask_attn.merge_function

        # process all keys at a time
        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.output_dim = self.model_dim * self.num_heads
        key_args.object_dim = self.key_dim
        key_args.hidden_sizes = [hs*self.num_heads for hs in key_args.hidden_sizes]
        # key_args.dropout = args.mask_attn.attention_dropout
        # key_args.use_layer_norm = True
        key_args.aggregate_final = False
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
        value_args.activation_final = "none"
        value_args.output_dim = self.model_dim * self.num_heads
        value_args.num_inputs = self.query_dim
        self.value_network = PairNetwork(value_args) # the pairnet is used as a keypair network
        # self.value_network = ConvNetwork(value_args)

        # returns k keys by applying a convolution
        final_args = copy.deepcopy(args)
        concatenated_values = self.merge_function == "cat"
        final_args.object_dim = self.model_dim * self.num_heads if concatenated_values else self.model_dim
        final_args.output_dim = self.key_dim
        final_args.dropout = args.mask_attn.attention_dropout
        # final_args.use_layer_norm = True
        final_args.hidden_sizes = [(hs*self.num_heads if concatenated_values else hs) for hs in final_args.pair.final_layers] # these should be wide enough to support model_dim * self.num_heads
        final_args.activation_final = final_args.activation
        final_args.pair.aggregate_final = False
        self.final_network = ConvNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def mask_softmax(self, softmax, mask):
        # print(softmax.shape, mask.shape)
        # print(softmax, mask, (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1))
        # print(softmax[0], (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1)[0])
        return (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1)

    def forward(self, input_keys, input_queries, mask, query_final=False):
        # applies the mask at the queries and the values
        # alteratively, apply the mask after the softmax
        # queries: batch x num_queries x model dim
        # input_keys: batch x num_keys x model dim
        start = time.time()
        batch_size = input_keys.shape[0]
        input_queries = queries * mask.unsqueeze(-1)
        # value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1).reshape(batch_size, -1) # todo: relative state operations here
        # value_inputs = queries # todo: relative state operations here
        # uncomment below if queries = queries * mask.unsqueeze(-1) is commented, or we want safer operations that mask out the whole value
        # value_inputs = (torch.cat([key.unsqueeze(1)] + [queries], dim=1) * mask.unsqueeze(-1)).reshape(batch_size, -1) # todo: relative state operations here
        keys = self.key_network(input_keys.transpose(-2,-1)).transpose(-2,-1) # batch x num_keys x model dim * num_heads
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1) # batch x num_keys x model dim * num_heads
        keys, queries = keys.view(batch_size, -1, self.num_heads, self.model_dim).transpose(1,2), queries.view(batch_size, -1, self.num_heads, self.model_dim).transpose(1,2).transpose(-1,-2)
        # print(value_inputs.shape, self.value_network)
        values = list()
        for i in range(input_keys.shape[1]): # batch x num keys x model dim
            values.append(self.value_network(torch.cat([input_keys[:,i], input_queries.reshape(batch_size, -1)])))
        values = torch.stack(values, dim=1).tranpose(2,3).transpose(1,2) # batch x num_keys x num_queries x model_dim * num_heads
        values = values.reshape(batch_size, self.num_heads, self.model_dim, keys.shape[1], queries.shape[1]).transpose(2,3).tranpose(3,4)
        # softmax = batch x heads x keys x model dim * batch x heads x model_dim x queries = batch x heads x keys x queries
        if query_final:
            # batch x heads x queries x keys x 1 * batch x heads x queries x keys x model_dim
            values = (self.softmax(torch.matmul(key, queries) / np.sqrt(self.model_dim)).unsqueeze(-1) * values )
            values = reduce_function("max", values.transpose(1,2).transpose(2,3), dim=3).transpose(1,2) # keys x queries x model_dim only merges heads using max
        else:
            # values = batch x heads x keys x 1 x queries * batch x heads x keys x queries x model_dim = batch x heads x keys x model_dim
            values = torch.matmul(self.softmax(torch.matmul(key, queries) / np.sqrt(self.model_dim)).unsqueeze(-2), values)
            # swap dimensions: batch x heads x keys x model_dim -> batch x model_dim * num_heads x keys
            values = reduce_function(self.merge_function, values.transpose(-1,-2), dim=1).transpose(-1,-2)
            values = self.final_network(values).tranpose(-1,-2) # batch x model_dim x keys
        return values

class MultiHeadAttentionBase(Network):
    def __init__(self, args):
        super().__init__(args)
        # just handles the final reasoning logic and multiple layers (on a single key, it reprocesses the key input for each layer)
        self.num_layers = args.mask_attn.num_layers
        self.repeat_layers = args.pair.repeat_layers
        self.query_aggregate = args.embedpair.query_aggregate
        if self.repeat_layers: 
            self.multi_head_attention = MultiHeadAttentionParallelLayer(args)
            layers = [self.multi_head_attention]
        else: 
            layers = [MultiHeadAttentionParallelLayer(args) for i in range(self.num_layers)]
            self.multi_head_attention = nn.ModuleList(layers)
        self.embed_dim = args.embed_inputs * args.mask_attn.num_heads

        # final_args = copy.deepcopy(args)
        # final_args.include_last = True
        # final_args.num_inputs = final_argsargs.mask_attn.merge_function == "cat".embed_inputs * args.mask_attn.num_heads
        # final_args.num_outputs = final_args.embed_inputs * args.mask_attn.num_heads
        # final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
        # self.final_layer = MLPNetwork(final_args)

        self.model = layers # + [self.final_layer]

    def forward(self, keys, queries, m):
        batch_size = keys.shape[0]
        for i in range(self.num_layers):
            start = time.time()
            if self.repeat_layers: keys = self.multi_head_attention(keys, queries, m, query_final=i==self.num_layers-1 and self.query_aggregate)
            else: keys = self.multi_head_attention[i](keys, queries, m, query_final=i==self.num_layers-1 and self.query_aggregate)
            # print("layer compute", time.time() - start)
        # keys = self.final_layer(keys)
        return keys

class MaskedAttentionNetwork(Network):
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
        if self.embed_dim <= 0:
            concatenated_values = args.mask_attn.merge_function == "cat" # if we use a difference reduction function, then the embed dim is independent of the number of heads
            if concatenated_values:
                self.embed_dim = self.model_dim * args.mask_attn.num_heads
            else:
                self.embed_dim = self.model_dim
        args.embed_inputs = self.embed_dim
        
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

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim # this is the only one that matters
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

    def forward(self, x, m=None):
        # x is an input of shape [batch, flattened dim of all target objects + flattened all query objects]
        # m is the batch, key, query mask
        # iterate over each instance
        # start = time.time()
        if self.needs_encoding:
            batch_size = x.shape[0]
            keys, queries = self.slice_input(x) # [batch, n_k, single_obj_dim], [batch, n_k, obj_dim]
            # print(keys, queries, self.gpu, self.key_encoding.model[0].weight.device, self.iscuda)
            keys = self.key_encoding(keys) # [batch, d_k, n_k]
            queries = self.query_encoding(queries) # [batch, d_q, n_q]
            queries = queries.transpose(-2,-1)
        else:
            keys, queries, m = x # assumes the encodings are already done
            batch_size = keys.shape[0]
        # print(m.shape, keys.shape, queries.shape, self.needs_encoding)
        # print(keys.shape)
        # slice = time.time()
        if m is None:
            # mgen = time.time()
            m = pytorch_model.wrap(torch.ones((batch_size, queries.shape[1], keys.shape[-1])), cuda=self.iscuda) # create ones of [batch, n_q]
            # print("mgen", time.time() - mgen)
        else: 
            # print("preslice", m.shape)
            # print("mask slice", batch_size, m, keys.shape, queries.shape)
            m = self.slice_masks(m, batch_size, keys.shape[-1])
        # print("slice", time.time() - slice)
        # mha = time.time()
        # print(m)
        # print(pytorch_model.wrap(torch.ones((batch_size, queries.shape[1], keys.shape[-1])), cuda=self.iscuda).shape)
        values = list() # the final output values
        values = self.multi_head_attention(keys, queries, m[:,i,:]) # batch x num_keys (x num_queries) x model_dim
        if self.query_aggregate:
            x = x.view(batch_size, -1, self.model_dim).tranpose(1,2)
            x = self.final_layer(x).tranpose(1,2)
        elif self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.embed_dim)
            # print(x.shape)
            x = self.final_layer(x)
        else:
            x = self.final_layer(x.transpose(1,2))
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
            m = m.transpose(2,1)
            m = m.reshape(batch_size, -1)
        # print("total compute", time.time() - start)
        return x
