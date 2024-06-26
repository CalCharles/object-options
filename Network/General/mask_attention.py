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
from Network.General.attn_utils import evaluate_key_query, mask_query

class AttentionHead(Network):
    def __init__(self, args):
        super().__init__(args)
        self.softmax =  nn.Softmax(-1)
        self.model_dim = args.mask_attn.model_dim # the dimension of the keys and queries after network
        self.key_dim = args.embed_inputs # the dimension of the key inputs
        self.head_dim = args.mask_attn.head_dim # should be key_dim / num_heads, integer divisible
        self.query_dim = args.embed_inputs

        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.num_outputs = self.model_dim
        key_args.dropout = args.mask_attn.attention_dropout
        key_args.use_layer_norm = args.use_layer_norm
        self.key_network = MLPNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim
        query_args.object_dim = self.query_dim
        query_args.dropout = args.mask_attn.attention_dropout
        query_args.use_layer_norm = args.use_layer_norm
        aggregate_final = False
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim
        value_args.pair.object_dim = self.query_dim
        value_args.pair.first_obj_dim = self.key_dim
        value_args.dropout = args.mask_attn.attention_dropout
        value_args.use_layer_norm = args.use_layer_norm
        value_args.pair.aggregate_final = False
        self.value_network = PairNetwork(value_args)

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim
        final_args.num_outputs = self.head_dim
        final_args.dropout = args.mask_attn.attention_dropout
        final_args.use_layer_norm = args.use_layer_norm
        final_args.hidden_sizes = final_args.pair.final_layers
        self.final_network = MLPNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def mask_softmax(self, softmax, mask):
        return (softmax * mask.unsqueeze(-1)).transpose(2,1)

    def forward(self, key, queries, mask, valid):
        start = time.time()
        batch_size = key.shape[0]
        value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1).reshape(batch_size, -1) # todo: relative state operations here
        key = self.key_network(key)
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
        queries = self.mask_softmax(queries, mask)
        values = self.value_network(value_inputs).reshape(batch_size, -1, self.model_dim)
        weights = evaluate_key_query(self.softmax, key, queries, mask, valid, single_key=False)
        values = torch.matmul(weights, values)
        values = values.reshape(batch_size, -1)
        values = self.final_network(values)
        # print("head compute", time.time() - start)
        return values        

class MultiHeadAttentionLayer(Network):
    def __init__(self, args):
        super().__init__(args)
        self.num_heads = args.mask_attn.num_heads
        # assert args.embed_inputs % self.num_heads == 0, f"head and key not divisible, key: {args.key_dim}, head: {self.num_heads}"
        args.mask_attn.head_dim = int(args.embed_inputs // self.num_heads)
        layers = [AttentionHead(args) for i in range(self.num_heads)] 
        self.attention_heads = nn.ModuleList(layers)

        self.model = layers

    def forward(self, key, queries, mask):
        key_parts = list()
        for i in range(self.num_heads): # TODO: head logic in parallel
            key_parts.append(self.attention_heads[i](key, queries, mask))
        return torch.cat(key_parts, dim=-1)

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
        # assert args.embed_inputs % self.num_heads == 0 or (not concatenated_values), f"head and key not divisible, key: {args.embed_inputs}, head: {self.num_heads}"
        self.head_dim = int(args.embed_inputs // self.num_heads) # should be key_dim / num_heads, integer divisible
        self.append_keys = args.mask_attn.append_keys
        self.no_hidden = args.mask_attn.no_hidden
        self.mask_mode = args.mask_attn.mask_mode

        # process one key at a time
        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.num_outputs = self.model_dim * self.num_heads
        key_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in key_args.hidden_sizes]
        # key_args.dropout = args.mask_attn.attention_dropout
        # key_args.use_layer_norm = True
        key_args.activation_final = "none"
        self.key_network = MLPNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim * self.num_heads
        query_args.object_dim = self.query_dim
        query_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in query_args.hidden_sizes]
        # query_args.dropout = args.mask_attn.attention_dropout
        # query_args.use_layer_norm = True
        query_args.aggregate_final = False
        query_args.activation_final = "none"
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim * self.num_heads
        value_args.pair.object_dim = self.query_dim
        value_args.pair.first_obj_dim = self.key_dim
        value_args.hidden_sizes = list() if self.no_hidden else [hs*self.num_heads for hs in value_args.hidden_sizes]
        # value_args.dropout = args.mask_attn.attention_dropout
        # value_args.use_layer_norm = True
        value_args.pair.aggregate_final = False
        value_args.activation_final = value_args.activation
        value_args.output_dim = self.model_dim * self.num_heads
        value_args.num_inputs = self.query_dim
        value_args.object_dim = self.query_dim
        if self.append_keys: self.value_network = PairNetwork(value_args)
        else: self.value_network = ConvNetwork(value_args)

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim * self.num_heads if concatenated_values else self.model_dim
        final_args.num_outputs = self.key_dim
        final_args.dropout = args.mask_attn.attention_dropout
        # final_args.use_layer_norm = True
        final_args.hidden_sizes = list() if self.no_hidden else [(hs*self.num_heads if concatenated_values else hs) for hs in final_args.pair.final_layers] # these should be wide enough to support model_dim * self.num_heads
        final_args.activation_final = final_args.activation
        self.final_network = MLPNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def mask_softmax(self, softmax, mask):
        # print(softmax.shape, mask.shape)
        # print(softmax, mask, (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1))
        # print(softmax[0], (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1)[0])
        return (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1)

    def forward(self, key, queries, mask, query_final=False, valid=None):
        # applies the mask at the queries and the values
        # alteratively, apply the mask after the softmax
        start = time.time()
        batch_size = key.shape[0]
        queries = mask_query(queries, mask, valid, single_key=True) if self.mask_mode == "query" else queries
        # print(key[0], queries[0], mask[0])
        # error
        if self.append_keys: value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1).reshape(batch_size, -1) # todo: relative state operations here
        else: value_inputs = queries.transpose(-2,-1) # todo: relative state operations here
        # uncomment below if queries = queries * mask.unsqueeze(-1) is commented, or we want safer operations that mask out the whole value
        # value_inputs = (torch.cat([key.unsqueeze(1)] + [queries], dim=1) * mask.unsqueeze(-1)).reshape(batch_size, -1) # todo: relative state operations here
        key = self.key_network(key)
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
        # print(value_inputs.shape, self.value_network)
        # print("mask", mask[0])
        # print("values", value_inputs[0])
        values = self.value_network(value_inputs)
        if not self.append_keys: values = values.transpose(1,2)
        values = values.reshape(batch_size, queries.shape[1], self.num_heads, self.model_dim).transpose(1,2)
        # print("value out", values[0], values.shape)
        # values = self.value_network(value_inputs).reshape(batch_size, self.num_heads, -1, self.model_dim) # -1 dim is num_queries
        # print("queries", queries[0])
        key, queries = key.reshape(batch_size, self.num_heads, self.model_dim, 1), queries.reshape(batch_size, -1, self.num_heads, self.model_dim).transpose(1,2)
        # print("key", key[0], key.shape)
        # print("queries", queries[0])
        # error
        # print(key.shape, queries.shape, values.shape)
        # print(key[0], queries[0], values[0], mask[0], self.model_dim)
        # print("before", values, torch.matmul(queries, key).shape, mask, 
        #     torch.matmul(queries, key)[...,0]  / np.sqrt(self.key_dim),
        #     self.mask_softmax(torch.matmul(queries, key)[...,0]  / np.sqrt(self.key_dim), mask),
        #     self.softmax(self.mask_softmax(torch.matmul(queries, key)[...,0]  / np.sqrt(self.key_dim), mask)))
        # values = torch.matmul(self.softmax(self.mask_softmax(F.relu(torch.matmul(queries, key)[...,0]) / np.sqrt(self.model_dim), mask)).unsqueeze(-2), values)
        
        # first matmul: batch x n_queries x d_model * batch x d_model x 1 = batch x n_queries
        # softmax over queries batch x softmaxed queries
        # matmul batch x softened queries x queries x values = batch x values
        # softmax ((Q \cdot M)K^T) / \sqrt(model dim)V^T 
        # print(torch.matmul(queries, key)[...,0].shape,
        #       torch.matmul(queries, key).shape,
        #       queries.shape, key.shape,
        #     #   queries[0], key[0],
        #         torch.matmul(queries, key)[...,0][0],
        #     self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim))[0],
        #       self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim)).unsqueeze(-2).shape,
        #        values[0].shape)
        # print(pytorch_model.unwrap(queries[0]), pytorch_model.unwrap(key[0]))
        # print((torch.matmul(queries, key)[...,0])[0,0])
        # print(self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim))[0,0])
        if query_final:
            # batch x heads x queries x 1 * batch x heads x queries x model_dim = batch x heads x queries x model_dim
            if self.mask_mode == "attn": weights = evaluate_key_query(self.softmax, key, queries, mask, valid, single_key=True)
            else: weights = evaluate_key_query(self.softmax, key, queries, None, None, single_key=True)
            values = (weights.unsqueeze(-1) * values )
            if self.merge_function == 'cat': values = values.transpose(3,2) #
            # if not cat: batch x keys x queries x model_dim (merges the heads)
            # if cat: batch x heads x model dim x queries -> batch x heads * model dim x queries (flip queries back)
            values = reduce_function(self.merge_function, values, dim=1) 
            if self.merge_function == 'cat': values = values.transpose(1,2)
            # values = reduce_function(self.merge_function, values.transpose(1,2), dim=2).transpose(1,2) # batch x keys x queries x model_dim merges with the same function as all the others
            # print(values.shape)
            # values of shape batch x queries x final dimension
            # weights of shape batch x queries
        else:
            # print(queries.shape, key.shape, values.shape)
            if self.mask_mode == "attn": weights = evaluate_key_query(self.softmax, key, queries, mask, valid, single_key=True)
            else: weights = evaluate_key_query(self.softmax, key, queries, None, None, single_key=True)
            values = torch.matmul(weights.unsqueeze(-2), values)[:,:,0,:] # batch x heads x 1 x queries * batch x heads x queries x model_dim = batch x heads x 1 x model_dim
            # uncomment below if commented both queries = queries * mask.unsqueeze(-1), and value_inputs = torch.cat([key.unsqueeze(1)] + [queries * mask.unsqueeze(-1)], dim=1).reshape(batch_size, -1)
            # values = torch.matmul(self.mask_softmax(self.softmax(torch.matmul(queries, key)[...,0] / np.sqrt(self.model_dim)).unsqueeze(-2), mask), values)
            # print("after", values)
            values = reduce_function(self.merge_function, values, dim=1)
            values = self.final_network(values)
            # values of shape batch x final dimension
            # weights of shape batch x queries
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

    def forward(self, key, queries, m, valid=None):
        batch_size = key.shape[0]
        weights = list()
        for i in range(self.num_layers):
            start = time.time()
            if self.repeat_layers: key, weight = self.multi_head_attention(key, queries, m, query_final=i==self.num_layers-1 and (not self.query_aggregate), valid=valid)
            else: key, weight = self.multi_head_attention[i](key, queries, m, query_final=i==self.num_layers-1 and (not self.query_aggregate), valid=valid)
            weights.append(weight)
            # print("layer compute", time.time() - start)
        # key = self.final_layer(key)
        return key, torch.stack(weights, dim = 1) # key of shape batch x final dim, weights of shape batch x num_layers x num_heads

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
        concatenate_values = args.mask_attn.merge_function == "cat"
        if self.embed_dim <= 0:
            if concatenate_values:
                self.embed_dim = self.model_dim * args.mask_attn.num_heads
            else:
                self.embed_dim = self.model_dim
        args.embed_inputs = self.embed_dim

        self.final_dim = self.embed_dim * args.mask_attn.num_heads if concatenate_values else self.model_dim
        
        
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

    def forward(self, x, m=None, valid=None, return_weights = False):
        # x is an input of shape [batch, flattened dim of all target objects + flattened all query objects]
        # m is the batch, key, query mask
        # iterate over each instance
        # start = time.time()
        if self.needs_encoding:
            batch_size = x.shape[0]
            keys, queries = self.slice_input(x) # [batch, n_k, single_obj_dim], [batch, n_q, obj_dim]
            # keys = keys * 0.0
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
        weights = list()
        for i in range(int(self.first_obj_dim // self.single_object_dim)):
            key = keys[...,i]
            # print(keys.shape, queries.shape, m[:,i,:].shape, m.shape)
            value, weight = self.multi_head_attention(key, queries, m[:,i,:], valid=valid)
            values.append(value)
            weights.append(weight)
        # print("mha", time.time() - mha)
        # print(value[0].shape)
        # print(x.shape, self.aggregate_final)
        x = torch.stack(values, dim=2) # [batch size, pairnet output dim, num_instances]
        if not self.query_aggregate:
            x = x.view(batch_size, -1, self.final_dim).transpose(1,2)
            x = self.final_layer(x).transpose(1,2)
            # print("after final", x.shape, self.final_dim)
            if return_weights:
                return x.view(batch_size, -1), torch.stack(weights, dim = 3) # batch x num_layers x num_heads x keys x queries
            return x.view(batch_size, -1)
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.embed_dim)
            # print(x.shape)
            x = self.final_layer(x)
        else:
            x = self.final_layer(x)
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
            m = m.transpose(2,1)
            m = m.reshape(batch_size, -1)
        # print("total compute", time.time() - start)
        if return_weights:
            return x, torch.stack(weights, dim = 3) # weights shape: batch x num_layers x num_heads x keys x queries
        return x