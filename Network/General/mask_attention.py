from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy, time
from Network.network_utils import reduce_function, get_acti, pytorch_model
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork

class AttentionHead(Network):
    def __init__(self, args):
        super().__init__(args)
        self.softmax =  nn.Softmax(-1)
        self.model_dim = args.mask_attn.model_dim # the dimension of the keys and queries after network
        self.key_dim = args.mask_attn.embed_dim # the dimension of the key inputs
        self.head_dim = args.mask_attn.head_dim # should be key_dim / num_heads, integer divisible
        self.query_dim = args.mask_attn.embed_dim

        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.num_outputs = self.model_dim
        key_args.dropout = args.mask_attn.attention_dropout
        key_args.use_layer_norm = True
        self.key_network = MLPNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim
        query_args.object_dim = self.query_dim
        query_args.dropout = args.mask_attn.attention_dropout
        query_args.use_layer_norm = True
        aggregate_final = False
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim
        value_args.pair.object_dim = self.query_dim
        value_args.pair.first_obj_dim = self.key_dim
        value_args.dropout = args.mask_attn.attention_dropout
        value_args.use_layer_norm = True
        value_args.pair.aggregate_final = False
        self.value_network = PairNetwork(value_args)

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim
        final_args.num_outputs = self.head_dim
        final_args.dropout = args.mask_attn.attention_dropout
        final_args.use_layer_norm = True
        final_args.hidden_sizes = final_args.pair.final_layers
        self.final_network = MLPNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def mask_softmax(self, softmax, mask):
        return (softmax * mask.unsqueeze(-1)).transpose(2,1)

    def forward(self, key, queries, mask):
        start = time.time()
        batch_size = key.shape[0]
        value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1).reshape(batch_size, -1) # todo: relative state operations here
        key = self.key_network(key)
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
        queries = self.mask_softmax(queries, mask)
        values = self.value_network(value_inputs).reshape(batch_size, -1, self.model_dim)
        values = torch.matmul(self.mask_softmax(self.softmax(torch.matmul(key.unsqueeze(1), queries)), mask) / np.sqrt(self.key_dim), values)
        values = values.reshape(batch_size, -1)
        values = self.final_network(values)
        print("head compute", time.time() - start)
        return values        

class MultiHeadAttentionLayer(Network):
    def __init__(self, args):
        super().__init__(args)
        self.num_heads = args.mask_attn.num_heads
        assert args.mask_attn.embed_dim % self.num_heads == 0, f"head and key not divisible, key: {args.key_dim}, head: {self.num_heads}"
        args.mask_attn.head_dim = int(args.mask_attn.embed_dim // self.num_heads)
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
        self.key_dim = args.mask_attn.embed_dim # the dimension of the key inputs
        self.query_dim = args.mask_attn.embed_dim
        self.num_heads = args.mask_attn.num_heads
        assert args.mask_attn.embed_dim % self.num_heads == 0, f"head and key not divisible, key: {args.key_dim}, head: {self.num_heads}"
        self.head_dim = int(args.mask_attn.embed_dim // self.num_heads) # should be key_dim / num_heads, integer divisible

        key_args = copy.deepcopy(args)
        key_args.num_inputs = self.key_dim
        key_args.num_outputs = self.model_dim * self.num_heads
        key_args.dropout = args.mask_attn.attention_dropout
        key_args.use_layer_norm = True
        self.key_network = MLPNetwork(key_args)

        query_args = copy.deepcopy(args)
        query_args.num_inputs = self.query_dim
        query_args.output_dim = self.model_dim * self.num_heads
        query_args.object_dim = self.query_dim
        query_args.dropout = args.mask_attn.attention_dropout
        query_args.use_layer_norm = True
        aggregate_final = False
        self.query_network = ConvNetwork(query_args)

        value_args = copy.deepcopy(args)
        value_args.num_outputs = self.model_dim * self.num_heads
        value_args.pair.object_dim = self.query_dim
        value_args.pair.first_obj_dim = self.key_dim
        value_args.dropout = args.mask_attn.attention_dropout
        value_args.use_layer_norm = True
        value_args.pair.aggregate_final = False
        self.value_network = PairNetwork(value_args)

        final_args = copy.deepcopy(args)
        final_args.num_inputs = self.model_dim * self.num_heads
        final_args.num_outputs = self.key_dim
        final_args.dropout = args.mask_attn.attention_dropout
        final_args.use_layer_norm = True
        final_args.hidden_sizes = final_args.pair.final_layers
        self.final_network = MLPNetwork(final_args)

        self.model = [self.key_network, self.query_network, self.value_network, self.final_network]

    def mask_softmax(self, softmax, mask):
        # print(softmax.shape, mask.shape, self.iscuda)
        return (softmax.transpose(-2,-1) * mask.unsqueeze(-1)).transpose(-2,-1)

    def forward(self, key, queries, mask=None):
        start = time.time()
        batch_size = key.shape[0]
        value_inputs = torch.cat([key.unsqueeze(1)] + [queries], dim=1).reshape(batch_size, -1) # todo: relative state operations here
        key = self.key_network(key)
        queries = self.query_network(queries.transpose(-2,-1)).transpose(-2,-1)
        values = self.value_network(value_inputs).reshape(batch_size, self.num_heads, -1, self.model_dim)
        key, queries = key.reshape(batch_size, self.num_heads, self.model_dim, 1), queries.reshape(batch_size, self.num_heads, -1, self.model_dim)
        # print(key.shape, queries.shape, values.shape)
        values = torch.matmul(self.mask_softmax(self.softmax(torch.matmul(queries, key)[...,0]), mask).unsqueeze(-2) / np.sqrt(self.key_dim), values)
        values = values.reshape(batch_size, -1)
        values = self.final_network(values)
        # print("head compute", time.time() - start)
        return values

class MultiHeadAttentionCluster(Network):
    def __init__(self, args, mask=None):
        if mask is not None:
            self.inter_mask = mask.detach() # make sure this mask cannot change
        else:
            self.inter_mask = torch.Parameter(torch.zeros(args.mask_attn.mask_dim))
        self.num_layers = args.mask_attn.num_layers
        layers = [MultiHeadAttentionParallelLayer(args) for i in range(self.num_layers)] + [self.inter_mask]

        self.model = layers

    def forward(self, key, queries):
        for i in range(self.num_layers):
            key = self.multi_head_attention[i](key, queries, self.inter_mask)
        key = self.final_layer(key)
        return key

# mixture of interaction experts code
class MultiHeadAttentionHot(Network):
    def __init__(self, args):
        super().__init__(args)
        self.mask_dim = args.mask_attn.mask_dim
        self.key_embed_dim = args.mask_attn.embed_dim
        # TODO: make the cluster function into a single operation instead of a loop
        first_layer = [MultiHeadAttentionCluster(args, mask=pytorch_model.wrap(args.mask_attn.passive_mask))]
        layers = first_layer + [MultiHeadAttentionCluster(args) for i in range(args.mask_attn.num_clusters - 1)]
        self.multi_head_attention_clusters = nn.ModuleList(layers)

        final_args = copy.deepcopy(args)
        final_args.include_last = True
        final_args.num_inputs = final_args.mask_attn.model_dim
        final_args.num_outputs = self.num_outputs
        final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
        self.final_layer = MLPNetwork(final_args)

        self.model = layers + [self.final_layer]

    def reset_passive_mask(self, passive_mask):
        self.multi_head_attention_clusters[0].inter_mask = pytorch_model.wrap(passive_mask, self.iscuda)

    def forward(self, key, queries, m):
        total_mask = torch.zeros(key.shape[0], self.mask_dim)
        total_key = torch.zeros(key.shape[0], self.key_embed_dim)
        for i in range(self.num_clusters): # we could probably do this in parallel
            inter_mask, k = self.multi_head_attention_clusters[i](key, queries)
            total_mask += inter_mask * m[...,i].unsqueeze(-1)
            total_key += k * m[...,i].unsqueeze(-1)
        k = self.final_layer(k)
        return total_mask, k

class MultiHeadAttentionBase(Network):
    def __init__(self, args):
        super().__init__(args)
        # just handles the final reasoning logic and multiple layers (on a single key, it reprocesses the key input for each layer)
        self.num_layers = args.mask_attn.num_layers
        layers = [MultiHeadAttentionParallelLayer(args) for i in range(self.num_layers)]
        self.multi_head_attention = nn.ModuleList(layers)
        self.embed_dim = args.mask_attn.embed_dim

        final_args = copy.deepcopy(args)
        final_args.include_last = True
        final_args.num_inputs = final_args.mask_attn.embed_dim
        final_args.num_outputs = final_args.mask_attn.embed_dim
        final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
        self.final_layer = MLPNetwork(final_args)

        self.model = layers + [self.final_layer]

    def reset_passive_mask(self, passive_mask):
        return

    def forward(self, key, queries, m):
        batch_size = key.shape[0]
        for i in range(self.num_layers):
            start = time.time()
            key = self.multi_head_attention[i](key, queries, m)
            # print("layer compute", time.time() - start)
        key = self.final_layer(key)
        return m, key


class MaskedAttentionNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        self.object_dim = args.pair.object_dim # the object dim is the dimension of the value input
        self.single_object_dim = args.pair.single_obj_dim
        self.first_obj_dim = args.pair.first_obj_dim # this should include all the instances of the object, should be divisible by self.object_dim
        self.aggregate_final = args.pair.aggregate_final
        self.reduce_fn = args.pair.reduce_function

        self.model_dim = args.mask_attn.model_dim
        self.embed_dim = args.mask_attn.embed_dim
        self.return_mask = args.mask_attn.return_mask
        self.total_obj_dim = args.pair.total_obj_dim
        self.expand_dim = args.pair.expand_dim
        self.total_instances = args.pair.total_instances
        self.total_targets = args.pair.total_targets

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


        if args.pair.aggregate_final:
            final_args = copy.deepcopy(args)
            final_args.include_last = True
            final_args.num_inputs = final_args.mask_attn.embed_dim
            final_args.num_outputs = self.num_outputs
            final_args.hidden_sizes = final_args.pair.final_layers # TODO: hardcoded final hidden sizes for now
            self.final_layer = MLPNetwork(final_args)
            layers = [] 
            attention_args = args
        else:
            attention_args = copy.deepcopy(args)
            attention_args.output_dim = args.pair.single_obj_dim
            layers = list()    
        self.multi_head_attention = MultiHeadAttentionCluster(args) if args.mask_attn.cluster else MultiHeadAttentionBase(args)
        layers = [self.multi_head_attention] + layers

        self.model = layers
        self.train()
        self.reset_network_parameters()

    def reset_passive_mask(self, passive_mask):
        self.multi_head_attention.reset_passive_mask(passive_mask)

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

    def forward(self, x, m=None):
        # x is an input of shape [batch, flattened dim of all target objects + flattened all query objects]
        # m is the query mask OR the one hot over head cluster mask
        # iterate over each instance
        start = time.time()
        batch_size = x.shape[0]
        keys, queries = self.slice_input(x) # [batch, n_k, single_obj_dim], [batch, n_k, obj_dim]
        keys = self.key_encoding(keys) # [batch, n_k, d_k]
        queries = self.query_encoding(queries) # [batch, n_q, d_q]
        queries = queries.transpose(-2,-1)
        values = list() # the final output values
        masks = list()
        if m is None: m = pytorch_model.wrap(torch.ones((batch_size, queries.shape[1])), cuda=self.iscuda) # create ones of [batch, n_q]
        for i in range(int(self.first_obj_dim // self.single_object_dim)):
            key = keys[...,i]
            mask, value = self.multi_head_attention(key, queries, m)
            masks.append(mask), values.append(value)
        # print(value[0].shape)
        x = torch.stack(values, dim=2) # [batch size, pairnet output dim, num_instances]
        m = torch.stack(masks, dim=2) # [batch size, pairnet output dim, num_instances]
        # print(x.shape, self.aggregate_final)
        if self.aggregate_final:
            x = reduce_function(self.reduce_fn, x) # reduce the stacked values along axis 2
            x = x.view(-1, self.embed_dim)
            # print(x.shape)
            x = self.final_layer(x)
        else:
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
            m = m.transpose(2,1)
            m = m.reshape(batch_size, -1)
        # print("total compute", time.time() - start)
        if self.return_mask: return m, x
        return x
