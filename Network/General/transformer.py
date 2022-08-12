from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# TODO: INCOMPLETE
class TransformerPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_final = kwargs["aggregate_final"]
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        # requires a fixed number of objects for regular pairwise sizes

        kwargs["aggregate_final"] = False
        kwargs["hidden_sizes"] = self.hidden_sizes[:len(self.hidden_sizes) - 1]
        kwargs["output_dim"] = self.hidden_sizes[-1]
        self.key_dim = self.hidden_sizes[-1]
        self.key_network = PairNetwork(**kwargs)
        kwargs["drop_first"] = True
        self.query_network = PairNetwork(**kwargs)
        kwargs["drop_first"] = False
        
        self.value_network = PairNetwork(**kwargs)
        self.softmax =  nn.Softmax(-1)
        
        self.layers = [self.key_network, self.value_network, self.query_network, self.softmax]

        kwargs["aggregate_final"] = self.aggregate_final
        kwargs["object_dim"] = self.hidden_sizes[-1]
        kwargs["output_dim"] = self.output_dim
        kwargs["hidden_sizes"] = [128, 128] # hardcoded final layer
        self.output_network = PointNetwork(**kwargs)
        self.layers.append(self.output_network)

        # reset kwargs
        kwargs["object_dim"] = self.object_dim
        kwargs["hidden_sizes"] = self.hidden_sizes
        self.model = nn.ModuleList(self.layers)
        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        kv_x, fx, px, batch_size = self.key_network.slice_input(x)
        K = self.key_network.run_networks(kv_x, px, batch_size).reshape(batch_size, -1, self.key_dim) # B x nobj x d
        V = self.value_network.run_networks(kv_x, px, batch_size).reshape(batch_size, -1, self.key_dim)
        q_x, fx, px, batch_size = self.query_network.slice_input(x)
        Q = self.query_network.run_networks(q_x, px, batch_size).reshape(batch_size, -1, self.key_dim)
        # print("KVQ", K.shape, V.shape, Q.shape)
        alpha = self.softmax(torch.matmul(Q, K.transpose(2,1)) / np.sqrt(self.key_dim)) # B x nobj x nobj keys along rows dim -1 
        # print(alpha.sum(), alpha.abs().sum())
        Z = torch.matmul(alpha, V) # B x nobj x d
        # TODO: pairwise information in a single dimension
        # print("zhape", Z.shape)
        x = Z.reshape(batch_size, -1)
        # print(x.abs().sum(), x.shape)
        x = self.output_network(x)
        # print(x.abs().sum(), x.sum())
        return x

class MultiheadedTransformerPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_final = kwargs["aggregate_final"]
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        # requires a fixed number of objects for regular pairwise sizes
        self.num_heads = kwargs["num_heads"]

        kwargs["hidden_sizes"] = self.hidden_sizes[:len(self.hidden_sizes) - 1]
        kwargs["output_dim"] = kwargs["hidden_sizes"][-1]
        self.heads = list()
        for h in range(self.num_heads):
            self.heads.append(TransformerPairNetwork(**kwargs))

        kwargs["object_dim"] = self.hidden_sizes[-1] * self.num_heads
        kwargs["hidden_sizes"] = [128, 128, 128] # hardcoded final layer
        kwargs["output_dim"] = self.output_dim
        self.output_network = PointNetwork(**kwargs)
        self.layers = self.heads + [self.output_network]
        
        # reset kwargs
        kwargs["object_dim"] = self.object_dim
        kwargs["hidden_sizes"] = self.hidden_sizes
        self.model = nn.ModuleList(self.layers)
        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        outputs = list()
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        for head in self.heads:
            outputs.append(head(x))
        outputs = torch.cat(outputs, dim=-1)
        result = self.output_network(outputs)
        if self.aggregate_final:
            return result
        else:
            return result.reshape(batch_size, -1)