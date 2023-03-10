from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from Network.General.key_pair import KeyPairNetwork
from Network.General.conv import ConvNetwork
from Network.network_utils import pytorch_model


class EmbedNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        self.hash_dim = args.comb_embed.hash_dim
        self.embed_dim = args.embed_inputs
        self.embed_type = args.comb_embed.embed_type

        self.encoders = list()
        for i in range(self.hash_dim ** 2):
            encode_layer_args = copy.deepcopy(args)
            encode_layer_args.object_dim = args.pair.object_dim
            encode_layer_args.include_last = False
            encode_layer_args.output_dim = self.embed_dim
            encode_layer_args.aggregate_final = True
            encode_layer_args.hidden_sizes = list()
            self.encoders.append(KeyPairNetwork(encode_layer_args))
        self.encoders = nn.ModuleList(self.encoders)
        decode_args = copy.deepcopy(args)
        decode_args.object_dim = self.embed_dim
        decode_args.output_dim = args.pair.single_object_dim
        decode_args.aggregate_final = True
        self.decoder = ConvNetwork(decode_args)
        self.decode_vector = np.array([2 ** i for i in range(self.hash_dim)])
        self.train()
        self.reset_network_parameters()

    def mask_to_indices(self, mask): # TODO: only takes in hard masks
        return (mask * pytorch_model.wrap(self.decode_vector)).sum(-1)

    def forward(self, x, m):
        idx = self.mask_to_indices(m)
        x = self.encoders[idx](x)
        x = self.decoder(x)
        return x