
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Network.network_utils import get_acti,get_inplace_acti, reset_linconv, reset_parameters, count_layers

## end of normalization functions
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_inputs, self.num_outputs = args.num_inputs, args.num_outputs
        self.use_layer_norm = args.use_layer_norm
        self.hs = [int(h) for h in args.hidden_sizes]
        self.init_form = args.init_form
        self.model = []
        self.acti = get_inplace_acti(args.activation)
        self.activation_final = get_inplace_acti(args.activation_final)
        self.activation_final_name = args.activation_final
        self.iscuda = False # this means we have to run .cuda() to get it on the GPU
        self.gpu = args.gpu

    def cuda(self, gpu=None):
        super().cuda()
        self.iscuda = True
        if gpu is not None: self.gpu = gpu
        for m in self.model:
            if issubclass(type(m), Network): m.cuda(gpu=gpu)
            else: m.cuda()
            # if hasattr(m, "weight"): print(type(m), m.weight.data)
        return self

    def cpu(self):
        super().cpu()
        self.iscuda = False
        for m in self.model:
            if issubclass(type(m), Network): m.cpu()
        return self

    def reset_network_parameters(self, n_layers=-1):
        return reset_parameters(self, self.init_form, n_layers=n_layers)


    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def forward(self, x):
        '''
        all should have a forward function, but not all forward functions have the same signature
        '''
        return

from Network.General.mlp import MLPNetwork
from Network.General.pair import PairNetwork
from Network.General.key_pair import KeyPairNetwork
from Network.General.linear_pair import LinearKeyPairNetwork
from Network.General.mask_attention import MaskedAttentionNetwork
from Network.General.parallel_attention import ParallelMaskedAttentionNetwork
from Network.General.multi_mask_attention import MultiMaskedAttentionNetwork
from Network.General.input_expand import InputExpandNetwork
from Network.General.raw_attention import RawAttentionNetwork
from Network.General.key_embed import EmbedPairNetwork
network_type = {"mlp": MLPNetwork, "pair": PairNetwork, "keypair": KeyPairNetwork, "keyembed": EmbedPairNetwork,
                "multiattn": MultiMaskedAttentionNetwork, "maskattn": MaskedAttentionNetwork, "parattn": ParallelMaskedAttentionNetwork,
                "rawattn": RawAttentionNetwork, "inexp": InputExpandNetwork, "linpair": LinearKeyPairNetwork}