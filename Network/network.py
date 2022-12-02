import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Network.network_utils import get_acti, reset_linconv, reset_parameters, count_layers

## end of normalization functions
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_inputs, self.num_outputs = args.num_inputs, args.num_outputs
        self.use_layer_norm = args.use_layer_norm
        self.hs = [int(h) for h in args.hidden_sizes]
        self.init_form = args.init_form
        self.model = []
        self.acti = get_acti(args.activation)
        self.activation_final = get_acti(args.activation_final)
        self.iscuda = False

    def cuda(self, gpu=None):
        super().cuda()
        self.iscuda = True
        for m in self.model:
            if issubclass(type(m), Network): m.cuda(gpu=gpu)
        return self

    def cpu(self):
        super().cpu()
        self.iscuda = False
        for m in self.model:
            if issubclass(type(m), Network): m.cpu()
        return self

    def reset_network_parameters(self, n_layers=-1):
        return reset_parameters(self, self.init_form, n_layers=n_layers)

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def set_parameters(self, param_val): 
        # sets the parameters of a model to the parameter values given as a single long vector
        # this is used for black box methods
        if len(param_val) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_val = param_val[pval_idx : pval_idx+param_size]
            if type(cur_param_val) == torch.Tensor:
                param.data = cur_param_val.reshape(param.size()).float().clone()
            else:
                param.data = torch.from_numpy(cur_param_val) \
                              .reshape(param.size()).float()
            pval_idx += param_size
        if self.iscuda:
            self.cuda()

    def count_parameters(self, reuse=True):
        # it may be necessary to know how many parameters there are in the model
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    def forward(self, x):
        '''
        all should have a forward function, but not all forward functions have the same signature
        '''
        return

from Network.General.mlp import MLPNetwork
from Network.General.pair import PairNetwork
from Network.General.key_pair import KeyPairNetwork
from Network.General.mask_attention import MaskedAttentionNetwork
from Network.General.input_expand import InputExpandNetwork
network_type = {"mlp": MLPNetwork, "pair": PairNetwork, "keypair": KeyPairNetwork, "maskattn": MaskedAttentionNetwork, "inexp": InputExpandNetwork}