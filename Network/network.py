import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Network.network_utils import get_acti

## end of normalization functions
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_inputs, self.num_outputs = args.num_inputs, args.num_outputs
        self.use_layer_norm = args.use_layer_norm
        self.hs = [int(h) for h in args.hidden_sizes]
        self.init_form = args.init_form
        self.layers = []
        self.acti = get_acti(args.activation)
        self.activation_final = get_acti(args.activation_final)
        self.iscuda = False

    def cuda(self):
        super().cuda()
        self.iscuda = True

    def cpu(self):
        super().cpu()
        self.iscuda = False

    def reset_parameters(self):
        # initializes the weights by iterating through ever layer of the model
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.model:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth": nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else: nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Network):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))
            elif type(layer) == nn.Linear or type(layer) == nn.Conv1d:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                    elif self.init_form == "uni":
                         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                    elif self.init_form == "smalluni":
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "knorm":
                        torch.nn.init.kaiming_normal_(layer.weight.data)
                    elif self.init_form == "kuni":
                        torch.nn.init.kaiming_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)

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
network_type = {"mlp": MLPNetwork, "pair": PairNetwork}