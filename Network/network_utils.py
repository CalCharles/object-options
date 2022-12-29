import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tianshou as ts
import copy

def run_optimizer(optimizer, model, loss, grad_variables=[]):
    optimizer.zero_grad()
    (loss.mean()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    grad_variables = [copy.deepcopy(gv.grad) for gv in grad_variables]  # stores the gradient of variables
    optimizer.step()
    return grad_variables

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def get_gradient(model, loss, grad_variables=[]):
    # gradients = list()
    # for gv in grad_variables:
    #     gv.register_hook(set_grad(gv))
    #     print(gv)
    #     gradients.append(torch.autograd.grad(loss.mean(), gv, retain_graph= True))
    # return gradients
    for gv in grad_variables:
        print(gv.requires_grad)
        gv.retain_grad()
    (loss.mean()).backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    grad_variables = [copy.deepcopy(gv.grad) for gv in grad_variables]  # stores the gradient of variables
    print(grad_variables[0].shape)
    return grad_variables


def cuda_string(device):
    if device < 0: device = "cpu"
    else: device = "cuda:" + str(device) 


class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True, device = None):
        if type(data) == torch.Tensor:
            if cuda: # TODO: dtype not handeled 
                # return data.clone().detach().cuda(device=device)
                return data.cuda(device=device)
            else:
                # return data.clone().detach()
                return data
        else:
            if cuda:
                return torch.tensor(data, dtype=dtype).cuda(device=device)
            else:
                return torch.tensor(data, dtype=dtype)

    @staticmethod
    def unwrap(data):
        if type(data) == torch.Tensor:
            return data.clone().detach().cpu().numpy()
        else:
            return data

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

# # normalization functions
class NormalizationFunctions():
    def __init__(self, **kwargs):
        pass

    def __call__(self, val):
        return

    def reverse(self, val):
        return

class ConstantNorm(NormalizationFunctions):
    def __init__(self, **kwargs):
        self.mean = kwargs['mean']
        self.std = kwargs['variance']
        self.inv_std = kwargs['invvariance']

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        return (val - self.mean) * self.inv_std

    def reverse(self, val):
        return val * self.std + self.mean

    def cuda(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            self.inv_std = self.inv_std.cuda()

    def cpu(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
            self.inv_std = self.inv_std.cpu()

def identity(x):
    return x

def get_acti(acti):
    if acti == "relu": return F.relu
    elif acti == "sin": return torch.sin
    elif acti == "sigmoid": return torch.sigmoid
    elif acti == "tanh": return torch.tanh
    elif acti == "softmax": return nn.SoftMax(-1)(x)
    elif acti == "none": return identity

def reduce_function(red, x):
    if red == "sum": x = torch.sum(x, dim=2)
    elif red == "prod": x = torch.prod(x, dim=2)
    elif red == "max": x = torch.max(x, 2, keepdim=True)[0]
    elif red == "mean": x = torch.mean(x, dim=2)
    return x

def reset_linconv(layer, init_form):
    if init_form == "orth":
        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
    elif init_form == "uni":
         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
    elif init_form == "smalluni":
        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
    elif init_form == "zero":
        nn.init.uniform_(layer.weight.data, 0, 0)
    elif init_form == "xnorm":
        torch.nn.init.xavier_normal_(layer.weight.data)
    elif init_form == "xuni":
        torch.nn.init.xavier_uniform_(layer.weight.data)
    elif init_form == "knorm":
        torch.nn.init.kaiming_normal_(layer.weight.data)
    elif init_form == "kuni":
        torch.nn.init.kaiming_uniform_(layer.weight.data)
    elif init_form == "eye":
        torch.nn.init.eye_(layer.weight.data)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)

def reset_parameters(network, init_form, n_layers=-1):
    # initializes the weights by iterating through ever layer of the model
    relu_gain = nn.init.calculate_gain('relu')
    total_layers = count_layers(network)
    layer_at = 0
    layer_list = network.model if hasattr(network, "model") else network
    if not hasattr(layer_list, '__iter__'): layer_list = [layer_list]
    for layer in layer_list:
        size_next = count_layers(layer)
        layer_next = layer_at + size_next
        if n_layers > 0: # TODO: handle case where we have to go into a subnetwork
            at_before = total_layers - layer_at - 1 >= n_layers
            next_before = total_layers - layer_next - 1 >= n_layers
            entering = at_before and (not next_before) and size_next > 1
            if (not entering and at_before):
                layer_at = layer_next
                continue
        if type(layer) == nn.Conv2d:
            if self.init_form == "orth": nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
            elif self.init_form == "zero": nn.init.uniform_(layer.weight.data, 0,0)
            else: nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
        elif hasattr(layer, "reset_network_parameters"): # this is a resettable network
            use_layers = n_layers - (total_layers - layer_at - size_next) if n_layers > 0 else n_layers
            layer.reset_network_parameters(n_layers=use_layers)
        elif type(layer) == ts.utils.net.common.MLP:
            reset_parameters(layer, init_form)
        elif type(layer) == nn.Parameter:
            nn.init.uniform_(layer.data, 0.0, 1.0)
        else: # layer is a list or basic layer linear or conv1d
            if type(layer) == nn.Linear or type(layer) == nn.Conv1d:
                fulllayer = [layer]
            elif type(layer) == nn.ModuleList:
                fulllayer = layer
            else:
                continue
            ml_layer_at = layer_at
            for layer in fulllayer[::-1]:
                if type(layer) == nn.Linear or type(layer) == nn.Conv1d:
                    reset_linconv(layer, init_form)
                else: # it is something to be handled recursively, n_layers not handled in this case
                    size_next = count_layers(layer)
                    use_layers = n_layers - (total_layers - ml_layer_at - size_next) if n_layers > 0 else n_layers
                    reset_parameters(layer, init_form, n_layers=use_layers) # assumes modulelists are sequential
                    ml_layer_at -= size_next
        layer_at = layer_next

def compare_nets(networka, networkb, n_layers=-1, diff_list = []):
    # initializes the weights by iterating through ever layer of the model
    total_layers = count_layers(networka)
    layer_at = 0
    layer_lista = networka.model if hasattr(networka, "model") else networka
    layer_listb = networkb.model if hasattr(networkb, "model") else networkb
    if not hasattr(layer_lista, '__iter__'): layer_lista = [layer_lista]
    if not hasattr(layer_listb, '__iter__'): layer_listb = [layer_listb]
    diff = 0
    for layera, layerb in zip(layer_lista, layer_listb):
        size_next = count_layers(layera)
        layer_next = layer_at + size_next
        if n_layers > 0: # TODO: handle case where we have to go into a subnetwork
            at_before = total_layers - layer_at - 1 >= n_layers
            next_before = total_layers - layer_next - 1 >= n_layers
            entering = at_before and (not next_before) and size_next > 1
            if (not entering and at_before):
                layer_at = layer_next
                continue
        if type(layera) != type(layerb):
            return -1000000000, list() # a magic number to show it didn't work
        if type(layera) == nn.Conv2d or type(layera) == nn.Linear or type(layera) == nn.Conv1d:
            diff_val = (layera.weight.data - layerb.weight.data).abs().sum()
            diff, diff_list = diff+diff_val, diff_list.append(diff_val)
        elif type(layera) == nn.ModuleList:
            for la, lb in zip(layera, layerb):
                use_layers = n_layers - (total_layers - layer_at - size_next) if n_layers > 0 else n_layers
                diff_val, diff_list = compare_nets(la, lb, n_layers=use_layers, diff_list=diff_list)
                diff = diff+diff_val
        elif type(layera) == nn.Parameter:
            diff_val = (layera.data - layerb.data).abs().sum()
            diff, diff_list = diff+diff_val, diff_list.append(diff_val)
        elif type(layera) == ts.utils.net.common.MLP or  hasattr(layera, "reset_network_parameters"):
            use_layers = n_layers - (total_layers - layer_at - size_next) if n_layers > 0 else n_layers
            diff_val, diff_list = compare_nets(layera, layerb, n_layers=use_layers, diff_list=diff_list)
            diff = diff+diff_val
        elif type(layera) == nn.ModuleList:
            for la, lb in zip(layera, layerb):
                diff_val, diff_list = compare_nets(la, lb, n_layers=use_layers, diff_list=diff_list)
                diff = diff + diff_val
        layer_at = layer_next
    return diff, diff_list

def count_layers(network):
    total_layers = 0
    layer_list = network.model if hasattr(network, "model") else network
    if not hasattr(layer_list, '__iter__'): layer_list = [layer_list]
    for layer in layer_list:
        if type(layer) == nn.LayerNorm: # we don't count layer norms
            continue
        elif hasattr(layer, "model"):
            total_layers += count_layers(layer)
        elif type(layer) == nn.Parameter or type(layer) == nn.Linear or type(layer) == nn.Conv1d or type(layer) == nn.Conv2d:
            total_layers += 1
        elif type(layer) == nn.ModuleList: # for modulelists, assume that all layers are at the same level
            # total_layers = max([count_layers(l)for l in layer])
            total_layers = sum([count_layers(l)for l in layer])
        else:
            pass

    return total_layers
