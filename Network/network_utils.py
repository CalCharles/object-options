import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def run_optimizer(optimizer, model, loss):
    optimizer.zero_grad()
    (loss.mean()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

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
                return data.clone().detach().cuda(device=device)
            else:
                return data.clone().detach()
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
    if red == "add": x = torch.sum(x, dim=2)
    elif red == "prod": x = torch.prod(x, dim=2)
    elif red == "max": x = torch.max(x, 2, keepdim=True)[0]
    return x
