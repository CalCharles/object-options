import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network
from Networks.network.network_utils import pytorch_model, get_acti
import copy

class DiagGaussianForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        mean_args = copy.deepcopy(kwargs)
        mean_args["activation_final"] = "tanh"
        self.mean = network_type[kwargs["net_type"]](**mean_args)
        std_args = copy.deepcopy(kwargs)
        std_args["activation_final"] = "tanh"
        self.std = network_type[kwargs["net_type"]](**std_args)
        self.model = [self.mean, self.std]

        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + self.base_variance

class InteractionNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inter_args = copy.deepcopy(kwargs)
        kwargs["num_outputs"] = 1
        inter_args["activation_final"] = "sigmoid"
        self.inter = network_type[kwargs["net_type"]](**kwargs)
        self.train()
        self.reset_parameters()
        
    def cuda(self):
        super().cuda()

    def cpu(self):
        super().cpu()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        return v
