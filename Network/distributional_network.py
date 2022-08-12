import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, network_type
from Network.network_utils import pytorch_model, get_acti
import copy

class DiagGaussianForwardNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        self.mean = network_type[args.net_type](mean_args)
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance

        self.train()
        self.reset_network_parameters()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + self.base_variance

class InteractionNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = 1
        inter_args.activation_final = "sigmoid"
        self.inter = network_type[args.net_type](inter_args)
        self.model = [self.inter]

        self.train()
        self.reset_network_parameters()
        
    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        return v
