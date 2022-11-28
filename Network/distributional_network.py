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

class DiagGaussianForwardMaskNetwork(Network):
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

        self.total_object_sizes = [args.total_object_sizes[n] for n in args.object_names]

        self.train()
        self.reset_network_parameters()

    def expand_mask(self, m):
        # m = batch x num_objects
        # TODO: make this not a for loop
        comb = list()
        for i in range(m.shape[-1]):
            comb.append(m[...,i] * pytorch_model.wrap(torch.ones(self.total_object_sizes[i]), cuda=self.iscuda))
        return torch.cat(comb, dim=-1)

    def forward(self, x, m):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        if not self.hot: x = x * self.expand_mask(m)
        mask, mean = self.mean(x)
        _, var = self.std(x)
        return torch.tanh(mean), torch.sigmoid(var) + self.base_variance, mask

class DiagGaussianForwardPadMaskNetwork(Network):
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
        self.hot = args.mask_attn.cluster
        self.maskattn = args.net_type in ["maskattn"] # currently only one kind of mask attention net

        self.object_dim = args.object_dim

        self.train()
        self.reset_network_parameters()

    def expand_mask(self, m):
        # m = batch x num_objects
        # TODO: make this not a for loop
        comb = list()
        for i in range(m.shape[-1]):
            comb.append(m[...,i].unsqueeze(-1) * pytorch_model.wrap(torch.ones(self.object_dim), cuda=self.iscuda))
        return torch.cat(comb, dim=-1)

    def forward(self, x, m):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        if not (self.hot or self.maskattn): m = self.expand_mask(m)
        mask, mean = self.mean(x, m)
        _, var = self.std(x, m)
        return (torch.tanh(mean), torch.sigmoid(var) + self.base_variance), mask

class InteractionMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        inter_args = copy.deepcopy(args)
        inter_args.num_outputs = inter_args.pair.total_instances
        inter_args.activation_final = "sigmoid"
        self.inter = network_type[args.net_type](inter_args)
        self.model = [self.inter]

        self.train()
        self.reset_network_parameters()
        
    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        v = self.inter.forward(x)
        return v
