import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, network_type
from Network.network_utils import pytorch_model, get_acti
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork
from Network.General.pair import PairNetwork
from Network.Dists.mask_utils import expand_mask, apply_symmetric
import copy, time


class DiagGaussianForwardMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        self.mean = network_type[args.net_type](mean_args)
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.needs_expand_mask = args.needs_expand_mask
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance

        self.total_object_sizes = [args.total_object_sizes[n] for n in args.object_names]

        self.train()
        self.reset_network_parameters()

    def expand_mask(self, m, batch_size=-1):
        # m = batch x num_objects
        # TODO: make this not a for loop
        comb = list()
        for i in range(m.shape[-1]):
            comb.append(m[...,i] * pytorch_model.wrap(torch.ones(self.total_object_sizes[i]), cuda=self.iscuda))
        return torch.cat(comb, dim=-1)

    def forward(self, x, m, valid=None):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        if valid is not None: m = m * valid # valid should be [batch, num_objects], m: [batch, num_objects]
        if not self.hot: x = x * self.expand_mask(m)
        mask, mean = self.mean(x)
        _, var = self.std(x)
        return torch.tanh(mean), torch.sigmoid(var) + self.base_variance, mask

class DiagGaussianForwardPadMaskNetwork(Network):
    def __init__(self, args):
        super().__init__(args)

        mean_args = copy.deepcopy(args)
        mean_args.activation_final = "none"
        # mean_args.num_outputs = mean_args.num_outputs * 2
        self.mean = network_type[args.net_type](mean_args)
        self.model = [self.mean]
        std_args = copy.deepcopy(args)
        std_args.activation_final = "none"
        self.std = network_type[args.net_type](std_args)
        self.model = [self.mean, self.std]
        self.base_variance = .01 # hardcoded based on normalized values, base variance 1% of the average variance
        self.cluster_mode = args.cluster.cluster_mode
        self.attention_mode = args.attention_mode
        self.keyembed_mode = args.net_type in ["keyembed"]
        self.num_clusters = args.cluster.num_clusters
        self.maskattn = args.net_type in ["maskattn", "rawattn"] # currently only one kind of mask attention net
        self.mask_dim = args.pair.total_instances # does not handle arbitary number of instances
        self.symmetric_key_query = args.symmetric_key_query # if these are the same, then we need to cat x to itself to fit key-query expectations

        self.object_dim = args.object_dim
        self.embed_dim = args.embed_inputs * max(1, args.mask_attn.num_heads * int(self.maskattn) )

        self.train()
        self.reset_network_parameters()

    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        self.class_index = class_index
        self.num_objects = num_objects
        if hasattr(self.mean, "reset_environment"): 
            self.mean.reset_environment(class_index, num_objects, first_obj_dim)
            self.std.reset_environment(class_index, num_objects, first_obj_dim)


    def get_masks(self):
        if self.maskattn:
            return 
        else:
            return self

    def forward(self, x, m=None, soft=False, mixed=False, flat=False, full=False, valid =None):
        # keyword hyperparameters are used only for consistency with the mixture of experts model
        # start = time.time()
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = apply_symmetric(self.symmetric_key_query, x)
        # if not (self.cluster_mode or self.maskattn or self.attention_mode or self.keyembed_mode): m = expand_mask(m, x.shape[0], self.embed_dim) # self.object_dim
        # else: m = expand_mask(m, x.shape[0], 1)
        # print("mask",time.time() - start)
        if self.attention_mode:
            mean, m1 = self.mean(x, m, hard = not soft, valid=valid)
            var, m2 = self.std(x, m, hard = not soft, valid=valid)
            m = torch.stack([m1,m2], dim=-1).max(dim=-1)[0]
        else:
            mean = self.mean(x, m, valid=valid)
            var = self.std(x, m, valid=valid)
        # print("meanvar",time.time() - start)
        mean = (torch.tanh(mean))
        var = (torch.sigmoid(var) + self.base_variance)
        # meanvar = self.mean(x, m)
        # mean = (torch.tanh(meanvar[...,:int(meanvar.shape[-1]//2)]))
        # var = (torch.sigmoid(meanvar[...,int(meanvar.shape[-1]//2):]) + self.base_variance)
        # print("forward",time.time() - start)
        return (mean, var), m

    def embeddings(self, x, valid=None):
        mean_e, recon = self.mean(x, return_embeddings=True, valid=valid)
        std_e, recon = self.std(x, return_embeddings=True, valid=valid)
        return mean_e, std_e, recon

    def reconstruction(self, x, valid=None):
        mean_r, recon = self.mean(x, return_reconstruction=True, valid=valid)
        std_r, recon = self.std(x, return_reconstruction=True, valid=valid)
        return mean_r, std_r, recon

    def weights(self, x, valid=None):
        mean_w, weights = self.mean(x, return_weights=True, valid=valid)
        std_w, weights = self.std(x, return_weights=True, valid=valid)
        return mean_w, std_w, weights