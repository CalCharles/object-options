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
from Network.Dists.forward_mask import DiagGaussianForwardPadMaskNetwork
import copy, time


class DiagGaussianForwardMultiMaskNetwork(Network):
    '''
    Class for EM based algorithms, where there are num_masks forward networks but \
    one shared embedding network. set_index sets the index of the network to the \
    desired network to run at a given time, and self.forward_models contains the \
    forward models at indexes, for optimization
    '''
    def __init__(self, args):
        super().__init__(args)
        self.index = 0
        self.num_networks = args.multi.num_masks

        self.object_dim = args.pair.object_dim
        self.embedding_output = args.multi.embedding_output
        self.first_obj_dim = args.pair.first_obj_dim
        self.symmetric_key_query = args.symmetric_key_query

        # create embedding network as a 1d convolution (used on the queries)
        embed_args = copy.deepcopy(args)
        embed_args.query_aggregate = False
        embed_args.hidden_sizes = args.multi.embedding_sizes
        embed_args.pair.object_dim = self.object_dim
        embed_args.output_dim = self.embedding_output # must have embed_inputs
        embed_args.activation_final = embed_args.activation
        embed_args.pair.aggregate_final = False
        embed_args.include_last = True
        self.embedding = ConvNetwork(embed_args)

        self.forward_models = list()
        forward_args = copy.deepcopy(args)
        forward_args.pair.object_dim = args.multi.embedding_output
        forward_args.symmetric_key_query = False # the symmetric key query is applied at the embed level
        if self.symmetric_key_query:
            forward_args.pair.first_obj_dim = int(args.pair.first_obj_dim // self.object_dim * args.multi.embedding_output)
            forward_args.pair.single_obj_dim = args.multi.embedding_output
        for i in range(forward_args.multi.num_masks):
            self.forward_models.append(DiagGaussianForwardPadMaskNetwork(forward_args))
        self.forward_models = nn.ModuleList(self.forward_models)
        print(forward_args.multi.num_masks)

        self.model = [self.embedding, self.forward_models]


        self.train()
        self.reset_network_parameters()
    
    def reset_environment(self, class_index, num_objects, first_obj_dim):
        self.first_obj_dim = first_obj_dim
        submodel_first = int(first_obj_dim // self.object_dim * self.embedding_output) if self.symmetric_key_query else first_obj_dim
        for model in self.forward_models:
            model.reset_environment(class_index, num_objects, submodel_first)
        self.total_instances = num_objects

    def set_index(self, idx):
        self.index = idx
    
    def reset_index(self, idx, optimizer_args, embedding_optimizer=False):
        self.index= idx
        self.forward_models[self.index].reset_network_parameters()
        plist = self.forward_models[self.index].parameters()
        if embedding_optimizer: plist+= self.embedding.parameters()
        optimizer = optim.Adam(plist,
                optimizer_args.lr, eps=optimizer_args.eps, betas=optimizer_args.betas, weight_decay=optimizer_args.weight_decay)
        return optimizer

    def get_queries(self, x):
        first_dim = 0 if self.symmetric_key_query else max(0, self.first_obj_dim)
        num_obj = int((x.shape[-1] - first_dim) // self.object_dim)
        queries = x[...,first_dim:].reshape(-1, num_obj, self.object_dim)
        return queries

    def forward(self, x, m=None, soft=False, mixed=False, flat=False, full=False):
        # keyword hyperparameters are used only for consistency with the mixture of experts model
        # start = time.time()
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        # print("mask", m, x, self.iscuda)
        q = self.embedding(self.get_queries(x).transpose(-2,-1)).transpose(-2,-1) # embedding replaces the queries
        if self.symmetric_key_query: x = torch.cat([(q.clone()).reshape(x.shape[0], -1), (q).reshape(x.shape[0], -1)], dim=-1) # apply the queries, and mask if symmetric
        else: x = torch.cat([x[...,:self.first_obj_dim],q.reshape(x.shape[0], -1)], dim=1) # reappend the embedded queries
        # print("mqx", m.shape, q.shape, x.shape)
        # else: q = (q * m.unsqueeze(-1)).reshape(x.shape[0], -1) # apply the mask if not symmetric

        meanvar, m = self.forward_models[self.index](x, m=m, soft=soft, mixed=mixed, flat=flat, full=full)
        return meanvar, m
