import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy, time
from Network.network import Network, network_type
from Network.network_utils import pytorch_model, get_acti



def expand_mask(m, batch_size, object_dim): # only for padded networks
    # m = batch x num_objects
    # TODO: make this not a for loop
    return torch.broadcast_to(m.unsqueeze(-1), (batch_size, m.shape[-1], object_dim)).reshape(batch_size, object_dim*m.shape[-1])
    # comb = list()
    # for i in range(m.shape[-1]):
    #     comb.append(m[...,i].unsqueeze(-1) * pytorch_model.wrap(torch.ones(object_dim), cuda=iscuda))
    # return torch.cat(comb, dim=-1)

def apply_probabilistic_mask(inter_mask, inter_dist=None, relaxed_inter_dist=None, mixed = False, test=None, dist_temperature=0, revert_mask=False):
    # applys the logic for a probabilistic mask
    # test = flat, which uses a threshold to decide the interaction mask
    # print("dist operation", inter_mask[:4], relaxed_inter_dist, inter_dist, test, mixed)
    if test is not None: return pytorch_model.unwrap(test(inter_mask)) if revert_mask else test(inter_mask)
    # print(soft, self.relaxed_inter_dist, revert_mask, inter_mask)
    # inter_dist = hard, which uses a hard sample
    if inter_dist is not None: 
        if mixed:
            return pytorch_model.unwrap(inter_dist(inter_mask).sample() * inter_mask) if revert_mask else inter_dist(inter_mask).sample() * inter_mask
        else:
            return pytorch_model.unwrap(inter_dist(inter_mask).sample()) if revert_mask else inter_dist(inter_mask).sample()  # hard masks don't need gradient
    # relaxed_inter is none is for weighted interaction, which uses the values directly
    # otherwise, use the soft distribution for interaction
    if relaxed_inter_dist is None: return pytorch_model.unwrap(inter_mask) if revert_mask else inter_mask
    else: return pytorch_model.unwrap(relaxed_inter_dist(dist_temperature, probs=inter_mask).rsample()) if revert_mask else relaxed_inter_dist(dist_temperature, probs=inter_mask).rsample()
