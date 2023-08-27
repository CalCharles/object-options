import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy, time
from Network.network_utils import pytorch_model, get_acti
MASK_ATTENTION_TYPES = ["maskattn", "parattn", "rawattn", "multiattn"] # currently only one kind of mask attention net

def apply_symmetric(symmetric_key_query, tobs, masked=None):
    if symmetric_key_query: return torch.cat([tobs.clone(), tobs], dim=-1)
    return tobs

def get_hot_mask(num_clusters, batch_size, num_keys, num_queries, i, iscuda): # if called with a numpy output, needs to be unwrapped
    if batch_size <= 0:
        if i >= 0:
            hot_vals = pytorch_model.wrap(torch.zeros(num_keys, num_clusters), cuda=iscuda)
            hot_vals[...,i] = 1
        else: hot_vals = pytorch_model.wrap(torch.ones(num_keys, num_clusters), cuda=iscuda) / float(num_clusters)
        return hot_vals 
    if i >= 0:
        hot_vals = pytorch_model.wrap(torch.zeros(batch_size, num_keys, num_clusters), cuda=iscuda)
        hot_vals[...,i] = 1
    else: hot_vals = pytorch_model.wrap(torch.ones(batch_size, num_keys, num_clusters), cuda=iscuda) / float(num_clusters)
    return hot_vals 

def get_active_mask(batch_size, num_keys, num_queries, iscuda):
    if batch_size <= 0: return pytorch_model.wrap(torch.ones(num_keys, num_queries), cuda=iscuda)
    return pytorch_model.wrap(torch.ones(batch_size, num_keys, num_queries), cuda=iscuda)

def get_passive_mask(batch_size, num_keys, num_queries, num_objects, class_index, iscuda):
    passive_masks = torch.zeros(num_keys, num_objects)
    indices = np.arange(num_keys) + class_index
    passive_masks[:, indices] = 1
    if batch_size <= 0: return pytorch_model.wrap(passive_masks, cuda=iscuda)
    return pytorch_model.wrap(passive_masks.broadcast_to(batch_size, num_keys, num_objects), cuda=iscuda)


def expand_mask(m, batch_size, object_dim): # only for padded networks
    # m = batch x num_keys*num_objects OR
    # batch x num_keys*num_objects if broadcast over keys
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

def count_keys_queries(first_obj_dim, single_obj_dim, object_dim, x):
    return first_obj_dim // single_obj_dim, int((x.shape[-1] - first_obj_dim) // object_dim)