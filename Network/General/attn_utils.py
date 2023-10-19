import numpy as np
import torch
import torch.nn.functional as F


def evaluate_key_query(softmax, keys, queries, mask, valid, single_key=False, gumbel=-1, renormalize=False):
    # computes the key query comparison
    # applies mask and valid, assuming that mask is probabilistic 
    # and valid is binary
    # Batch, heads, num queries, embed x Batch, heads, embed, num_keys 
    weights = torch.matmul(queries, keys).transpose(-1,-2) # batch, heads, num_keys, num_queries
    # if we are using a stochastic attention weights, apply the gumbel softmax instead here
    if gumbel > 0: weights = F.gumbel_softmax(weights, tau = gumbel, hard = False, dim=-1) # does not change shape, but assumes queries in last layer
    else: weights = softmax(weights / np.sqrt(queries.shape[-1])) # softmax expected along dim=-1, values in 0,1
    # masks override the weights, then renormalizes TODO: use -inf instead?
    # if valid is not None: print(valid.shape, mask.shape, queries.shape, keys.shape, weights.shape)
    if mask is not None:
        # print(queries.shape, keys.shape, weights.shape, mask.shape)
        if len(mask.shape) == 2: mask = mask.unsqueeze(-2)# check if mask includes keys
        weights = weights * torch.broadcast_to(mask.unsqueeze(1), (weights.shape[0], weights.shape[1], 1, mask.shape[-1])) # Batch, heads, num_keys, num queries x Batch, heads, 1, num_queries
    if valid is not None: 
        if len(valid.shape) == 2: valid = valid.unsqueeze(-2)# check if mask includes keys
        weights = weights * torch.broadcast_to(valid.unsqueeze(1), (weights.shape[0], weights.shape[1], 1, mask.shape[-1])) # Batch, heads, num_keys, num queries x Batch, heads, 1, num_queries
    if renormalize: weights = weights / (weights.sum(axis=-1).unsqueeze(-1) + 1e-4) # renormalizing along queries after zeroing out
    if single_key: weights = weights[:, :, 0]
    return weights # batch x heads x keys (if not single key) x queries 

def mask_query(queries, mask, valid, single_key = False):
    if mask is not None: queries = queries * (mask.unsqueeze(-1) if single_key else mask)
    if valid is not None: queries = queries * valid.unsqueeze(-1)
    return queries