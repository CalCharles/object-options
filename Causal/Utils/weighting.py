import torch
import copy
import numpy as np
from Causal.Utils.get_error import get_error, error_types

def uni_weights(rollouts):
    passive_error = torch.ones(len(rollouts))
    binaries = np.ones(len(rollouts))
    weights = np.ones(len(rollouts)).astype(np.float64) / float(len(rollouts))
    return passive_error, weights, binaries

def passive_binary(passive_error, weighting, proximity, done):
    passive_error_cutoff, passive_error_upper, weighting_ratio, weighting_schedule = weighting
    if len(passive_error) < 1000: binaries = copy.deepcopy(passive_error) # in case we want to preserve passive error, but don't copy if this is too large
    else: binaries = passive_error
    binaries[binaries<=passive_error_cutoff] = 0
    binaries[binaries>passive_error_upper] = 0 # if the error is too high, this might be an anomaly
    binaries[binaries>passive_error_cutoff] = 1
    binaries[done == 1] = 0 # if done, disregard

    # use proximity to narrow the range of weights, if proximity is not used, these should be ones TODO: replace with feasibility?
    binaries = (binaries.astype(int) * proximity.astype(int)).astype(np.float128).squeeze()
    return binaries

def proximity_binary(full_model, rollouts,object_rollouts=None, full=False, pall=False):
    # construct proximity batches if necessary
    etype = error_types.PROXIMITY_FULL if full else (error_types.PROXIMITY_ALL if pall else error_types.PROXIMITY)
    proximal = get_error(full_model, rollouts, object_rollouts, error_type=etype).astype(int)
    proximal_inst = get_error(full_model, rollouts, object_rollouts, error_type=etype, reduced=False).astype(int) # the same as above if not multiinstanced
    non_proximal = (proximal != 1).astype(int)
    non_proximal_inst = (proximal_inst != 1).astype(int)
    # non_proximal_weights = non_proximal.squeeze() / np.sum(non_proximal) if np.sum(non_proximal) != 0 else np.ones(non_proximal.shape) / len(non_proximal)
    np_binaries = non_proximal.sum(axis=-1).astype(bool).astype(int)
    p_binaries = proximal.sum(axis=-1).astype(bool).astype(int)
    return np_binaries, p_binaries, non_proximal, proximal

def separate_weights(weighting, full_model, rollouts, proximity, trace=None, object_rollouts=None): # this should work for all cases because passive_likelihood reduces, the only difference is the passive error threshold
    '''
    Generates weights either based on the passive error, the trace, or returns a single vector. This value is shared across the training computation
    '''
    passive_error_cutoff, passive_error_upper, weighting_ratio, weighting_schedule = weighting
    if weighting_ratio >= 0:
        passive_error =  - get_error(full_model, rollouts, object_rollouts, error_type = error_types.PASSIVE_LIKELIHOOD).astype(int)
        done =  np.expand_dims(get_error(full_model, rollouts, object_rollouts, error_type = error_types.DONE).squeeze(), -1)
        # print(object_rollouts.target_diff.shape, done.shape, passive_error.shape)
        print(np.concatenate([passive_error, rollouts.target_diff if object_rollouts is None else object_rollouts.target_diff, done], axis=-1))
        print("passive", np.concatenate([passive_error, rollouts.target_diff if object_rollouts is None else object_rollouts.target_diff, done], axis=-1)[(passive_error > passive_error_cutoff).squeeze()][:100])
        print("passive", np.concatenate([passive_error, rollouts.target_diff if object_rollouts is None else object_rollouts.target_diff, done], axis=-1)[(passive_error > passive_error_cutoff).squeeze()][100:200])
        print(np.concatenate([passive_error, rollouts.target_diff if object_rollouts is None else object_rollouts.target_diff, done], axis=-1)[(passive_error > passive_error_cutoff).squeeze()][200:300])
        # weighting hyperparameters, if passive_error_cutoff > 0 then using passive weighting
        binaries = passive_binary(passive_error, weighting, proximity, done)
        if len(binaries.shape) > 1 and binaries.shape[-1] > 1: 
            binaries = np.sum(binaries, axis=-1)
            binaries[binaries > 1] = 1 
        weights = get_weights(weighting_ratio, binaries)
        if np.sum(binaries) == 0:
            print("NO PASSIVE FOUND")
            passive_error, weights, binaries = uni_weights(rollouts)
        print("assive error", np.sum(binaries), passive_error[passive_error > 0], binaries, weights)
        # error
    elif trace is not None:
        passive_error = trace.copy()
        binaries = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        weights = get_weights(weighting_ratio, binaries)
    else: # no special weighting on the samples
        passive_error, weights, binaries = uni_weights(rollouts)
    print(passive_error.shape, weights.shape, binaries.shape, object_rollouts is not None,
                                # rollouts.weight_binary.shape, len(rollouts.weight_binary.shape), 
                                "weight_binary" in rollouts,
                                  len(binaries.shape) == 1 and ((object_rollouts is not None and len(object_rollouts.weight_binary.shape) == 2) or
                                 ("weight_binary" in rollouts and len(rollouts.weight_binary.shape) == 2)))
    # if len(binaries.shape) == 1 and ((object_rollouts is not None and len(object_rollouts.weight_binary.shape) == 2) or
    #                              ("weight_binary" in rollouts and len(rollouts.weight_binary.shape) == 2)): binaries = np.expand_dims(binaries, -1)
    if len(binaries.shape) == 1 and (("weight_binary" in rollouts and len(rollouts.weight_binary.shape) == 2) or (object_rollouts is not None and "weight_binary" in object_rollouts and len(object_rollouts.weight_binary.shape) == 2)): binaries = np.expand_dims(binaries, -1)
    if object_rollouts is None: rollouts.weight_binary[:len(rollouts)] = binaries
    else: object_rollouts.weight_binary[:len(rollouts)] = (np.sum(binaries, axis=-1) > 0).astype(int) if len(binaries.shape) > 1 else binaries
    return passive_error, weights, binaries


def get_weights(ratio_lambda, binaries):
    # binaries are 0-1 values, either the trace values (supervised interactions)
    # or where the passive error exceeds a threshold, possibly combined with proximal states


    # determine error based binary weights
    weights = binaries.copy()
    num_weighted = binaries.copy().astype(bool).astype(int)

    # passes through if we are using uniform weights
    if ratio_lambda <= 0:
        if np.sum(weights) == 0:
            weights = np.ones(weights.shape)
        weights = (weights.astype(np.float64) / np.sum(weights).astype(np.float64))
        weights[weights < 0] = 0
        if len(weights.shape) == 2: weights = weights[:,0] # squeeze the last dimension
        return weights

    # generate a ratio based on the number of live versus dead
    total_live = np.sum(weights)
    total_dead = np.sum((num_weighted + 1)) - np.sum(num_weighted) * 2
    # print(weights)

    # for a ratio lambda of 1, will get 50-50 probability of sampling a "live" (high passive error) versus "dead" (low passive error)
    live_factor = np.float64(np.round(total_dead / total_live * ratio_lambda))
    # print("live factor", ratio_lambda, np.sum((weights + 1)), total_dead, total_live, live_factor)
    weights = (weights * live_factor) + 1
    weights = (weights.astype(np.float64) / np.sum(weights).astype(np.float64))
    if len(weights.shape) == 2: weights = weights[:,0] # squeeze the last dimension
    return weights