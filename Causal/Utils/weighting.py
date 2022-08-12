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


def separate_weights(weighting, full_model, rollouts, proximity, trace):
    passive_error_cutoff, passive_error_upper, weighting_ratio, weighting_schedule = weighting
    if weighting_ratio >= 0:
        passive_error =  - get_error(full_model, rollouts, error_type = error_types.PASSIVE_LIKELIHOOD)
        done =  get_error(full_model, rollouts, error_type = error_types.DONE)
        print(np.concatenate([passive_error, rollouts.target_diff, done], axis=-1)[(passive_error > 0).squeeze()][:100])
        print(np.concatenate([passive_error, rollouts.target_diff, done], axis=-1)[(passive_error > 0).squeeze()][100:200])
        print(np.concatenate([passive_error, rollouts.target_diff, done], axis=-1)[(passive_error > 0).squeeze()][200:300])
        # weighting hyperparameters, if passive_error_cutoff > 0 then using passive weighting
        binaries = passive_binary(passive_error, weighting, proximity, done)
        weights = get_weights(weighting_ratio, binaries)
    elif trace is not None:
        passive_error = trace.copy()
        binaries = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        weights = get_weights(weighting_ratio, binaries)
    else: # no special weighting on the samples
        passive_error, weights, binaries = uni_weights(rollouts)
    rollouts.weight_binary[:len(rollouts)] = binaries
    return passive_error, weights, binaries


def get_weights(ratio_lambda, binaries):
    # binaries are 0-1 values, either the trace values (supervised interactions)
    # or where the passive error exceeds a threshold, possibly combined with proximal states


    # determine error based binary weights
    weights = binaries.copy()

    # passes through if we are using uniform weights
    if ratio_lambda < 0:
        return (weights.astype(np.float64) / np.sum(weights).astype(np.float64))

    # generate a ratio based on the number of live versus dead
    total_live = np.sum(weights)
    total_dead = np.sum((weights + 1)) - np.sum(weights) * 2

    # for a ratio lambda of 1, will get 50-50 probability of sampling a "live" (high passive error) versus "dead" (low passive error)
    live_factor = np.float128(np.round(total_dead / total_live * ratio_lambda))
    weights = (weights * live_factor) + 1
    weights = (weights.astype(np.float64) / np.sum(weights).astype(np.float64))
    return weights