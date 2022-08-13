import numpy as np
from State.object_dict import ObjDict
from Network.network_utils import pytorch_model
import copy
from Causal.Utils.instance_handling import compute_likelihood, split_instances
from State.feature_selector import broadcast
from tianshou.data import Batch

error_names = [# an enumerator for different error types
    "PASSIVE",# mean of passive subtracted with target,
    "ACTIVE",# mean of active subtracted with target
    "PASSIVE_VAR",# variance of passive,
    "ACTIVE_VAR",# variance of active
    "PASSIVE_RAW",# mean of passive values,
    "ACTIVE_RAW",# mean of active values,
    "LIKELIHOOD",# weighted likelihood, multiplying active output with the interaction
    "PASSIVE_LIKELIHOOD", # likelihood under the passive model
    "ACTIVE_LIKELIHOOD", # likelihood of data under the active model
    "INTERACTION", #interaction values
    "INTERACTION_RAW",
    "INTERACTION_BINARIES",
    "PROXIMITY",# measures if two objects are close together
    "TRACE",# just sends back the trace values
    "DONE", # just sends back the done values
]

error_types = ObjDict({error_names[i]: i for i in range(len(error_names))})

outputs = lambda x: x > error_types.ACTIVE_LIKELIHOOD 

def check_proximity(full_model, parent_state, target, normalized=False):
    num_batch = parent_state.shape[0] if len(parent_state.shape) > 1 else 1
    parent_pos = np.where(full_model.position_masks[full_model.names.primary_parent])[0]
    target_pos = np.where(full_model.position_masks[full_model.names.target])[0]
    parent = parent_state[...,parent_pos] # TODO: assumes parent state is unnormalized
    if full_model.multi_instanced: # assumes a single parent
        target = full_model.norm.reverse(target) if normalized else target# norms first because norm expects a certain shape
        target = split_instances(target, full_model.obj_dim)[...,target_pos]
        parent = broadcast(parent, target.shape[1], cat=False).transpose(1,0,2) if len(target.shape) == 3 else broadcast(parent, target.shape[0], cat=False) # broadcast parent for every child
        diff = np.linalg.norm(parent-target, ord=1, axis=-1)
        if reduced: return np.expand_dims(np.min(diff, axis=-1) < full_model.proximity_epsilon, -1) if full_model.proximity_epsilon > 0 else np.ones((num_batch, 1)).astype(bool)
        else: return diff < full_model.proximity_epsilon if full_model.proximity_epsilon > 0 else np.ones((num_batch, target.shape[1])).astype(bool)
    else: target = target[...,target_pos]
    target = full_model.norm.reverse(target, idxes=target_pos) if normalized else target
    return np.expand_dims(np.linalg.norm(parent-target, ord=1, axis=-1) < full_model.proximity_epsilon, -1) if full_model.proximity_epsilon > 0 else np.ones((num_batch, 1)).astype(bool) # returns binarized differences


def compute_error(full_model, error_type, part, normalized = False, reduced=True, prenormalize=False):
    # @param part is the segment of rollout data
    # @param normalized asked for normalized outputs and comparisons
    # @param reduced reduces along the final output, combining the features of object state
    # @param prenormalize normalizes the inputs TODO: might not be implemented yet
    # computes the value for 
    rv = lambda x: full_model.norm.reverse(x, form="dyn" if full_model.predict_dynamics else "target") # self.output_normalization_function.reverse
    rv_var = lambda x: full_model.norm.reverse(x, form="dyn" if full_model.predict_dynamics else "diff") # self.output_normalization_function.reverse
    nf = lambda x: full_model.norm(x, form = "dyn" if full_model.predict_dynamics else "target") # self.output_normalization_function
    num_batch = len(part)

    # if the part is not normalized, this should normalize it
    if prenormalize: part = full_model.normalize_batch(copy.deepcopy(part))
    
    # handles 3 different targets, predicting the trace, predicting the dynamics, or predicting the next target
    if error_type == error_types.INTERACTION: target = part.trace
    elif full_model.predict_dynamics: target = part.target_diff
    else: target = part.next_target

    
    if error_type == error_types.PASSIVE or error_type == error_types.PASSIVE_RAW:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(part.target, cuda=full_model.iscuda))[0])
    elif error_type == error_types.ACTIVE or error_type == error_types.ACTIVE_RAW:
        output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(part.inter_state, cuda=full_model.iscuda))[0])
    if error_type == error_types.PASSIVE_VAR:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(part.target, cuda=full_model.iscuda))[1])
    elif error_type == error_types.ACTIVE_VAR:
        output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(part.inter_state, cuda=full_model.iscuda))[1])
    if error_type <= error_types.ACTIVE:
        if reduced:
            if not normalized: # this can only be the case for PASSIVE and ACTIVE
                return np.expand_dims(np.linalg.norm(rv(pytorch_model.unwrap(output)) - rv(target), ord=1, axis=-1), axis=-1)
            return np.expand_dims(np.linalg.norm(pytorch_model.unwrap(output) - target, ord=1, axis=-1), axis=-1)
        else:
            if not normalized: return np.abs(rv(pytorch_model.unwrap(output)) - rv(target))
            return np.abs(pytorch_model.unwrap(output) - target)
    if error_type <= error_types.ACTIVE_VAR:
        if not normalized: return rv_var(np.stack(pytorch_model.unwrap(output)))
    if error_type <= error_types.ACTIVE_RAW:
        if not normalized: return rv(np.stack(pytorch_model.unwrap(output)))
        return np.stack(pytorch_model.unwrap(output))

    # likelihood type error computation
    if error_type == error_types.LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.weighted_likelihoods(part)[-1])
    elif error_type == error_types.PASSIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.passive_likelihoods(part)[-1])
    elif error_type == error_types.ACTIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.active_likelihoods(part)[-1])
    if error_type <= error_types.ACTIVE_LIKELIHOOD:
        if reduced: output = - compute_likelihood(full_model, num_batch, - output)
        return output

    # interaction type errors
    if error_type == error_types.INTERACTION:
        output = full_model.interaction(part)
        return np.abs(pytorch_model.unwrap(output)-target) # should probably use CE loss
    elif error_type == error_types.INTERACTION_RAW:
        return pytorch_model.unwrap(full_model.interaction(part))
    elif error_type == error_types.INTERACTION_BINARIES:
        _, _, _, _, _, _, active_log_probs, passive_log_probs = full_model.likelihoods(part)
        binaries = pytorch_model.unwrap(full_model.test.compute_binary(- active_log_probs.sum(dim=-1),
                                                - passive_log_probs.sum(dim=-1)).unsqueeze(-1))
        return binaries

    if error_type == error_types.PROXIMITY:
        return check_proximity(full_model, part.parent_state, part.target)
    
    if error_type == error_types.TRACE: return part.trace
    if error_type == error_types.DONE: return part.done
    raise Error("invalid error type")


def get_error(full_model, rollouts, error_type=0, reduced=True, normalized=False, prenormalize = False):
    # computes some term over the entire rollout, iterates through batches of 500 to avoid overloading the GPU

    # gets all the data from rollouts, in the order of the data (for assignment)
    if type(rollouts) == Batch:
        batch = rollouts
    else:
        batch, indices = rollouts.sample(0) if len(rollouts) != rollouts.maxsize else (rollouts, np.arange(rollouts.maxsize))

    model_error = []
    for i in range(int(np.ceil(len(batch) / min(500,len(batch))))): # run 500 at a time, so that we don't overload the GPU
        part = batch[i*500:(i+1)*500]
        done_flags = 1-part.done
        values = compute_error(full_model, error_type, part, normalized=normalized, reduced=reduced, prenormalize=prenormalize) * done_flags if not outputs(error_type) else compute_error(full_model, error_type, part, normalized=normalized, reduced=reduced)
        model_error.append(values)
    return np.concatenate(model_error, axis=0)
