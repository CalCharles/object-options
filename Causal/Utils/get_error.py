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
    "PROXIMITY_FLAT", # gets the flattened proximity between two states given in names
    "PROXIMITY_FULL", # gets the full proximity (all other objects) for one object
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


def compute_error(full_model, error_type, part, obj_part, normalized = False, reduced=True, prenormalize=False, object_names=None):
    # @param part is the segment of rollout data
    # @param normalized asked for normalized outputs and comparisons
    # @param reduced reduces along the final output, combining the features of object state
    # @param prenormalize normalizes the inputs TODO: might not be implemented yet
    # computes the value for 
    rv = lambda x: full_model.norm.reverse(x, form="dyn" if full_model.predict_dynamics else "target", name=full_model.name) # self.output_normalization_function.reverse
    rv_var = lambda x: full_model.norm.reverse(x, form="dyn" if full_model.predict_dynamics else "diff", name=full_model.name) # self.output_normalization_function.reverse
    nf = lambda x: full_model.norm(x, form = "dyn" if full_model.predict_dynamics else "target", name=full_model.name) # self.output_normalization_function
    num_batch = len(part)

    # if the part is not normalized, this should normalize it
    is_full = obj_part is not None
    if prenormalize: 
        part = full_model.normalize_batch(copy.deepcopy(part))
        if is_full: obj_part = full_model.normalize_batch(copy.deepcopy(obj_part))
    
    # if is_full then the obj_part contains object specific components
    if is_full:
        obj_part.target = obj_part.obs
        obj_part.next_target = obj_part.obs_next
        obj_part.inter_state = part.obs
        obj_part.tarinter_state = np.concatenate([obj_part.target, obj_part.inter_state], axis=-1)
    use_part = obj_part if is_full else part

    # handles 3 different targets, predicting the trace, predicting the dynamics, or predicting the next target
    if error_type == error_types.INTERACTION: target = use_part.trace
    elif full_model.predict_dynamics: target = use_part.target_diff
    else: target = use_part.next_target

    
    if error_type == error_types.PASSIVE or error_type == error_types.PASSIVE_RAW:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(use_part.target, cuda=full_model.iscuda))[0])
    elif error_type == error_types.ACTIVE or error_type == error_types.ACTIVE_RAW:
        if not is_full: output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(use_part.inter_state, cuda=full_model.iscuda))[0])
        else: output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(use_part.tarinter_state, cuda=full_model.iscuda), pytorch_model.wrap(use_part.inter, cuda=full_model.iscuda))[0])
    if error_type == error_types.PASSIVE_VAR:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(use_part.target, cuda=full_model.iscuda))[1])
    elif error_type == error_types.ACTIVE_VAR:
        if not is_full: output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(use_part.inter_state, cuda=full_model.iscuda))[1])
        else: output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(use_part.tarinter_state, cuda=full_model.iscuda), pytorch_model.wrap(use_part.inter, cuda=full_model.iscuda))[1])
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
        output = pytorch_model.unwrap(full_model.weighted_likelihoods(use_part)[-1])
    elif error_type == error_types.PASSIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.passive_likelihoods(use_part)[-1])
        print(output, use_part.obs)
    elif error_type == error_types.ACTIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.active_likelihoods(use_part)[-1])
    if error_type <= error_types.ACTIVE_LIKELIHOOD:
        if reduced: output = - compute_likelihood(full_model, num_batch, - output, is_full=is_full)
        return output

    # interaction type errors
    if error_type == error_types.INTERACTION:
        output = full_model.interaction(use_part)
        return np.abs(pytorch_model.unwrap(output)-target) # should probably use CE loss
    elif error_type == error_types.INTERACTION_RAW:
        return pytorch_model.unwrap(full_model.interaction(use_part))
    elif error_type == error_types.INTERACTION_BINARIES:
        _, _, _, _, _, _, active_log_probs, passive_log_probs = full_model.likelihoods(part)
        binaries = pytorch_model.unwrap(full_model.test.compute_binary(- active_log_probs.sum(dim=-1),
                                                - passive_log_probs.sum(dim=-1)).unsqueeze(-1))
        return binaries        

    if error_type == error_types.PROXIMITY:
        return check_proximity(full_model, part.parent_state, part.target, normalized=normalized)
    if error_type == error_types.PROXIMITY_FLAT:
        factored_state = full_model.unflatten(part.obs)
        part.parent_state, part.target = full_model.get_object[object_names.parent], full_model.get_object[object_names.target]
        return check_proximity(full_model, part.parent_state, part.target)
    if error_type == error_types.PROXIMITY_FULL: # only works with a padding extractor
        full_state = full_model.norm.reverse(part.obs, form="inter")
        target_state = full_model.norm.reverse(obj_part.target, name=full_model.name)
        return get_full_proximity(full_model, full_state, target_state, normalized=normalized)

    if error_type == error_types.TRACE: return use_part.trace
    if error_type == error_types.DONE: return part.done
    raise Error("invalid error type")

def get_full_proximity(full_model, flattened_state, target_state, normalized=False):
    # print(flattened_state.reshape(flattened_state.shape[:-1] + (flattened_state.shape[-1] // full_model.pad_size, full_model.pad_size)).shape)
    if len(flattened_state.shape) == 1: # no batches
        flattened_state = flattened_state.reshape(flattened_state.shape[:-1] + (flattened_state.shape[-1] // full_model.pad_size, full_model.pad_size))
        dists = np.linalg.norm(flattened_state[...,:full_model.pos_size] - target_state[...,:full_model.pos_size], axis=-1)
    else: 
        flattened_state = flattened_state.reshape(flattened_state.shape[:-1] + (flattened_state.shape[-1] // full_model.pad_size, full_model.pad_size)).transpose(1,0,2)
        dists = np.linalg.norm(flattened_state[...,:full_model.pos_size] - target_state[...,:full_model.pos_size], axis=-1).transpose(1,0)
    proximity = dists < full_model.proximity_epsilon
    # print(dists[0], proximity[0], proximity.shape)
    return proximity

def get_error(full_model, rollouts, object_rollout=None, error_type=0, reduced=True, normalized=False, prenormalize = False, object_names=None):
    # computes some term over the entire rollout, iterates through batches of 500 to avoid overloading the GPU

    # gets all the data from rollouts, in the order of the data (for assignment)
    if type(rollouts) == Batch:
        batch = rollouts
        obj_batch = object_rollout
    else:
        batch, indices = rollouts.sample(0) if len(rollouts) != rollouts.maxsize else (rollouts, np.arange(rollouts.maxsize))
        obj_batch = None if object_rollout is None else (object_rollout.sample(0)[0] if len(object_rollout) != object_rollout.maxsize else object_rollout)

    model_error = []
    for i in range(int(np.ceil(len(batch) / min(500,len(batch))))): # run 500 at a time, so that we don't overload the GPU
        part = batch[i*500:(i+1)*500]
        obj_part = None if obj_batch is None else object_rollout[i*500:(i+1)*500]
        done_flags = np.expand_dims((1-part.done).squeeze(), -1)
        values = compute_error(full_model, error_type, part, obj_part, normalized=normalized, reduced=reduced, prenormalize=prenormalize, object_names = object_names)
        if not outputs(error_type): values = values * done_flags
        # print("done flags", error_names[error_type], done_flags.shape, values.shape, compute_error(full_model, error_type, part, obj_part, normalized=normalized, reduced=reduced, prenormalize=prenormalize, object_names = object_names).shape)
        model_error.append(values)
    return np.concatenate(model_error, axis=0)
