import numpy as np
from State.object_dict import ObjDict
from Network.network_utils import pytorch_model

error_types = ObjDict() # an enumerator for different error types
error_types.PASSIVE = 0 # mean of passive
error_types.ACTIVE = 1 # mean of active
error_types.PASSIVE_VAR = 2 # variance of passive
error_types.ACTIVE_VAR = 3 # variance of active
error_types.LIKELIHOOD = 4 # weighted likelihood, multiplying active output with the interaction
error_types.PASSIVE_LIKELIHOOD = 5
error_types.ACTIVE_LIKELIHOOD = 6
error_types.LIKELIHOOD_FULL = 7
error_types.PASSIVE_LIKELIHOOD_FULL = 8
error_types.ACTIVE_LIKELIHOOD_FULL = 9
error_types.INTERACTION = 10
error_types.INTERACTION_BINARIES = 11
error_types.PASSIVE_RAW = 12 # mean of passive, unnormalized
error_types.ACTIVE_RAW = 13 # mean of active, unnormalized
error_types.PROXIMITY = 14 # measures if two objects are close together

outputs = lambda x: x > error_types.ACTIVE_LIKELIHOOD 

def compute_error(full_model, error_type, part, normalized = False):
    # computes the value for 
    rv = lambda x: full_model.norm.reverse(x, form="dyn" if full_model.predict_dynamics else "target") # self.output_normalization_function.reverse
    nf = lambda x: full_model.norm(x, form = "dyn" if full_model.predict_dynamics else "target") # self.output_normalization_function
    num_batch = len(part)

    # handles 3 different targets, predicting the trace, predicting the dynamics, or predicting the next target
    if error_type == error_types.INTERACTION: target = part.trace
    elif full_model.predict_dynamics: target = part.target_diff
    else: target = part.next_target
    
    if error_type == error_types.PASSIVE or error_type == error_types.PASSIVE_RAW:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(part.target, cuda=full_model.iscuda))[0])
    elif error_type == error_types.ACTIVE or error_type == error_types.ACTIVE_RAW:
        output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(part.inter_state, cuda=full_model.iscuda))[0])
    if error_type == error_types.PASSIVE_VAR or error_type == error_types.PASSIVE_RAW:
        output = pytorch_model.unwrap(full_model.passive_model(pytorch_model.wrap(part.target, cuda=full_model.iscuda))[1])
    elif error_type == error_types.ACTIVE_VAR or error_type == error_types.ACTIVE_RAW:
        output = pytorch_model.unwrap(full_model.active_model(pytorch_model.wrap(part.inter_state, cuda=full_model.iscuda))[1])
    if error_type <= error_types.ACTIVE_VAR:
        if not normalized: # this can only be the case for PASSIVE and ACTIVE
            return np.expand_dims(np.linalg.norm(rv(pytorch_model.unwrap(output)) - rv(target), ord=1), axis=-1)
        return np.expand_dims(np.linalg.norm(pytorch_model.unwrap(output) - target, ord=1), axis=-1)

    # likelihood type error computation
    if error_type == error_types.LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.weighted_likelihoods(part)[-1].sum(dim=-1))
    elif error_type == error_types.PASSIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.passive_likelihoods(part)[-1].sum(dim=-1))
    elif error_type == error_types.ACTIVE_LIKELIHOOD:
        output = pytorch_model.unwrap(full_model.active_likelihoods(part)[-1].sum(dim=-1))
    elif error_type == error_types.LIKELIHOOD_FULL:
        output = pytorch_model.unwrap(full_model.weighted_likelihoods(part)[-1])
    elif error_type == error_types.PASSIVE_LIKELIHOOD_FULL:
        output = pytorch_model.unwrap(full_model.passive_likelihoods(part)[-1])
    elif error_type == error_types.ACTIVE_LIKELIHOOD_FULL:
        output = pytorch_model.unwrap(full_model.active_likelihoods(part)[-1])
    if error_type <= error_types.ACTIVE_LIKELIHOOD_FULL:
        return pytorch_model.unwrap(output)

    # interaction type errors
    if error_type == error_types.INTERACTION:
        output = full_model.interaction(part)
        return np.abs(pytorch_model.unwrap(output)-target) # should probably use CE loss
    elif error_type == error_types.INTERACTION_BINARIES:
        _, _, _, _, _, _, active_log_probs, passive_log_probs = full_model.likelihoods(part)
        binaries = full_model.testing_module.compute_binary(pytorch_model.unwrap(active_log_probs.sum(dim=-1)),
                                                pytorch_model.unwrap(passive_log_probs.sum(dim=-1)))
        return binaries

    if error_type <= error_types.ACTIVE_RAW:
        return pytorch_model.unwrap(torch.stack(output))

    if error_type == error_types.PROXIMITY:
        parent_pos = np.where(full_model.position_masks[full_model.names.primary_parent])[0]
        target_pos = np.where(full_model.position_masks[full_model.names.target])[0]
        parent = part.parent_state[...,parent_pos]
        print(target_pos)
        target = full_model.norm.reverse(part.target[...,target_pos], idxes=target_pos)
        return np.linalg.norm(parent-target, ord=1) < full_model.proximity_epsilon if full_model.proximity_epsilon > 0 else np.ones(num_batch).astype(bool) # returns binarized differences
    raise Error("invalid error type")


def get_error(full_model, rollouts, error_type=0):
    # computes some term over the entire rollout, iterates through batches of 500 to avoid overloading the GPU

    # gets all the data from rollouts, in the order of the data (for assignment)
    batch, indices = rollouts.sample(0) if len(rollouts) != rollouts.maxsize else (rollouts, np.arange(rollouts.maxsize))
    done_flags = 1-batch.done

    model_error = []
    for i in range(int(np.ceil(len(batch) / min(500,len(batch))))): # run 500 at a time, so that we don't overload the GPU
        part = batch[i*500:(i+1)*500]
        values = compute_error(full_model, error_type, part, normalized=False) * done_flags[i*500:(i+1)*500] if not outputs(error_type) else compute_error(full_model, error_type, part, normalized=False)
        model_error.append(values)
    return np.concatenate(model_error, axis=0)
