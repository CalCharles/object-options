

error_types = ObjDict() # an enumerator for different error types
error_types.PASSIVE = 0 # mean of passive
error_types.ACTIVE = 1 # mean of active
error_types.LIKELIHOOD = 2 # weighted likelihood, multiplying active output with the interaction
error_types.PASSIVE_LIKELIHOOD = 3
error_types.ACTIVE_LIKELIHOOD = 4
error_types.INTERACTION = 5
error_types.INTERACTION_BINARIES = 6

def compute_error(full_model, error_type, part, normalized = False):
    # computes the value for 
    rv = full_model.norm.reverse # self.output_normalization_function.reverse
    nf = full_model.norm # self.output_normalization_function
    # print((rv(mean) - target).abs().sum(dim=1).shape)
    num_batch = mean.shape[0]

    # handles 3 different targets, predicting the trace, predicting the dynamics, or predicting the next target
    if not predict_target: target_state = batch.trace
    elif full_model.predict_dynamics: target_state = batch.target_diff
    else: target_state = batch.next_target
    
    if error_type == error_types.PASSIVE:
        output = full_model.passive_model(pytorch_model.wrap(part.target, cuda=full_model.iscuda))[0]
    elif error_type == error_types.ACTIVE:
        output = full_model.active_model(pytorch_model.wrap(part.inter_state, cuda=full_model.iscuda))[0]
    if error_type < error_type.ACTIVE:
        if not normalized: # this can only be the case for PASSIVE and ACTIVE
            return np.expand_dims(np.linalg.norm(rv(pytorch_model.unwrap(output)) - rv(target_state), ord=1), axis=-1)
        return np.expand_dims(np.linalg.norm(pytorch_model.unwrap(output) - target_state, ord=1), axis=-1)

    # likelihood type error computation
    if error_type == error_types.LIKELIHOOD:
        output = full_model.weighted_likelihoods(part)[-1].sum(dim=-1)
    elif error_type == error_types.PASSIVE_LIKELIHOOD:
        output = full_model.passive_likelihoods(part)[-1].sum(dim=-1)
    elif error_type == error_types.ACTIVE_LIKELIHOOD:
        output = full_model.active_likelihoods(part)[-1].sum(dim=-1)
    if error_type < error_type.ACTIVE_LIKELIHOOD:
        return pytorch_model.unwrap(output)

    # interaction type errors
    if error_type == error_types.INTERACTION:
        output = full_model.interaction(part)
        return np.abs(pytorch_model.unwrap(output)-target_state) # should probably use CE loss
    elif error_type == error_types.INTERACTION_BINARIES:
        _, _, _, _, _, _, active_log_probs, passive_log_probs = full_model.likelihoods(part)
        binaries = full_model.testing_module.compute_binary(pytorch_model.unwrap(active_log_probs.sum(dim=-1)),
                                                pytorch_model.unwrap(passive_log_probs.sum(dim=-1)))
        return binaries
    raise Error("invalid error type")


def get_error(full_model, rollouts, error_type=0, predict_target=False):
    # computes some term over the entire rollout, iterates through batches of 500 to avoid overloading the GPU

    # gets all the data from rollouts, in the order of the data (for assignment)
    batch = rollouts.sample(0) if len(rollouts) != rollouts.maxsize else rollouts
    done_flags = 1-batch.done

    model_error = []
    for i in range(len(batch) / 500): # run 500 at a time, so that we don't overload the GPU
        part = batch[i*500:(i+1)*500]
        model_error.append(compute_error(full_model, error_type, part, normalized=False) * done_flags[i*500:(i+1)*500])
    return np.concatenate(model_error, axis=0)
