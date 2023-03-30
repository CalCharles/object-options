def evaluate_active_interaction(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    mask_loss = interaction_mask.sum(-1).mean()
    full_loss = active_nlikelihood + mask_loss * args.full_inter.lasso_lambda
    return full_loss

# def evaluate_active_forward(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity)

def input_gradients(full_model, args, rollouts, object_rollout, normalize=False):
    # computes the log gradient magnitudes of the interaction mask, the active inputs and the full inputs
    input_gradients = list()
    for i in range(int(np.ceil(len(rollouts) / 512 ))):
        full_batch = rollouts[i*512:(i+1)*512]
        batch = object_rollout[i*512:(i+1)*512]
        idxes = np.linspace(i*512,min((i+1)*512, len(rollouts)))
        batch.tarinter_state = np.concatenate([batch.obs, full_batch.obs], axis=-1)
        batch.inter_state = full_batch.obs

        # run get the full active outputs (passive for interaction binaries)
        active_inputs, active_params, active_dist, active_log_probs = full_model.active_open_likelihoods(batch, normalize=normalize)
        # done flags
        done_flags = pytorch_model.wrap(1-full_batch.done, cuda = full_model.iscuda).squeeze().unsqueeze(-1)
        
        # full loss
        active_likelihood_full, active_prediction_params = - active_log_probs, active_params
        active_loss = compute_likelihood(full_model, args.train.batch_size, active_likelihood_full, done_flags=done_flags, is_full = True)
        grad_variables = get_gradient(full_model, active_loss, grad_variables=[active_inputs])

        # print(interaction_likelihood, active_soft_inputs, grad_variables[1])
        query_grad_variables = grad_variables[0][...,batch.obs.shape[-1]:]
        key_grad_variables = grad_variables[0][...,:batch.obs.shape[-1]]
        # print(grad_variables[1].shape, active_soft_inputs[...,batch.obs.shape[-1]:].shape, active_soft_inputs.shape, batch.obs.shape)
        log_magnitude = np.log(np.abs(query_grad_variables))
        query_sum_value = np.sum(log_magnitude.reshape(512, -1, full_model.obj_dim), axis=-1) # breaks down into batch x num_queries x query_dim then sums along query_dim
        # print(active_soft_inputs.shape, grad_variables[1].shape)
        print("grad", np.concatenate((batch.trace, 
                        query_sum_value), axis=-1)[:10])
                        # , axis=-1)[:3], grad_variable_revalue[1].reshape(512, interaction_likelihood.shape[-1], -1)[:3])
        input_gradients.append(query_sum_value)
        # print("grad il variance", interaction_likelihood)
        # print("grad as variance", np.sum(grad_variable_statistics[1].reshape(interaction_likelihood.shape[1], -1), axis=-1))
        # return idxes, interaction_loss, interaction_likelihood, hard_interaction_mask, weight_count, done_flags, grad_variables
    return np.concatenate(input_gradients, axis=-1)
