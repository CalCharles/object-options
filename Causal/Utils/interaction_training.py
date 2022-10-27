def evaluate_active_interaction(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity):
    active_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - active_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    # passive_nlikelihood = compute_likelihood(full_model, args.train.batch_size, - passive_log_probs, done_flags=done_flags, reduced=False, is_full = True)
    mask_loss = interaction_mask.sum(-1).mean()
    full_loss = active_nlikelihood + mask_loss * args.full_inter.lasso_lambda
    return full_loss

# def evaluate_active_forward(full_model, args, active_params, interaction_mask, active_log_probs, done_flags, proximity)

