import numpy as np
from Causal.EMFAC.mask_max import mask_assign, mask_most_common, mask_likelihood_probabilities, generate_masks
from Causal.EMFAC.Training.em_train_interaction import mask_train_interaction
from Causal.EMFAC.Training.em_train_forward import train_forward_masked
from Causal.EMFAC.Training.em_train_combined import train_forward_combined
from Causal.Training.loggers.forward_logger import forward_logger
from Causal.Training.loggers.interaction_logger import interaction_logger
import torch.optim as optim

def train_EMFAC(full_model, args, rollout, object_rollout, test_rollout, test_object_rollout):
    forward_loggers = [forward_logger("forward_" + full_model.name + str(i), args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename) for i in range(args.EMFAC.num_masks)]
    inter_logger = interaction_logger("interaction_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename)
    full_logger = forward_logger("full_" + full_model.name, args.record.record_graphs, args.inter.active.active_log_interval, full_model, filename=args.record.log_filename)
    interaction_optimizer = optim.Adam(full_model.interaction_model.parameters(), args.interaction_net.optimizer.alt_lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
    full_optimizer = optim.Adam(full_model.parameters(), args.interaction_net.optimizer.lr, eps=args.interaction_net.optimizer.eps, betas=args.interaction_net.optimizer.betas, weight_decay=args.interaction_net.optimizer.weight_decay)
    for i in range(args.train.num_iters):
        train_EMFAC_step(full_model, args, rollout, object_rollout, test_rollout, test_object_rollout, forward_loggers, inter_logger, full_logger, interaction_optimizer, full_optimizer, i)

def train_EMFAC_step(full_model, args, rollout, object_rollout, test_rollout, test_object_rollout, forward_loggers, inter_logger, full_logger, interaction_optimizer, full_optimizer, i):
    hard_masks, soft_masks = mask_assign(full_model, args, rollout, object_rollout) # gets he interaction output for each value
    k_masks = mask_most_common(hard_masks, full_model, args, rollout, object_rollout) # gets the k most common masks TODO: get the k most common mask relational functions, allow subset masks
    model_performances = list()
    for model_index, mask in enumerate(k_masks): # TODO: multithreading on the forward model training
        mperf = train_mask_forward(full_model, model_index, args, mask, soft_masks, rollout, object_rollout, test_rollout, test_object_rollout, forward_loggers[model_index], i)
        model_performances.append(mperf)
    model_performances = np.concatenate(model_performances, axis=-1)
    interaction_masks = generate_masks(args, k_masks, model_performances)
    mask_train_interaction(full_model, args, interaction_masks, soft_masks, rollout, object_rollout, test_rollout, test_object_rollout, interaction_optimizer, inter_logger)
    train_forward_combined(full_model, args, rollout, object_rollout, test_rollout, test_object_rollout, full_optimizer, full_logger, i)


def train_mask_forward(full_model, model_index, args, mask, soft_masks, rollouts, object_rollouts, test_rollouts, test_object_rollouts, forward_logger, i):
    weights = mask_likelihood_probabilities(mask, soft_masks) # gets probability of data point under mask assignment and soft mask, TODO: don't penalize subset masks
    model_performance = train_forward_masked(full_model, model_index, args, mask, rollouts, object_rollouts, test_rollouts, test_object_rollouts, weights, forward_logger, i)
    return model_performance