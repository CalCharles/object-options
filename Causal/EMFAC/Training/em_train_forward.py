import copy
from Causal.EMFAC.Training.train_forward_log_likelihood import train_forward_log_likelihood

def train_forward_masked(full_model, model_index, args, mask, rollouts, object_rollouts, test_rollouts, test_object_rollouts, weights, logger, i):
    '''
    trains a forward model for args.EMFAC.E_step_iters using the given mask and weights
    @return The performance of the final learned model on the weighted dataset
    '''
    args = copy.deepcopy(args)
    args.train.num_iters = args.EMFAC.E_step_iters
    active_optimizer = full_model.active_model.reset_index(model_index, args.active_net.optimizer)
    loss = train_forward_log_likelihood(full_model, args, 
                                rollouts, object_rollouts, test_rollouts, test_object_rollouts, 
                                weights, active_optimizer, logger, i, given_mask=mask)
    return loss
