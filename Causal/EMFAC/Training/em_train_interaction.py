from Causal.FullInteraction.Training.full_train_trace import train_binaries
from Causal.Utils.weighting import get_weights
import copy

def mask_train_interaction(full_model, args, interaction_masks, soft_masks, rollouts, object_rollouts, test_rollout, test_object_rollout, interaction_optimizer, inter_logger):
    '''
    trains the interaction model according to the last k largest masks
    @return the interaction component of the full model is updated
    '''
    args = copy.deepcopy(args)
    niters, args.train.num_iters = args.train.num_iters, args.EMFAC.M_step_iters
    binaries = object_rollouts.sample(0)[0].weight_binary if object_rollouts is not None else rollouts.sample(0)[0].weight_binary
    weights = get_weights(ratio_lambda=args.inter.active.weighting[2], binaries=binaries)
    train_binaries(full_model, rollouts, object_rollouts, args, interaction_optimizer, interaction_masks, inter_logger, weights)        
    args.train.num_iters = niters