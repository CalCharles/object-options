import numpy as np
from Causal.Utils.get_error import get_error, error_types

def mask_assign(full_model, args, rollouts, object_rollouts):
    '''
    Assigns each data point with a mask by evaluating the interaction model to get both the soft (probability)
    and hard (probabilistically assigned binary) masks
    @param rollouts, object_rollouts: N states with n factors
    @param full_model model with forward and interaction components
    @param args a namespace of args
    @return N x n binary vector, N x n [0,1] vector
    '''
    soft_masks = get_error(full_model, rollouts, object_rollouts, error_type=error_types.INTERACTION_RAW)
    hard_masks = full_model.apply_mask(soft_masks, soft=False)
    return hard_masks, soft_masks

def mask_most_common(hard_masks, full_model, args, rollout, object_rollout):
    '''
    Gets the args.EMFAC.num_masks most common masks based on the hard mask assignment.
    TODO: return the most common mask functions (instead of true masks)
    TODO: incorporate likelihoods from the soft masks
    @param hard_masks Nxn binary vector 
    '''
    values, counts = np.unique(hard_masks, return_counts=True, axis=0)
    print(values, counts)
    vidx = np.argpartition(-counts, kth=args.EMFAC.num_masks)[:args.EMFAC.num_masks]
    return values[vidx]

def mask_likelihood_probabilities(mask, soft_masks):
    '''
    gets probability of data point under mask assignment and soft mask
    TODO: don't penalize subset masks

    @return probabilities of all the data points by normalization
    '''
    WEIGHT_LAMBDA = 0.4 # deemphasizes the magnitude of difference by e^WEIGHT_LAMBDA
    likelihoods = np.sum(np.log(mask * soft_masks + 1e-6), axis=-1)
    weights = (likelihoods - np.min(likelihoods) + WEIGHT_LAMBDA) / np.sum(likelihoods - np.min(likelihoods) + WEIGHT_LAMBDA )
    return weights

def generate_masks(args, k_masks, model_performances):
    '''
    assigns each data point in the training set with a mask, based on the
    argmax model performance and the mask cost
    TODO: generate masks using a confidence strategy
    '''
    print(k_masks, np.sum(k_masks * args.EMFAC.binary_cost, axis=-1), k_masks.shape, model_performances[:10])
    mask_cost_performances = np.sum(k_masks * args.EMFAC.binary_cost, axis=-1) + model_performances
    mask_choices = np.argmin(mask_cost_performances, axis=-1)
    print("mp", mask_cost_performances[np.arange(len(model_performances)), mask_choices][:10], mask_choices[:10] )
    # confidence computed by the difference between the best mask and the one chosen
    # assumes that the best mask is from the full mask
    mask_confidences = np.max(model_performances, axis=-1) - model_performances[:, mask_choices]

    return k_masks[mask_choices]