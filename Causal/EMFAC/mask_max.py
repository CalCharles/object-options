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
    print(values)
    vidx = np.argpartition(-counts, kth=args.EMFAC.num_masks)[:args.EMFAC.num_masks]
    print("values, counts", values[vidx], counts[vidx])
    return values[vidx]

def mask_likelihood_probabilities(masks, soft_masks, passive_mask, model_mask_weights, model_likelihoods=None):
    '''
    gets probability of data point under mask assignment and soft mask
    TODO: don't penalize subset masks

    @return probabilities of all the data points by normalization
    '''
    idxes = np.random.randint(len(soft_masks), size = (10,))
    mask_likelihoods = list()
    for i, mask in enumerate(masks):
        likelihoods = np.mean(np.log(mask * soft_masks + ((1-mask) * (1-soft_masks)) + 1e-6), axis=-1) # log probability of datapoint, averaged per element
        if model_likelihoods is not None: 
            # print(model_mask_weights[0], likelihoods.shape, model_likelihoods.shape)
            likelihoods = likelihoods + model_likelihoods[:, i] * model_mask_weights[0]
        # print(mask, soft_masks[idxes], likelihoods[idxes])
        likelihoods = likelihoods + model_mask_weights[1] * np.sum(np.abs(mask - passive_mask))
        mask_likelihoods.append(likelihoods)
    mask_likelihoods = np.stack(mask_likelihoods, axis=1)
    # print(likelihoods, mask, soft_masks) 
    weight_sets = list()
    REG_VAL, CLIP_VAL = 3, -10 # alters the rate of the values
    min_likelihood = np.min(mask_likelihoods) # compute relative to the least likely mask
    for i in range(mask_likelihoods.shape[-1]):
        likelihood_set = np.exp(np.clip(REG_VAL * mask_likelihoods[:, i], CLIP_VAL, 0))
        # print(likelihood_set[idxes], REG_VAL * mask_likelihoods[idxes, i])
        weights = (likelihood_set + model_mask_weights[2])
        print(weights[idxes],  model_mask_weights[2])
        weights = weights / np.sum(weights)
        # print(i, np.log(weights[idxes]), weights[idxes], min_likelihood)
        weight_sets.append(weights)
     # / np.sum(likelihoods - np.min(likelihoods) + WEIGHT_LAMBDA )
    print(np.stack(weight_sets, axis=0).shape)
    return np.stack(weight_sets, axis=0)

def generate_masks(args, k_masks, passive_mask, model_performances):
    '''
    assigns each data point in the training set with a mask, based on the
    argmax model performance and the mask cost
    TODO: generate masks using a confidence strategy
    '''
    print(k_masks, np.sum(k_masks * args.EMFAC.binary_cost, axis=-1), k_masks.shape, model_performances[:10], model_performances.shape, passive_mask.shape)
    mask_cost_performances = np.sum(np.abs(k_masks - passive_mask) * args.EMFAC.binary_cost, axis=-1) + model_performances
    mask_choices = np.argmin(mask_cost_performances, axis=-1)
    print("mp", mask_cost_performances[np.arange(len(model_performances)), mask_choices][:10], mask_choices[:10] )
    # confidence computed by the difference between the best mask and the one chosen
    # assumes that the best mask is from the full mask
    mask_confidences = np.max(model_performances, axis=-1) - model_performances[np.arange(len(model_performances)), mask_choices]

    return k_masks[mask_choices]