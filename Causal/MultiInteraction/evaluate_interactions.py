import numpy as np
import torch
from Network.network_utils import pytorch_model
from Causal.Utils.instance_handling import get_batch, get_valid


def compute_distance(params1, params2): # computes the wasserstein 2 distance, TODO: could use KL distance
    # return torch.norm(params1[0] - params2[0], p=2, dim=-1) + torch.norm((params1[1] + params2[1] - 2 * torch.sqrt(params1[1] * params2[1])), p=2, dim=-1)
    return torch.norm(params1[0] - params2[0], p=2, dim=-1) # + torch.norm((params1[1] + params2[1] - 2 * torch.sqrt(params1[1] * params2[1])), p=2, dim=-1)

BATCH_SIZE = 1024

def evaluate_buffer(num_iters, full_model, buffer, object_buffer, args, environment, sampling_mode=False):
    all_inters = list()
    num_iters = int(np.ceil(len(buffer) / BATCH_SIZE)) if num_iters <= 0 else num_iters
    # for dv in [0.2 * i for i in range(10)]:
    for i in range(num_iters):
        if sampling_mode: full_batch, batch, idxes = get_batch(BATCH_SIZE, full_model.form == "all", buffer, object_buffer, None)
        else: full_batch, batch, idxes = get_batch((i*BATCH_SIZE,(i+1)*BATCH_SIZE), full_model.form == "all", buffer, object_buffer, None)
        num_factors = len(environment.all_names) - 2
        all_combinations = [np.arange(num_factors) for i in range(args.multi_inter.max_combination)]
        all_combinations = np.array(np.meshgrid(*all_combinations)).T.reshape(-1,args.multi_inter.max_combination)
        # TODO: make general to not just full_name
        inter_masks = np.ones((len(batch), environment.object_instanced[args.EMFAC.full_train], num_factors))
        valid = get_valid(batch.valid, full_model.valid_indices)
        all_dists = list()
        for comb in all_combinations:
            print("name: ", [environment.all_names[c] for c in comb], comb, full_model.valid_indices)
            given_mask = np.ones((len(batch), num_factors))
            given_mask[...,comb] = 0
            given_valid = np.expand_dims(np.sum(batch.valid[...,comb], axis=-1).astype(bool).astype(float), axis=-1)
            #  compute the full prediction parameters
            active_full, inter, hot_mask, full_mask, target, _, active_full_log_probs, _ = full_model.active_open_likelihoods(batch)
            active_likelihood_full, active_prediction_params = - active_full_log_probs, active_full if not full_model.cluster_mode else (active_full[0][...,target.shape[-1]:target.shape[-1] * 2], active_full[1][...,target.shape[-1]:target.shape[-1] * 2])
            # compute the masked prediction parameters
            active_given_params, _, _, _, target, active_given_dist, _, active_given_inputs= full_model.given_likelihoods(batch, given_mask)
            # reshape to add in num_instances of target
            active_prediction_params, active_given_params = (active_prediction_params[0].reshape(len(batch), inter_masks.shape[1], -1), active_prediction_params[1].reshape(len(batch), inter_masks.shape[1], -1)), (active_given_params[0].reshape(len(batch), inter_masks.shape[1], -1), active_given_params[1].reshape(len(batch), inter_masks.shape[1], -1))
            # compute the distance between the full and the given
            dists = pytorch_model.unwrap(compute_distance(active_prediction_params, active_given_params)) # distances as batch, target factors
            all_dists.append(dists)
            close = dists > args.multi_inter.dist_epsilon
            # close = dists > dv
            inter_masks[...,comb] = np.expand_dims(close.astype(int), -1)
            usable_idxes = np.nonzero(given_valid.squeeze() * valid.squeeze() * (1-batch.done).squeeze())
            tstidxes = np.random.choice(usable_idxes[0], size=(30,), replace=False)
            target = target.reshape(len(batch), inter_masks.shape[1], -1)
            print(comb, given_mask[0],
                    np.concatenate([np.expand_dims(np.expand_dims(idxes[tstidxes], axis=-1), axis=-1), np.expand_dims(valid[tstidxes], axis=-1), 
                                    # pytorch_model.unwrap(target[tstidxes]), 
                                    pytorch_model.unwrap((active_prediction_params[0][tstidxes] - target[tstidxes]).abs().sum(dim=-1).unsqueeze(-1)),
                                    pytorch_model.unwrap((active_prediction_params[0][tstidxes] - active_given_params[0][tstidxes]).abs().sum(dim=-1).unsqueeze(-1)),
                                    # np.expand_dims(batch.valid[tstidxes], axis=1),
                                    np.expand_dims(dists[tstidxes], axis=-1), inter_masks[tstidxes][...,comb], np.expand_dims(batch.trace[tstidxes][...,comb], axis=1)], axis=-1))
                    # np.concatenate([np.expand_dims(idxes[tstidxes], axis=-1), given_valid[tstidxes], valid[tstidxes], inter_masks[tstidxes][...,0,comb], batch.trace[tstidxes][...,comb], dists[tstidxes]], axis=-1))
            # print(pytorch_model.unwrap((active_prediction_params[0] - target).abs().sum(dim=-1))[usable_idxes] )
        all_inters.append(inter_masks)
        all_dists = np.concatenate(all_dists, axis=-1)
        inter_masks[...,full_model.valid_indices] = 1 # the passive mask is unstable, so set it to always on
        usable_idxes = np.nonzero(given_valid.squeeze() * valid.squeeze())
        bin_error = inter_masks[usable_idxes][:,0] - batch.trace[usable_idxes]
        # print(np.concatenate([np.expand_dims(idxes[usable_idxes], axis=-1), given_valid[usable_idxes], valid[usable_idxes], inter_masks[usable_idxes][:,0], batch.trace[usable_idxes], all_dists[usable_idxes], bin_error], axis=-1)[:100])
        print("at distance", args.multi_inter.dist_epsilon)
        print("false positive:", np.sum((bin_error > 0).astype(int)) / np.sum(np.abs(bin_error)) )
        print("false negative:", np.sum((bin_error < 0).astype(int)) / np.sum(np.abs(bin_error)) )
        print(len(usable_idxes), inter_masks.shape[-2], (inter_masks.shape[-1]-1))
        print("total_error:", np.sum(np.abs(bin_error)) / (len(given_valid.squeeze() * valid.squeeze()) * inter_masks.shape[-2] * (inter_masks.shape[-1]-1)))
    all_inters = np.concatenate(all_inters, axis=0)
    return all_inters


def evaluate_null_interaction(full_model, train_all_buffer, train_object_rollout, test_all_buffer, test_object_rollout, args, environment):
    '''
    evaluates the interaction binaries using null combinations of up to args.multi_inter.max_combination for every value in the buffers
    '''
    train_inters = evaluate_buffer(args.train.num_iters, full_model, train_all_buffer, train_object_rollout, args, environment, sampling_mode=True)
    test_inters = evaluate_buffer(args.train.num_iters, full_model, test_all_buffer, test_object_rollout, args, environment, sampling_mode=True)
    return train_inters, test_inters
