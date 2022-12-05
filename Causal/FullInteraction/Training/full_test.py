import numpy as np
from Causal.Utils.get_error import get_error, error_types
import logging

def test_full(full_model, test_full_buffer, test_object_buffer, args, environment):
	''' train, test| prediction l1 per-element
	train, test| max, min, mean per-element likelihood
	interaction binary, true FP, FN
	samples of 128: target, binary, trace, passive error, prediction means
	'''
	# next target or target diff predicted
	test_target = test_object_buffer.target_diff if full_model.predict_dynamics else test_object_buffer.next_target
	test_valid = (test_full_buffer.done != 1).squeeze()[:len(test_full_buffer)] # TODO: needs to regulate the length because of a bug in Tianshou

	# l1 difference between passive and active for train and test, per value, per element, without done states
	test_l1_passive = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.PASSIVE, reduced=False)[test_valid]
	test_l1_active = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.ACTIVE, reduced=False)[test_valid]
	test_l1_passive_mean = np.mean(test_l1_passive, axis=0)
	test_l1_active_mean = np.mean(test_l1_active, axis=0)


	# l1 difference between passive and active for train and test, per value, per element, without done states
	test_raw_passive = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.PASSIVE_RAW, reduced=False, normalized=True)[test_valid]
	test_raw_active = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.ACTIVE_RAW, reduced=False, normalized=True)[test_valid]

	# variance for passive and active for test
	test_passive_var = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.PASSIVE_VAR)[test_valid]
	test_active_var = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.ACTIVE_VAR)[test_valid]

	# passive and active likelihoods per value, per element
	test_like_passive = (get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type = error_types.PASSIVE_LIKELIHOOD, reduced=False)[test_valid])
	test_like_active = (get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type = error_types.ACTIVE_LIKELIHOOD, reduced=False)[test_valid])

	# passive and active likelihoods per element, meaned
	test_like_pmean = np.mean(test_like_passive, axis=0)
	test_like_amean = np.mean(test_like_active, axis=0)

	# passive and active likelihoods maxed, per element 
	test_like_pmax = np.max(test_like_passive, axis=0)
	test_like_amax = np.max(test_like_active, axis=0)
	test_like_pmin = np.min(test_like_passive, axis=0)
	test_like_amin = np.min(test_like_active, axis=0)

	# passive and active likelihoods, totaled average
	test_like_p = np.sum(np.mean(test_like_passive, axis=0), axis=-1)
	test_like_a = np.sum(np.mean(test_like_active, axis=0), axis=-1)

	# open active likelihood, per element
	test_open_like = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type = error_types.LIKELIHOOD, reduced=False)[test_valid]
	# weighted active likelihood, totaled
	test_open_full = np.sum(np.mean(test_open_like, axis=0), axis=-1)
	# weighted active likelihood, per element
	test_open_mean = np.mean(test_open_like, axis=0)

	# interaction binaries (these are binaries per state)
	test_bin = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type = error_types.INTERACTION_BINARIES)[test_valid]
	# interaction trace (these are length(all) vectors per state)
	test_trace = test_object_buffer.trace[:len(test_object_buffer)][test_valid]
	# interaction likelihood values (length all vectors per state)
	test_likev = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type = error_types.INTERACTION_RAW)[test_valid]
	# interaction mask values flat (length all vectors per state) 
	test_flat = full_model.test(test_likev)
	# non-passive predictions matching
	test_nonpassive_flat = full_model.inter_passive(test_flat).astype(int)
	test_nonpassive_trace = full_model.inter_passive(test_trace).astype(int)



	# l1 difference between the trace and the flat, per element
	flat_diff = np.mean(np.abs(test_trace - test_flat.astype(int)), axis=0)
	flat_miss = np.sum(np.abs(test_trace - test_flat.astype(int)), axis=-1)
	flat_average_miss = np.sum(flat_diff, axis=-1)

	# l1 difference between the trace and the likev
	likev_diff = np.mean(np.abs(test_trace - test_likev.astype(int)), axis=0)
	likev_miss = np.sum(np.abs(test_trace - test_likev.astype(int)), axis=-1)
	likev_average_miss = np.sum(likev_diff, axis=-1)

	# binary accuracy at predicting non-passive dynamics
	bin_diff = np.mean(np.abs(test_nonpassive_trace - test_bin.astype(int)), axis=0)

	# flat accuracy at predicting non-passive dynamics
	flat_passive_diff = np.mean(np.abs(test_nonpassive_trace - test_nonpassive_flat), axis=0)

	# proximity
	# print("getting error", test_full_buffer.parent_state[:10], test_full_buffer.target[:10])
	test_prox = get_error(full_model, test_full_buffer, object_rollout=test_object_buffer, error_type=error_types.PROXIMITY_FULL, normalized=True)[test_valid]

	inter_points = (flat_miss >= 1) + (likev_miss >= 1)
	log_values = {
		'l1_passive': test_l1_passive_mean,
		'l1_active': test_l1_active_mean,
		'like_passive': test_like_pmean,
		'like_active': test_like_amean,
		'like_pmax': test_like_pmax,
		'like_amax': test_like_amax,
		'like_pmin': test_like_pmin,
		'like_amin': test_like_amin,
		'like_p': test_like_p,
		'like_a': test_like_a,
		'like_full': test_open_full,
		'like_mean': test_open_mean,
		'flat_diff': flat_diff,
		'flat_average_miss': flat_average_miss,
		'likev_diff': likev_diff,
		'likev_average_miss': likev_average_miss,
		'bin_diff': bin_diff,
		'flat_passive_diff': flat_passive_diff,
		"inter_points": np.sum(inter_points.astype(int))
	}

	log_string  = f'\n\ntest_results:'
	for key in log_values.keys():
		log_string += '\n' + key + f': {log_values[key]}'


	target = test_object_buffer.obs[:len(test_object_buffer)][test_valid]#[inter_points]
	inter_state = test_full_buffer.obs[:len(test_full_buffer)][test_valid]#[inter_points]
	next_target = test_object_buffer.obs_next[:len(test_object_buffer)][test_valid]#[inter_points]
	target_diff = test_object_buffer.target_diff[:len(test_object_buffer)][test_valid]#[inter_points]

	passive_samp = np.concatenate([target_diff, test_raw_passive, test_l1_passive, test_passive_var, test_likev], axis=-1)[inter_points]
	active_samp = np.concatenate([target_diff, test_raw_active, test_l1_active, test_active_var, test_likev], axis=-1)[inter_points]
	bins = np.concatenate([
							test_likev, 
							test_flat, 
							test_bin, 
							test_trace, 
							test_prox,
							# test_l1_passive,
							np.expand_dims(np.sum(test_open_like, axis=-1), -1),
							np.expand_dims(np.sum(test_like_passive, axis=-1), -1), 
							np.expand_dims(np.sum(test_like_active, axis=-1), -1),
							full_model.norm.reverse(inter_state, form = "inter"),
							full_model.norm.reverse(next_target, name=full_model.name)], axis=-1)[inter_points]
	log_string += '\n\nSampled computation: '
	# log_string += f'\ntarget: {target[:128]}'
	# log_string += f'\nnext: {next_target[:128]}'
	# log_string += f'\ndiff: {target_diff[:128]}'
	log_string += f'\npassive_pred_diff: {passive_samp}'
	log_string += f'\nactive_pred_diff: {active_samp}'
	log_string += f'\nlike, pred, bin, trace, prox, like, plike, alike: {bins}'
	logging.info(log_string)
	print(log_string, np.sum(inter_points.astype(int)))
	return log_values