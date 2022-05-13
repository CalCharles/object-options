import numpy as np
from Causal.Utils.get_error import get_error, error_types

def test_full(full_model, train_buffer, test_buffer, args, object_names, environment)
	''' train, test| prediction l1 per-element
	train, test| max, min, mean per-element likelihood
	interaction binary, true FP, FN
	samples of 128: target, binary, trace, passive error, prediction means
	'''
	# next target or target diff predicted
	train_target = train_buffer.next_target if full_model.predict_dynamics else train_buffer.target_diff
	test_target = test_buffer.next_target if full_model.predict_dynamics else test_buffer.target_diff

	# l1 difference between passive and active for train and test, per value, per element, without done states
	train_l1_passive = (np.abs(get_error(full_model, train_buffer, error_type=error_types.PASSIVE) - train_target ))[train_buffer.done != 1]
	train_l1_active = (np.abs(get_error(full_model, train_buffer, error_type=error_types.ACTIVE) - train_target ))[train_buffer.done != 1]
	test_l1_passive = (np.abs(get_error(full_model, test_buffer, error_type=error_types.PASSIVE) - test_target ))[test_buffer.done != 1]
	test_l1_active = (np.abs(get_error(full_model, test_buffer, error_type=error_types.ACTIVE) - test_target ))[test_buffer.done != 1]

	# variance for passive and active for test
	test_passive_var = np.mean(get_error(full_model, test_buffer, error_type=error_types.PASSIVE)[test_buffer.done != 1], axis=0)
	test_active_var = np.mean(get_error(full_model, test_buffer, error_type=error_types.ACTIVE)[test_buffer.done != 1], axis=0)

	# passive and active likelihoods per value, per element
	train_like_passive = (get_error(full_model, train_buffer, error_type = error_types.PASSIVE_LIKELIHOOD)[train_buffer.done != 1])
	train_like_active = (get_error(full_model, train_buffer, error_type = error_types.ACTIVE_LIKELIHOOD)[train_buffer.done != 1])
	test_like_passive = (get_error(full_model, test_buffer, error_type = error_types.PASSIVE_LIKELIHOOD)[test_buffer.done != 1])
	test_like_active = (get_error(full_model, test_buffer, error_type = error_types.ACTIVE_LIKELIHOOD)[test_buffer.done != 1])

	# passive and active likelihoods per element, meaned
	train_like_pmean = np.mean(train_like_passive, axis=0)
	train_like_amean = np.mean(train_like_active, axis=0)
	test_like_pmean = np.mean(test_like_passive, axis=0)
	test_like_amean = np.mean(test_like_active, axis=0)

	# passive and active likelihoods maxed, per element 
	train_like_pmax = np.max(train_like_passive, axis=0)
	train_like_amax = np.max(train_like_active, axis=0)
	train_like_pmin = np.min(train_like_passive, axis=0)
	train_like_amin = np.min(train_like_active, axis=0)
	test_like_pmax = np.max(test_like_passive, axis=0)
	test_like_amax = np.max(test_like_active, axis=0)
	test_like_pmin = np.min(test_like_passive, axis=0)
	test_like_amin = np.min(test_like_active, axis=0)

	# passive and active likelihoods, totaled average
	train_like_p = np.sum(np.mean(train_like_passive, axis=0), axis=-1)
	train_like_a = np.sum(np.mean(train_like_active, axis=0), axis=-1)
	test_like_p = np.sum(np.mean(test_like_passive, axis=0), axis=-1)
	test_like_a = np.sum(np.mean(test_like_active, axis=0), axis=-1)

	# weighted active likelihood, per element
	train_like = get_error(full_model, train_buffer, error_type = error_types.LIKELIHOOD)[train_buffer.done != 1]
	test_like = get_error(full_model, test_buffer, error_type = error_types.LIKELIHOOD)[test_buffer.done != 1]
	# weighted active likelihood, totaled
	train_like_full = np.sum(np.mean(train_like, axis=0), axis=-1)
	test_like_full = np.sum(np.mean(test_like, axis=0), axis=-1)
	# weighted active likelihood, per element
	train_like_mean = np.mean(train_like, axis=0)
	test_like_mean = np.mean(test_like, axis=0)

	# interaction binaries
	train_bin = get_error(full_model, train_buffer, error_type = error_types.INTERACTION_BINARIES)[train_buffer.done != 1]
	test_bin = get_error(full_model, test_buffer, error_type = error_types.INTERACTION_BINARIES)[test_buffer.done != 1]
	# interaction trace
	train_trace = train_buffer.trace
	test_trace = test_buffer.trace
	# interaction likelihood values
	train_likev = get_error(full_model, train_buffer, error_type = error_types.INTERACTION)[train_buffer.done != 1]
	test_likev = get_error(full_model, test_buffer, error_type = error_types.INTERACTION)[test_buffer.done != 1]
	# interaction likelihood values 
	train_likepred = full_model.test(train_likev)
	test_likepred = full_model.test(test_likev)

	# false positives and negatives compared to binaries
	BP_train, BP_test = np.mean((train_bin > train_likepred).astype(np.float64) * train_buffer.done, axis=0), np.mean((test_bin > test_likepred).astype(np.float64) * test_buffer.done, axis=0)
	BN_train, BN_test = np.mean((train_bin < train_likepred).astype(np.float64) * train_buffer.done, axis=0), np.mean((test_bin < test_likepred).astype(np.float64) * test_buffer.done, axis=0)

	# false positives and negatives compared to trace
	FP_train, FP_test = np.mean((train_trace > train_likepred).astype(np.float64) * train_buffer.done, axis=0), np.mean((test_trace > test_likepred).astype(np.float64) * test_buffer.done, axis=0)
	FN_train, FN_test = np.mean((train_trace < train_likepred).astype(np.float64) * train_buffer.done, axis=0), np.mean((test_trace < test_likepred).astype(np.float64) * test_buffer.done, axis=0)

	# proximity
	test_prox = get_error(full_model, test_buffer, error_type=error_types.PROXIMITY)

	log_string = f'train_results:'
	log_string += f'\nl1_passive: {np.mean(train_l1_passive, axis=0)}'
	log_string += f'\nl1_active: {np.mean(train_l1_active, axis=0)}'
	log_string += f'\nlike_passive: {train_like_pmean}'
	log_string += f'\nlike_active: {train_like_amean}'
	log_string += f'\nlike_pmax: {train_like_pmax}'
	log_string += f'\nlike_amax: {train_like_amax}'
	log_string += f'\nlike_pmin: {train_like_pmin}'
	log_string += f'\nlike_amin: {train_like_amin}'
	log_string += f'\nlike_p: {train_like_p}'
	log_string += f'\nlike_a: {train_like_a}'
	log_string += f'\nlike_full: {train_like_full}'
	log_string += f'\nlike_mean: {train_like_mean}'
	log_string += f'\nBP: {BP_train}'
	log_string += f'\nBN: {BN_train}'
	log_string += f'\nFP: {FP_train}'
	log_string += f'\nFN: {FN_train}'

	log_string  += f'\n\ntest_results:'
	log_string += f'\nl1_passive: {np.mean(test_l1_passive, axis=0)}'
	log_string += f'\nl1_active: {np.mean(test_l1_active, axis=0)}'
	log_string += f'\nlike_passive: {test_like_pmean}'
	log_string += f'\nlike_active: {test_like_amean}'
	log_string += f'\nlike_pmax: {test_like_pmax}'
	log_string += f'\nlike_amax: {test_like_amax}'
	log_string += f'\nlike_pmin: {test_like_pmin}'
	log_string += f'\nlike_amin: {test_like_amin}'
	log_string += f'\nlike_p: {test_like_p}'
	log_string += f'\nlike_a: {test_like_a}'
	log_string += f'\nlike_full: {test_like_full}'
	log_string += f'\nlike_mean: {test_like_mean}'
	log_string += f'\nBP: {BP_test}'
	log_string += f'\nBN: {BN_test}'
	log_string += f'\nFP: {FP_test}'
	log_string += f'\nFN: {FN_test}'

	inter_points = test_prox == 1 or test_trace == 1 or test_likepred == 1 or test_bin == 1 # high passive error could be added

	target = test_buffer.target[test_buffer.done != 1][inter_points]
	next_target = test_buffer.next_target[test_buffer.done != 1][inter_points]
	target_diff = test_buffer.target_diff[test_buffer.done != 1][inter_points]

	passive_samp = np.concatenate([test_l1_passive, test_passive_var, test_likev], axis=-1)[inter_points]
	active_samp = np.concatenate([test_l1_active, test_active_var, test_likev], axis=-1)[inter_points]
	bins = np.concatenate([test_likev, test_likepred, test_bin, test_trace, test_prox, test_like_full, test_like_p, test_like_a, test_buffer.done], axis=-1)[inter_points]
	log_string += 'Sampled computation: '
	log_string += f'\ntarget: {target[:32]}'
	log_string += f'\nnext: {next_target[:32]}'
	log_string += f'\ndiff: {target_diff[:32]}'
	log_string += f'\npassive_pred_diff: {passive_samp[:32]}'
	log_string += f'\nactive_pred_diff: {active_samp[:32]}'
	log_string += f'\nlike, pred, bin, trace, prox, alike, plike, done: {bins[:32]}'
