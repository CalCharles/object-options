import numpy as np
from Causal.Utils.get_error import get_error, error_types
import logging

def test_full_train(full_model, train_buffer, args, object_names, environment):
	# prints out training assessment, with most of the same values as test_full
	# predicted target for the model
	train_target = train_buffer.next_target if full_model.predict_dynamics else train_buffer.target_diff

	train_valid = (train_buffer.done != 1).squeeze()
	
	# l1 difference per element
	train_l1_passive = get_error(full_model, train_buffer, error_type=error_types.PASSIVE, reduced=False)[train_valid]
	train_l1_active = get_error(full_model, train_buffer, error_type=error_types.ACTIVE, reduced=False)[train_valid]
	
	# active and passive mean, max min and sum
	train_like_passive = (get_error(full_model, train_buffer, error_type = error_types.PASSIVE_LIKELIHOOD, reduced=False)[train_valid])
	train_like_active = (get_error(full_model, train_buffer, error_type = error_types.ACTIVE_LIKELIHOOD, reduced=False)[train_valid])
	train_like_pmean = np.mean(train_like_passive, axis=0)
	train_like_amean = np.mean(train_like_active, axis=0)
	train_like_pmax = np.max(train_like_passive, axis=0)
	train_like_amax = np.max(train_like_active, axis=0)
	train_like_pmin = np.min(train_like_passive, axis=0)
	train_like_amin = np.min(train_like_active, axis=0)
	train_like_p = np.sum(np.mean(train_like_passive, axis=0), axis=-1)
	train_like_a = np.sum(np.mean(train_like_active, axis=0), axis=-1)

	# interaction likelihood values
	train_like = get_error(full_model, train_buffer, error_type = error_types.LIKELIHOOD, reduced=False)[train_valid]
	train_like_full = np.sum(np.mean(train_like, axis=0), axis=-1)
	train_like_mean = np.mean(train_like, axis=0)
	train_bin = get_error(full_model, train_buffer, error_type = error_types.INTERACTION_BINARIES)[train_valid]
	train_trace = train_buffer.trace[train_valid]
	train_likev = get_error(full_model, train_buffer, error_type = error_types.INTERACTION_RAW)[train_valid]
	train_likepred = full_model.test(train_likev)

	BN_train = np.mean((train_bin > train_likepred).astype(np.float64), axis=0) / np.sum(train_bin.astype(np.float64)) 
	BP_train = np.mean((train_bin < train_likepred).astype(np.float64), axis=0) / np.sum((train_bin == 0).astype(np.float64)) 
	FN_train = np.mean((train_trace > train_likepred).astype(np.float64), axis=0) / np.sum(train_trace.astype(np.float64))
	FP_train = np.mean((train_trace < train_likepred).astype(np.float64), axis=0) / np.sum((train_trace == 0).astype(np.float64))

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
	log_string += f'\nBP: {BP_train.squeeze()}'
	log_string += f'\nBN: {BN_train.squeeze()}'
	log_string += f'\nFP: {FP_train.squeeze()}'
	log_string += f'\nFN: {FN_train.squeeze()}'
	print(log_string)
	logging.info(log_string)

def test_full(full_model, test_buffer, args, object_names, environment):
	''' train, test| prediction l1 per-element
	train, test| max, min, mean per-element likelihood
	interaction binary, true FP, FN
	samples of 128: target, binary, trace, passive error, prediction means
	'''
	# next target or target diff predicted
	test_target = test_buffer.target_diff if full_model.predict_dynamics else test_buffer.next_target
	test_valid = (test_buffer.done != 1).squeeze()[:len(test_buffer)] # TODO: needs to regulate the length because of a bug in Tianshou

	# print(np.concatenate([test_buffer.target[:100], test_target[:100], test_buffer.done[:100], test_buffer.true_done[:100]],axis=-1))
	# print(np.concatenate([test_buffer.target[100:200], test_target[100:200], test_buffer.done[100:200], test_buffer.true_done[100:200]],axis=-1))
	# print(np.concatenate([test_buffer.target[200:300], test_target[200:300], test_buffer.done[200:300], test_buffer.true_done[200:300]],axis=-1))
	# print(np.concatenate([test_buffer.target[300:400], test_target[300:400], test_buffer.done[300:400], test_buffer.true_done[300:400]],axis=-1))
	# print(np.concatenate([test_buffer.target[400:500], test_target[400:500], test_buffer.done[400:500], test_buffer.true_done[400:500]],axis=-1))
	# print(full_model.predict_dynamics)
	# error

	# l1 difference between passive and active for train and test, per value, per element, without done states
	test_l1_passive = get_error(full_model, test_buffer, error_type=error_types.PASSIVE, reduced=False)[test_valid]
	test_l1_active = get_error(full_model, test_buffer, error_type=error_types.ACTIVE, reduced=False)[test_valid]

	# l1 difference between passive and active for train and test, per value, per element, without done states
	test_raw_passive = get_error(full_model, test_buffer, error_type=error_types.PASSIVE_RAW, reduced=False, normalized=True)[test_valid]
	test_raw_active = get_error(full_model, test_buffer, error_type=error_types.ACTIVE_RAW, reduced=False, normalized=True)[test_valid]

	# variance for passive and active for test
	test_passive_var = get_error(full_model, test_buffer, error_type=error_types.PASSIVE_VAR)[test_valid]
	test_active_var = get_error(full_model, test_buffer, error_type=error_types.ACTIVE_VAR)[test_valid]

	# passive and active likelihoods per value, per element
	test_like_passive = (get_error(full_model, test_buffer, error_type = error_types.PASSIVE_LIKELIHOOD, reduced=False)[test_valid])
	test_like_active = (get_error(full_model, test_buffer, error_type = error_types.ACTIVE_LIKELIHOOD, reduced=False)[test_valid])

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

	# weighted active likelihood, per element
	test_like = get_error(full_model, test_buffer, error_type = error_types.LIKELIHOOD, reduced=False)[test_valid]
	# weighted active likelihood, totaled
	test_like_full = np.sum(np.mean(test_like, axis=0), axis=-1)
	# weighted active likelihood, per element
	test_like_mean = np.mean(test_like, axis=0)

	# interaction binaries
	test_bin = get_error(full_model, test_buffer, error_type = error_types.INTERACTION_BINARIES)[test_valid]
	# interaction trace
	test_trace = test_buffer.trace[:len(test_buffer)][test_valid]
	# interaction likelihood values
	test_likev = get_error(full_model, test_buffer, error_type = error_types.INTERACTION_RAW)[test_valid]
	# interaction likelihood values predictive binary 
	test_likepred = full_model.test(test_likev)

	# false positives and negatives compared to binaries
	BN_test = np.mean((test_bin > test_likepred).astype(np.float64), axis=0) / np.sum(test_bin.astype(np.float64))
	BP_test = np.mean((test_bin < test_likepred).astype(np.float64), axis=0) / np.sum((test_bin == 0).astype(np.float64))

	# false positives and negatives compared to trace
	FN_test = np.mean((test_trace > test_likepred).astype(np.float64), axis=0) / np.sum(test_trace.astype(np.float64))
	FP_test = np.mean((test_trace < test_likepred).astype(np.float64), axis=0) / np.sum((test_trace == 0).astype(np.float64))

	# proximity
	test_prox = get_error(full_model, test_buffer, error_type=error_types.PROXIMITY)[test_valid]

	log_string  = f'\n\ntest_results:'
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
	log_string += f'\nBP: {BP_test.squeeze()}'
	log_string += f'\nBN: {BN_test.squeeze()}'
	log_string += f'\nFP: {FP_test.squeeze()}'
	log_string += f'\nFN: {FN_test.squeeze()}'

	# inter_points = (test_prox == 1).squeeze() + (test_trace.squeeze() == 1) + (test_likepred == 1).squeeze() + (test_bin == 1).squeeze() # high passive error could be added
	# inter_points = (test_trace.squeeze() == 1) + (test_likepred == 1).squeeze() + (test_bin == 1).squeeze() # high passive error could be added
	# inter_points = (test_trace.squeeze() != test_likepred.squeeze()) + (test_likepred.squeeze() != test_bin.squeeze()).squeeze() + (test_bin.squeeze() != test_trace.squeeze()).squeeze() # high passive error could be added
	inter_points = ((test_likepred.squeeze()) + (test_bin.squeeze())).astype(bool).squeeze() # high passive error could be added
	target = test_buffer.target[:len(test_buffer)][test_valid]#[inter_points]
	inter_state = test_buffer.inter_state[:len(test_buffer)][test_valid]#[inter_points]
	next_target = test_buffer.next_target[:len(test_buffer)][test_valid]#[inter_points]
	target_diff = test_buffer.target_diff[:len(test_buffer)][test_valid]#[inter_points]

	passive_samp = np.concatenate([target_diff, test_raw_passive, test_l1_passive, test_passive_var, test_likev], axis=-1)[inter_points]
	active_samp = np.concatenate([target_diff, test_raw_active, test_l1_active, test_active_var, test_likev], axis=-1)[inter_points]
	bins = np.concatenate([
							test_likev, 
							test_likepred, 
							test_bin, 
							test_trace, 
							test_prox,
							np.expand_dims(np.sum(test_like, axis=-1), -1), 
							np.expand_dims(np.sum(test_like_passive, axis=-1), -1), 
							np.expand_dims(np.sum(test_like_active, axis=-1), -1),
							full_model.norm.reverse(inter_state, form = "inter"),
							full_model.norm.reverse(next_target)], axis=-1)[inter_points]
	log_string += '\n\nSampled computation: '
	# log_string += f'\ntarget: {target[:128]}'
	# log_string += f'\nnext: {next_target[:128]}'
	# log_string += f'\ndiff: {target_diff[:128]}'
	log_string += f'\npassive_pred_diff: {passive_samp[:128]}'
	log_string += f'\nactive_pred_diff: {active_samp[:128]}'
	log_string += f'\nlike, pred, bin, trace, prox, like, plike, alike: {bins[:128]}'
	logging.info(log_string)
	print(log_string)