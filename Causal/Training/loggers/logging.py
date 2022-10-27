from Causal.Utils.get_error import get_error, error_names
import numpy as np

def print_errors(full_model, rollouts, object_rollouts=None, error_types=[0], sample_num=50, prenormalize=False):
	# if object_rollouts is not None then we are in full mode
	inter_state, next_target = rollouts.obs[:sample_num] if object_rollouts is not None else rollouts.inter_state[:sample_num], object_rollouts.obs_next[:sample_num] if object_rollouts is not None else rollouts.next_target[:sample_num]
	if not prenormalize: sample_list = [full_model.norm.reverse(inter_state, form="inter"), full_model.norm.reverse(next_target, name=full_model.name)]
	else: sample_list = [inter_state, next_target]
	for error_type in error_types:
		err_vals = get_error(full_model, rollouts, error_type=error_type, object_rollout = object_rollouts, reduced=False, normalized=False, prenormalize=prenormalize)
		if len(err_vals.shape) == 3: err_vals = np.sum(err_vals, axis=-1)
		if len(err_vals.shape) == 1: err_vals = np.expand_dims(err_vals, axis=-1)
		print("error total", error_names[error_type], np.mean(err_vals, axis=0), np.sum(err_vals, axis=0), err_vals.shape)
		sample_list.append(err_vals[:sample_num])
	np.set_printoptions(threshold=100000, linewidth=300, precision=4, suppress=True)
	print("error values", "Inter", "Next Target", [error_names[et] for et in error_types], np.concatenate(sample_list, axis=-1))
	np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
