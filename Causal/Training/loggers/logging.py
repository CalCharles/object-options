from Causal.Utils.get_error import get_error, error_names
import numpy as np

def print_errors(full_model, rollouts, error_types=[0], sample_num=50, prenormalize=False):
	if not prenormalize: sample_list = [full_model.norm.reverse(rollouts.inter_state[:sample_num], form="inter"), full_model.norm.reverse(rollouts.next_target[:sample_num])]
	else: sample_list = [rollouts.inter_state[:sample_num], rollouts.next_target[:sample_num]]
	for error_type in error_types:
		err_vals = get_error(full_model, rollouts, error_type, reduced=False, normalized=False, prenormalize=prenormalize)
		print("error total", error_names[error_type], np.mean(err_vals, axis=0), np.sum(err_vals, axis=0))
		if len(err_vals.shape) == 1: err_vals = np.expand_dims(err_vals, axis=-1)
		sample_list.append(err_vals[:sample_num])
	np.set_printoptions(threshold=100000, linewidth=300, precision=4, suppress=True)
	print("error values", "Inter", "Next Target", [error_names[et] for et in error_types], np.concatenate(sample_list, axis=-1))
	np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
