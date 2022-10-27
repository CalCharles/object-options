import numpy as np
import copy
from State.feature_selector import broadcast
from Network.network_utils import pytorch_model
from Environment.Normalization.full_norm import compute_norm, compute_reverse, generate_multiobject_norm

MINVAR = 1E-6

def create_dict(value_dict, pad_object_length, append_length):
	completed_dict = dict()
	lim_dict = dict()
	for n in value_dict.keys():
		pad_length = pad_object_length - value_dict[n][1].shape[0]
		mean = (value_dict[n][1] + value_dict[n][0])/2
		var = abs(value_dict[n][1] - value_dict[n][0])/2 + MINVAR
		low, high = value_dict[n][0], value_dict[n][1]
		if pad_length > 0:
			mean, var = np.concatenate((mean, np.zeros(pad_length)), axis=-1), np.concatenate((var, np.zeros(pad_length)), axis=-1) + MINVAR
			low, high = np.concatenate((low, np.zeros(pad_length) - MINVAR), axis=-1), np.concatenate((high, np.zeros(pad_length)), axis=-1) + MINVAR
		if append_length > 0:
			mean, var = np.concatenate((mean, np.zeros(append_length)), axis=-1), np.concatenate((var, np.ones(append_length)), axis=-1)
			low, high = np.concatenate((low, np.zeros(append_length)), axis=-1), np.concatenate((high, np.ones(append_length)), axis=-1)
		completed_dict[n] = (mean, var)
		lim_dict[n] = (low, high)
	return completed_dict, lim_dict

class PadNormalizationModule(): # TODO: FULL REWRITE TO HANDLE INSTANCED-COUNTED STATE
	def __init__(self, lim_dict, dynamics_dict, object_counts, object_names, pad_object_length, append_length):
		# @param inter_names is the ordering of the names for the interaction state
		self.lim_dict = lim_dict # the bounds of positions for where an object can be
		self.dynamics_dict = dynamics_dict # the bounds for the amount an object can change in a single timestep
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.pad_size = pad_object_length
		self.append_size = append_length
		self.norm_dict, self.pad_lim_dict = create_dict(self.lim_dict, self.pad_size, 0)
		self.dynamics_norm_dict, self.pad_dynamics_dict = create_dict(self.dynamics_dict, self.pad_size, 0)
		self.difference_norm, self.difference_lim = {n: (np.zeros(self.norm_dict[n][0].shape), self.norm_dict[n][1] * 2) for n in object_names}, \
							{n: (self.pad_lim_dict[n][0] - self.pad_lim_dict[n][1], self.pad_lim_dict[n][0] + self.pad_lim_dict[n][1]) for n in object_names}
		self.multi_names = set(["target", "dyn", "diff"])
		self.inter_names = object_names
		self.object_counts = object_counts # environment object counts 

		# specify different norms
		self.raw_norm, self.raw_lim = (128, 128), (0, 256) # assumes images are ranged 256
		# interaction state norm
		self.append_norm_dict, self.append_lim_dict = create_dict(self.lim_dict, self.pad_size, append_length)
		inter_names = copy.deepcopy(self.inter_names)
		for n in ["Reward", "Done"]:
			inter_names.remove(n)
		self.inter_norm, self.inter_lim = generate_multiobject_norm(self.append_norm_dict, inter_names, object_counts), generate_multiobject_norm(self.append_lim_dict, self.inter_names, object_counts)
		self.rel_norm, self.rel_lim = (np.zeros(self.inter_norm[0].shape), self.inter_norm[1] * 2), \
							(self.inter_lim[0] - self.inter_lim[1], self.inter_lim[0] + self.inter_lim[1])
		
		# gets the appropriate normalization values based on the target
		self.norm_forms = {"target": self.norm_dict, "dyn": self.dynamics_norm_dict, "diff": self.difference_norm,
					"inter": self.inter_norm, "rel": self.rel_norm, "raw": self.raw_norm}
		self.lim_forms = {"target": self.pad_lim_dict, "dyn": self.pad_dynamics_dict, "diff": self.difference_lim,
					"inter": self.inter_lim, "rel": self.rel_lim, "raw": self.raw_lim}
		# TODO: handle relative norm between block and obstacles (internal relative?)

	def get_mean_var(self, form, name, idxes):
		# logic for partial additional uss an integer after "additional"
		if form in self.multi_names: norm = self.norm_forms[form][name] # can't have an object named anything in the norm dict
		else: norm = self.norm_forms[form]
		if norm is None: return None, None
		mean = norm[0][idxes] if idxes is not None else norm[0]
		var = norm[1][idxes] if idxes is not None else norm[1]
		return mean, var

	def __call__(self, state, form="target", name=None, idxes=None):
		'''
		takes the normalization of the state, the form decides which norm to use
		valid forms: target, inter, parent, difference, relative
		'''
		mean, var = self.get_mean_var(form, name, idxes)
		if mean is None: return state
		return compute_norm(mean, var, state)

	def reverse(self, state, form = "target", name=None, idxes=None):
		mean, var = self.get_mean_var(form, name, idxes)
		if mean is None: return state
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target", name=None):
		if form in self.multi_names: lims = self.lim_forms[form][name] # can't have an object named anything in the norm dict
		else: lims = self.lim_forms[form]
		return np.clip(state, lims[0], lims[1])