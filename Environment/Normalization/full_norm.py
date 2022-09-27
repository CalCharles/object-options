import numpy as np
from State.feature_selector import broadcast
from Network.network_utils import pytorch_model

def compute_norm(mean, var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return (state - mean) / var
	mean, var = broadcast(mean, state.shape[-1] // len(var)),  broadcast(var, state.shape[-1] // len(var))
	return (state - mean) / var

def compute_reverse(mean,var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return state * var + mean
	mean, var = broadcast(mean, state.shape[-1] // len(var)),  broadcast(var, state.shape[-1] // len(var))
	return state * var + mean	

def generate_multiobject_norm(nl_dict, names, object_counts):
	firstv = np.concatenate([(broadcast(nl_dict[n][0], object_counts[n])) for n in names], axis=-1)
	secondv = np.concatenate([(broadcast(nl_dict[n][1], object_counts[n])) for n in names], axis=-1)
	return (firstv, secondv) 

class FullNormalizationModule(): # TODO: FULL REWRITE TO HANDLE INSTANCED-COUNTED STATE
	def __init__(self, lim_dict, dynamics_dict, target_name, object_counts, all_names):
		# @param inter_names is the ordering of the names for the interaction state
		self.lim_dict = lim_dict # the bounds of positions for where an object can be
		self.dynamics_dict = dynamics_dict # the bounds for the amount an object can change in a single timestep
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.dynamics_norm_dict = {n: ((self.dynamics_dict[n][1] + self.dynamics_dict[n][0])/2, (self.dynamics_dict[n][1] - self.dynamics_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.target_name = target_name
		self.parent_names = copy.deepcopy(inter_names).remove(target_name)
		self.inter_names = all_names
		self.object_counts = object_counts # environment object counts 

		# specify different norms
		self.raw_norm, self.raw_lim = (128, 128), (0, 256) # assumes images are ranged 256
		self.target_norm, self.target_lim = self.norm_dict[self.object_names.target], self.lim_dict[self.object_names.target]
		# interaction state norm
		self.inter_norm, self.inter_lim = generate_multiobject_norm(self.norm_dict, inter_names, object_counts), generate_multiobject_norm(self.lim_dict, inter_names, object_counts)
		self.parent_norm, self.parent_lim = generate_multiobject_norm(self.norm_dict, , object_counts) if len(self.object_names.additional) > 0 else None, generate_multiobject_norm(self.lim_dict, self.object_names.additional, object_counts) if len(self.object_names.additional) > 0 else None
		self.part_norm, self.part_lim = {n: self.norm_dict[n] for n in all_names}, {n: self.lim_dict[n] for n in all_names}
		
		self.difference_norm, self.difference_lim = (np.zeros(self.norm_dict[self.object_names.target][0].shape), self.norm_dict[self.object_names.target][1] * 2), (self.target_lim[0] - self.target_lim[1], self.target_lim[0] + self.target_lim[1])
		# gets the appropriate normalization values based on the target
		self.norm_forms = {"target": self.target_norm, "inter": self.inter_norm, "parent": self.parent_norm, "diff": self.difference_norm, "dyn": self.dynamics_norm,"part": self.part_norm, "raw": self.raw_norm}
		self.lim_forms = {"target": self.target_lim, "inter": self.inter_lim, "parent": self.parent_lim, "diff": self.difference_lim, "dyn": self.dynamics_lim,"part": self.part_lim, "raw": self.raw_lim}
		# TODO: handle relative norm between block and obstacles (internal relative?)

	def get_mean_var(self, form, idxes):
		# logic for partial additional uss an integer after "additional"
		if form not in list(self.norm_forms.keys()): norm = self.norm_forms["part"][form] # can't have an object named anything in the norm dict
		else: norm = self.norm_forms[form]
		if norm is None: return None, None
		mean = norm[0][idxes] if idxes is not None else norm[0]
		var = norm[1][idxes] if idxes is not None else norm[1]
		return mean, var

	def __call__(self, state, form="target", idxes=None):
		'''
		takes the normalization of the state, the form decides which norm to use
		valid forms: target, inter, parent, difference, relative
		'''
		mean, var = self.get_mean_var(form, idxes)
		# print(state, mean,var, form)
		if mean is None: return state
		return compute_norm(mean, var, state)

	def reverse(self, state, form = "target", idxes=None):
		mean, var = self.get_mean_var(form, idxes)
		# print(state, mean, var, form)
		if mean is None: return state
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target"):
		if form not in list(self.lim_forms.keys()): lims = self.lim_forms["part"][form] # can't have an object named anything in the norm dict
		else: lims = self.lim_forms[form]
		return np.clip(state, lims[0], lims[1])