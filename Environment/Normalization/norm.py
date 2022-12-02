import numpy as np
from State.feature_selector import broadcast
from Network.network_utils import pytorch_model

def compute_norm(mean, var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return (state - mean) / var
	elif state.shape[-1] < len(mean): # using the first n elements
		return (state - mean[:state.shape[-1]]) / var[:state.shape[-1]]
	mean, var = broadcast(mean, state.shape[-1] // len(var)),  broadcast(var, state.shape[-1] // len(var))
	return (state - mean) / var

def compute_reverse(mean,var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return state * var + mean
	elif state.shape[-1] < len(mean): # using the first n elements
		return state * var[:state.shape[-1]] + mean[:state.shape[-1]]
	mean, var = broadcast(mean, state.shape[-1] // len(var)),  broadcast(var, state.shape[-1] // len(var))
	return state * var + mean	

def generate_multiobject_norm(nl_dict, names, object_counts):
	firstv = np.concatenate([(broadcast(nl_dict[n][0], object_counts[n])) for n in names], axis=-1)
	secondv = np.concatenate([(broadcast(nl_dict[n][1], object_counts[n])) for n in names], axis=-1)
	return (firstv, secondv) 

def generate_relative_norm(norm1, norm2):
	relative_norm = (norm1[0] - norm2[0], norm1[1] + norm2[1]  + 1e-6)
	relative_lim = (relative_norm[0]-relative_norm[1], relative_norm[0] + relative_norm[1])
	return relative_norm, relative_lim

class NormalizationModule():
	def __init__(self, lim_dict, true_lim_dict, dynamics_dict, object_names, object_counts, inter_names):
		# @param inter_names is the ordering of the names for the interaction state
		self.lim_dict = lim_dict # the bounds of positions for where an object can be
		self.true_lim_dict = true_lim_dict
		self.dynamics_dict = dynamics_dict # the bounds for the amount an object can change in a single timestep
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, abs(self.lim_dict[n][1] - self.lim_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.dynamics_norm_dict = {n: ((self.dynamics_dict[n][1] + self.dynamics_dict[n][0])/2, abs(self.dynamics_dict[n][1] - self.dynamics_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.object_names = object_names
		self.object_counts = object_counts # environment object counts 

		# specify different norms
		self.raw_norm, self.raw_lim = (128, 128), (0, 256) # assumes images are ranged 256
		self.target_norm, self.target_lim = self.norm_dict[self.object_names.target], self.lim_dict[self.object_names.target]
		self.parent_norm, self.parent_lim = self.norm_dict[self.object_names.primary_parent], self.lim_dict[self.object_names.primary_parent]
		self.dynamics_norm, self.dynamics_lim = self.dynamics_norm_dict[self.object_names.target], self.dynamics_dict[self.object_names.target]
		# interaction state norm
		self.inter_norm, self.inter_lim = generate_multiobject_norm(self.norm_dict, inter_names, object_counts), generate_multiobject_norm(self.lim_dict, inter_names, object_counts)
		self.additional_norm, self.additional_lim = generate_multiobject_norm(self.norm_dict, self.object_names.additional, object_counts) if len(self.object_names.additional) > 0 else None, generate_multiobject_norm(self.lim_dict, self.object_names.additional, object_counts) if len(self.object_names.additional) > 0 else None
		self.additional_part_norm, self.additional_part_lim = [self.norm_dict[n] for n in self.object_names.additional] if len(self.object_names.additional) > 0 else None, [self.lim_dict[n] for n in self.object_names.additional] if len(self.object_names.additional) > 0 else None

		# parent relative norm
		paddi_vals = [generate_relative_norm(self.norm_dict[n], self.norm_dict[self.object_names.primary_parent]) for n in self.object_names.additional] if len(self.object_names.additional) > 0 else None
		if paddi_vals is None: self.paddi_part_norm, self.paddi_part_lim = None, None
		else: self.paddi_part_norm, self.paddi_part_lim = [pv[0] for pv in paddi_vals], [pv[1] for pv in paddi_vals]
		# target relative norm
		taddi_vals = [generate_relative_norm(self.norm_dict[n], self.norm_dict[self.object_names.target]) for n in self.object_names.additional] if len(self.object_names.additional) > 0 else None
		if taddi_vals is None: self.taddi_part_norm, self.taddi_part_lim = None, None
		else: self.taddi_part_norm, self.taddi_part_lim = [tv[0] for tv in taddi_vals], [tv[1] for tv in taddi_vals]
		
		self.difference_norm, self.difference_lim = (np.zeros(self.norm_dict[self.object_names.target][0].shape), self.norm_dict[self.object_names.target][1] * 2), (self.target_lim[0] - self.target_lim[1], self.target_lim[0] + self.target_lim[1])
		# relative norm is between the parent and the target
		self.relative_norm, self.relative_lim = generate_relative_norm(self.norm_dict[self.object_names.primary_parent], self.norm_dict[self.object_names.target])

		# gets the appropriate normalization values based on the target
		self.norm_forms = {"target": self.target_norm, "inter": self.inter_norm, "parent": self.parent_norm, "additional": self.additional_norm, "additional_part": self.additional_part_norm, "diff": self.difference_norm, "dyn": self.dynamics_norm, "paddi": self.paddi_part_norm, "taddi": self.taddi_part_norm, "rel": self.relative_norm, "raw": self.raw_norm}
		self.lim_forms = {"target": self.target_lim, "inter": self.inter_lim, "parent": self.parent_lim, "additional": self.additional_lim, "additional_part": self.additional_part_lim, "diff": self.difference_lim, "dyn": self.dynamics_lim, "paddi": self.paddi_part_norm, "taddi": self.taddi_part_norm, "rel": self.relative_lim, "raw": self.raw_lim}
		# TODO: handle relative norm between block and obstacles (internal relative?)

	def get_mean_var(self, form, idxes):
		# logic for partial additional uss an integer after "additional"
		if form.find("additional") != -1 and form != "additional": norm = self.norm_forms["additional_part"][int(form[len("additional"):])]
		elif form.find("taddi") != -1: norm = self.norm_forms["taddi"][int(form[len("taddi"):])]
		elif form.find("paddi") != -1: norm = self.norm_forms["paddi"][int(form[len("paddi"):])]
		else: norm = self.norm_forms[form]
		if norm is None: return None, None
		mean = norm[0][idxes] if idxes is not None else norm[0]
		var = norm[1][idxes] if idxes is not None else norm[1]
		return mean, var

	def __call__(self, state, form="target", idxes=None, name=None):
		'''
		takes the normalization of the state, the form decides which norm to use
		valid forms: target, inter, parent, difference, relative
		Name is to match the signature of full models and does nothing
		'''
		mean, var = self.get_mean_var(form, idxes)
		# print(state, mean,var, form)
		if mean is None: return state
		return compute_norm(mean, var, state)

	def reverse(self, state, form = "target", idxes=None, name=None):
		mean, var = self.get_mean_var(form, idxes)
		# print(state, mean, var, form)
		if mean is None: return state
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target"):
		if form.find("additional") != -1 and form != "additional": lims = self.lim_forms["additional_part"][int(form[len("additional"):])]
		elif form.find("taddi") != -1: lims = self.lim_forms["taddi"][int(form[len("taddi"):])]
		elif form.find("paddi") != -1: lims = self.lim_forms["paddi"][int(form[len("paddi"):])]
		else: lims = self.lim_forms[form]
		return np.clip(state, lims[0], lims[1])

class MappedNorm(): # performs normalization for a masked out component
	def __init__(self, lim_dict, dynamics_dict, target, mask):
		self.lim_dict = lim_dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		mask = mask.astype(bool)
		self.mapped_norm, self.mapped_lim = (self.norm_dict[target][0][mask], self.norm_dict[target][1][mask]), (self.lim_dict[target][0][mask], self.lim_dict[target][1][mask])
		self.mapped_dynamics = dynamics_dict[target][1][mask] # assumes dynamics dict is symetric
		self.mask = mask
		self.norm = self.__call__

	def __call__(self, state):
		mean = self.mapped_norm[0]
		var = self.mapped_norm[1]
		return compute_norm(mean, var, state)

	def reverse(self, state):
		mean = self.mapped_norm[0]
		var = self.mapped_norm[1]
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target"):
		return np.clip(state, self.mapped_lim[0], self.mapped_lim[1])
