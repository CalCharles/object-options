import numpy as np
from State.feature_selector import broadcast
from Network.network_utils import pytorch_model

def compute_norm(mean, var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return (state - mean) / var
	mean, var = broadcast(mean, len(state) // len(var)),  broadcast(var, len(state) // len(var))
	return state - mean / var

def compute_reverse(mean,var, state):
	state = pytorch_model.unwrap(state)
	if state.shape[-1] == len(mean): # not multiple instanced
		return state * var + mean
	mean, var = broadcast(mean, len(state) // len(var)),  broadcast(var, len(state) // len(var))
	return state * var + mean

class NormalizationModule():
	def __init__(self, lim_dict, dynamics_dict, object_names):
		self.lim_dict = lim_dict # the bounds of positions for where an object can be
		self.dynamics_dict = dynamics_dict # the bounds for the amount an object can change in a single timestep
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.dynamics_norm_dict = {n: ((self.dynamics_dict[n][1] + self.dynamics_dict[n][0])/2, (self.dynamics_dict[n][1] - self.dynamics_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		self.object_names = object_names

		# specify different norms
		self.raw_norm, self.raw_lim = (128, 128), (0, 256) # assumes images are ranged 256
		self.target_norm, self.target_lim = self.norm_dict[self.object_names.target], self.lim_dict[self.object_names.target]
		self.parent_norm, self.parent_lim = self.norm_dict[self.object_names.primary_parent], self.lim_dict[self.object_names.primary_parent]
		self.dynamics_norm, self.dynamics_lim = self.dynamics_norm_dict[self.object_names.target], self.dynamics_dict[self.object_names.target]
		# interaction state norm
		inter_names = self.object_names.parents + [self.object_names.target]
		self.inter_norm, self.inter_lim = np.concatenate([self.norm_dict[n] for n in inter_names], axis=-1), np.concatenate([self.lim_dict[n] for n in inter_names], axis=-1)
		self.additional_norm, self.additional_lim = np.concatenate([self.norm_dict[n] for n in self.object_names.additional], axis=-1) if len(self.object_names.additional) > 0 else None, np.concatenate([self.lim_dict[n] for n in self.object_names.additional], axis=-1) if len(self.object_names.additional) > 0 else None
		
		self.difference_norm, self.difference_lim = (np.zeros(self.norm_dict[self.object_names.target][0].shape), self.norm_dict[self.object_names.target][1] * 2), (self.target_lim[0] - self.target_lim[1], self.target_lim[0] + self.target_lim[1])
		# relative norm is between the parent and the target
		self.relative_norm = (self.norm_dict[self.object_names.primary_parent][0] - self.norm_dict[self.object_names.target][0], self.norm_dict[self.object_names.primary_parent][1] + self.norm_dict[self.object_names.target][1]  + 1e-6)
		self.relative_lim = (self.relative_norm[0]-self.relative_norm[1], self.relative_norm[0] + self.relative_norm[1])
		# gets the appropriate normalization values based on the target
		self.norm_forms = {"target": self.target_norm, "inter": self.inter_norm, "parent": self.parent_norm, "additional": self.additional_norm, "diff": self.difference_norm, "dyn": self.dynamics_norm, "rel": self.relative_norm, "raw": self.raw_norm}
		self.lim_forms = {"target": self.target_lim, "inter": self.inter_lim, "parent": self.parent_lim, "additional": self.additional_lim, "diff": self.difference_lim, "dyn": self.dynamics_lim, "rel": self.relative_lim, "raw": self.raw_lim}
		# TODO: handle relative norm between block and obstacles (internal relative?)

	def __call__(self, state, form="target", idxes=None):
		'''
		takes the normalization of the state, the form decides which norm to use
		valid forms: target, inter, parent, difference, relative
		'''
		norm = self.norm_forms[form]
		if norm is None: return state
		mean = norm[0][idxes] if idxes is not None else norm[0]
		var = norm[1][idxes] if idxes is not None else norm[1]
		return compute_norm(mean, var, state)

	def reverse(self, state, form = "target", idxes=None):
		norm = self.norm_forms[form]
		if norm is None: return state
		mean = norm[0][idxes] if idxes is not None else norm[0]
		var = norm[1][idxes] if idxes is not None else norm[1]
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target"):
		lims = self.lim_forms[form]
		return np.clip(state, lims[0], lims[1])



class MappedNorm(): # performs normalization for a masked out component
	def __init__(self, lim_dict, dynamics_dict, target, mask):
		self.lim_dict = lim_dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2 + 1e-6) for n in lim_dict.keys()}
		mask = mask.astype(bool)
		self.mapped_norm, self.mapped_lim = (self.norm_dict[target][0][mask], self.norm_dict[target][1][mask]), (self.lim_dict[target][0][mask], self.lim_dict[target][1][mask])
		self.mask = mask

	def norm(self, state):
		mean = self.norm_dict[0]
		var = self.norm_dict[1]
		return compute_norm(mean, var, state)

	def reverse(self, state):
		mean = self.norm_dict[0]
		var = self.norm_dict[1]
		return compute_reverse(mean, var, state)

	def clip(self, state, form="target"):
		return np.clip(state, self.mapped_lim[0], self.mapped_lim[1])
