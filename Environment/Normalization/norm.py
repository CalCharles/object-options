
class NormalizationModule():
	def __init__(self, lim_dict, object_names):
		self.lim_dict = lim_dict
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2) for n in lim_dict.keys()}
		self.object_names = object_names

		self.raw_norm, self.raw_lim = (128, 128), (0, 256) # assumes images are ranged 256
		self.target_norm, self.target_lim = self.norm_dict[self.object_names.target], self.lim_dict[self.object_names.target]
		inter_names = self.object_names.parent + self.object_names.target
		self.inter_norm, self.inter_lim = np.concatenate([self.norm_dict[n] for n in inter_names], axis=-1), np.concatenate([self.lim_dict[n] for n in inter_names], axis=-1)
		self.difference_norm, self.difference_lim = (np.zeros(self.norm_dict[self.object_names.target][0].shape), self.norm_dict[self.object_names.target][1] * 2) 
		self.relative_norm = (self.norm_dict[self.object_names.primary_parent][0] - self.norm_dict[self.object_names.target][0], self.norm_dict[self.object_names.primary_parent][1] + self.norm_dict[self.object_names.target][1])
		self.relative_lim = (self.relative_norm[0]-self.relative_norm[1], self.relative_norm[0] + self.relative_norm[1])
		self.norm_forms = {"target": self.target_norm, "inter": self.inter_norm, "diff": self.difference_norm, "rel": self.relative_norm, "raw": self.raw_norm}
		self.lim_forms = {"target": self.target_lim, "inter": self.inter_lim, "diff": self.difference_lim, "rel": self.relative_lim, "raw": self.raw_lim}
		# TODO: handle relative norm between block and obstacles (internal relative?)

def __call__(self, state, form="target"):
	'''
	takes the normalization of the state, the form decides which norm to use
	valid forms: target, inter, parent, difference, relative
	'''
	norm = self.norm_forms[form]
	if len(state) == len(norm[0]): # not multiple instanced
		return (state - norm[0]) / norm[1]
	mean, var = broadcast(norm[0], len(state) // len(norm[1])),  broadcast(norm[1], len(state) // len(norm[1]))
	return state - mean / var

def reverse(self, state, form = "target"):
	norm = self.norm_forms[form]
	if len(state) == len(norm[0]): # not multiple instanced
		return state * norm[1] + norm[0]
	mean, var = broadcast(norm[0], len(state) // len(norm[1])),  broadcast(norm[1], len(state) // len(norm[1]))
	return state * var + mean

def clip(self, state, form="target"):
	lims = self.lim_forms[form]
	return np.clip(state, lims[0], lims[1])