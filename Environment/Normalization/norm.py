
class NormalizationModule():
	def __init__(self, lim_dict, object_names):
		self.lim_dict = lim_dict
		# convert min and max in lim_dict to mean and range/2 in norm dict
		self.norm_dict = {n: ((self.lim_dict[n][1] + self.lim_dict[n][0])/2, (self.lim_dict[n][1] - self.lim_dict[n][0])/2) for n in lim_dict.keys()}
		self.object_names = object_names

		self.raw_norm = (128, 128) # assumes images are ranged 256
		self.target_norm = self.norm_dict[self.object_names.target]
		self.inter_norm = np.concatenate([self.norm_dict[n] for n in self.object_names.parent + self.object_names.target], axis=-1)
		self.difference_norm = (np.zeros(self.norm_dict[self.object_names.target][0].shape), self.norm_dict[self.object_names.target][1] * 2) 
		self.relative_norm = (self.norm_dict[self.object_names.primary_parent][0] - self.norm_dict[self.object_names.target][0], self.norm_dict[self.object_names.primary_parent][1] + self.norm_dict[self.object_names.target][1])
		# TODO: handle relative norm between block and obstacles (internal relative?)

def __call__(self, state, form="target"):
	'''
	takes the normalization of the state, the form decides which norm to use
	valid forms: target, inter, parent, difference, relative
	'''
	if form == "iarget": norm = self.target_norm
	elif form == "inter": norm = self.inter_norm
	elif form == "diff": norm = self.difference_norm
	elif form == "rel": norm = self.relative_norm
	elif form == "raw": norm = self.raw_norm
	if len(state) == len(norm[0]): # not multiple instanced
		return (state - norm[0]) / norm[1]
	mean, var = broadcast(norm[0], len(state) // len(norm[1])),  broadcast(norm[1], len(state) // len(norm[1]))
	return state - mean / var


def reverse(self, state, form = "target"):
	if form == "iarget": norm = self.target_norm
	elif form == "inter": norm = self.inter_norm
	elif form == "diff": norm = self.difference_norm
	elif form == "rel": norm = self.relative_norm
	elif form == "raw": norm = self.raw_norm
	if len(state) == len(norm[0]): # not multiple instanced
		return state * norm[1] + norm[0]
	mean, var = broadcast(norm[0], len(state) // len(norm[1])),  broadcast(norm[1], len(state) // len(norm[1]))
	return state * var + mean