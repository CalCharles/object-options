from State.full_selector import FullSelector
from State.pad_selector import PadSelector
from State.feature_selector import construct_object_selector
import numpy as np
import copy

class CausalFullExtractor():
	def __init__(self, environment, append_id=False):
		self.names = environment.object_names
		object_sizes, object_instanced, object_names = copy.deepcopy(environment.object_sizes), copy.deepcopy(environment.object_instanced), copy.deepcopy(environment.object_names)
		for n in ["Reward", "Done"]:
			del object_sizes[n]
			del object_instanced[n]
			object_names.remove(n)
		self.inter_selector = FullSelector(environment.object_sizes, environment.object_instanced, environment.object_names)
		
		self.target_selectors = {n: construct_object_selector([n], environment) for n in self.names}
		self.full_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in self.names]
		self.complete_instances = [int(environment.object_instanced[n]) for n in environment.object_names]
		self.pad_dim = max(list(environment.object_sizes.values()))
		self.append_dim =len(list(environment.object_sizes.keys())) * int(append_id)
		self.expand_dim = self.pad_dim + self.append_dim
		self.complete_object_sizes = [int(environment.object_instanced[n] * self.expand_dim) for n in self.names]

		# padi stands for passive_additonal
		self.parent_size = self.inter_selector.output_size()
		self.object_sizes = {n: self.inter_selector.pad_size for n in self.names}
		self.total_target_sizes = {n: self.object_size * environment.object_instanced[n] for n in self.names}

	def _get_dims(self, name):
		return self.object_size, self.object_size, self.total_target_sizes[name]

	def get_selectors(self):
		return self.target_selectors, self.inter_selector


class CausalPadExtractor():
	def __init__(self, environment, append_id, no_objects = True):
		self.names = environment.object_names
		self.all_names = copy.deepcopy(environment.all_names)
		object_sizes, object_instanced, object_names = copy.deepcopy(environment.object_sizes), copy.deepcopy(environment.object_instanced), copy.deepcopy(environment.object_names)

		if no_objects:
			for n in ["Reward", "Done"]:
				del object_sizes[n]
				del object_instanced[n]
				object_names.remove(n)
				self.all_names.remove(n)
		self.object_names = object_names

		use_names = self.names
		self.inter_selector = PadSelector(object_sizes, object_instanced, object_names, {n: np.ones(object_sizes[n]).astype(bool) for n in object_names}, append_id=append_id)
		self.unpad_inter_selector = FullSelector(environment.object_sizes, environment.object_instanced, environment.object_names)
		self.target_selectors = {n: construct_object_selector([n], environment, pad=True, append_id=False) for n in use_names} # don't append ID for target state
		self.target_select = construct_object_selector(use_names, environment, pad=True, append_id=False)
		self.full_object_sizes = [int(object_instanced[n] * object_sizes[n]) for n in self.object_names]
		self.complete_instances = [int(object_instanced[n]) for n in self.object_names]
		self.num_instances = [int(environment.object_instanced[n]) for n in use_names]
		self.pad_dim = max(list(object_sizes.values()))
		self.append_dim =len(list(object_sizes.keys())) * int(append_id)
		self.expand_dim = self.pad_dim + self.append_dim
		self.complete_object_sizes = [int(object_instanced[n] * self.expand_dim) for n in self.object_names]

		# sizes
		self.total_inter_size = self.inter_selector.output_size()
		self.object_size = self.inter_selector.append_pad_size
		self.single_object_size = self.target_selectors[use_names[0]].append_pad_size
		self.total_target_sizes = {n: self.single_object_size * environment.object_instanced[n] for n in use_names}

	def get_padding(self, state, name):
		if name is None or name == "all":
			return self.target_select.get_padding(state)
		return self.target_selectors[name].get_padding(state)

	def _get_dims(self, name):
		if name is None or name == "all":
			return self.object_size, self.single_object_size, self.object_size * np.sum(self.complete_instances)
		print(self.total_target_sizes)
		return self.object_size, self.single_object_size, self.total_target_sizes[name]

	def get_selectors(self, all=False):
		if all: return self.target_select, self.inter_selector
		return self.target_selectors, self.inter_selector
	
	def reverse_extract(self, state, target=True):
		if target: return {n: state[...,i*self.pad_dim:(i+1)*self.pad_dim] for i, n in enumerate(self.all_names)}
		return {n: state[...,i*self.expand_dim:(i+1)*self.expand_dim] for i, n in enumerate(self.all_names)}
