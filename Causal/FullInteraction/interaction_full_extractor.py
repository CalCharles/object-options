from State.full_selector import FullSelector
from State.pad_selector import PadSelector
from State.feature_selector import construct_object_selector
import numpy as np

class CausalFullExtractor():
	def __init__(self, environment, append_id=False):
		self.names = environment.object_names
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
	def __init__(self, environment, append_id):
		self.names = environment.object_names
		self.inter_selector = PadSelector(environment.object_sizes, environment.object_instanced, environment.object_names, {n: np.ones(environment.object_sizes[n]).astype(bool) for n in environment.object_names}, append_id=append_id)
		self.target_selectors = {n: construct_object_selector([n], environment, pad=True, append_id=False) for n in self.names} # don't append ID for target state
		self.full_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in self.names]
		self.complete_instances = [int(environment.object_instanced[n]) for n in environment.object_names]
		self.pad_dim = max(list(environment.object_sizes.values()))
		self.append_dim =len(list(environment.object_sizes.keys())) * int(append_id)
		self.expand_dim = self.pad_dim + self.append_dim
		self.complete_object_sizes = [int(environment.object_instanced[n] * self.expand_dim) for n in self.names]

		# sizes
		self.total_inter_size = self.inter_selector.output_size()
		self.object_size = self.inter_selector.append_pad_size
		self.single_object_size = self.target_selectors[self.names[0]].append_pad_size
		self.total_target_sizes = {n: self.single_object_size * environment.object_instanced[n] for n in self.names}

	def _get_dims(self, name):
		return self.object_size, self.single_object_size, self.total_target_sizes[name]

	def get_selectors(self):
		return self.target_selectors, self.inter_selector