from State.full_selector import FullSelector
from State.pad_selector import PadSelector

class CausalFullExtractor():
	def __init__(self, environment, use_padding):
		self.names = environment.object_names
		self.inter_selector = FullSelector(environment.object_sizes, environment.object_instanced, environment.object_names)
		self.target_selectors = {n: construct_object_selector([n], environment) for n in self.names}

		# padi stands for passive_additonal
		self.parent_size = self.inter_selector.output_size()
		self.object_size = self.inter_selector.pad_size

	def _get_dims(self, environment):
		return self.object_size, self.target_size

	def get_selectors(self):
		return self.target_selector, self.inter_selector


class CausalPadExtractor():
	def __init__(self, environment, use_padding):
		self.names = environment.object_names
		self.inter_selector = PadSelector(environment.object_sizes, environment.object_instanced, environment.object_names, {n: np.ones(environment.object_sizes[n]).astype(bool) for n in environment.object_names})
		self.target_selectors = {n: construct_object_selector([n], environment, pad=True) for n in self.names}

		# padi stands for passive_additonal
		self.parent_size = self.inter_selector.output_size()
		self.object_size = self.inter_selector.pad_size

	def _get_dims(self, environment):
		return self.object_size, self.target_size

	def get_selectors(self):
		return self.target_selector, self.inter_selector