# interact selectors
import numpy as np
import copy
from State.feature_selector import construct_object_selector

def construct_selectors(object_names, environment):
    target_select = construct_object_selector([object_names.target], environment)
    parent_selectors = {p: construct_object_selector([p], environment) for p in object_names.parents}
    additional_select = construct_object_selector(object_names.additional, environment) if len(object_names.additional) > 0 else None
    parent_select = construct_object_selector(object_names.parents, environment)
    inter_select = construct_object_selector(object_names.parents + [object_names.target], environment)
    return target_select, parent_selectors, additional_select, parent_select, inter_select

class FullExtractor():
	def __init__(self, target_name, environment):
		self.name = target_name
		self.name_order = environment.object_names
		self.target_index = environment.object_names.index(self.name)
		self.inter_selector = construct_object_selector(self.name_order, environment)
		self.parents = copy.deepcopy(self.name_order).pop(self.target_index)
		self.parent_selector = construct_object_selector(self.name_order)
		self.target_selector = construct_object_selector([self.name], environment)

		self.total_target_size = int(self.max_target * environment.object_sizes[self.names.target])
		self.total_parent_size = int(np.sum([environment.object_instanced[n] * environment.object_sizes[n] for n in self.parents]))
		self.total_inter_size = int(np.sum([environment.object_instanced[n] * environment.object_sizes[n] for n in self.name_order]))
		self.total_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in self.name_order]
		self.complete_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in self.name_order]

		self.target_dim, self.object_dims = self._get_dims(environment)

	def _get_dims(self, environment):
		return environment.object_sizes[self.names.target], [environment.object_sizes[n] for n in self.name_order]

	def get_selectors(self):
		return self.inter_selector, self.parent_selector, self.target_selector

class CausalExtractor():
	def __init__(self, object_names, environment):
		all_names = object_names.parents + [object_names.target]
		self.names = object_names
		self.multi_instanced = [n for n in all_names if environment.object_instanced[n] > 1]
		self.single_instanced = [n for n in all_names if environment.object_instanced[n] == 1]
		self.target_instanced = environment.object_instanced[object_names.target] > 1

		self.passive_additional = list()
		self.padi_single = list()
		self.active = list() # all active components, single instanced first followed by multi instanced
		self.active_multi = list() # multi-instanced active
		self.num_parents = list()
		for n in self.names.parents[1:]: # single instanced always precedes multi-instanced, target preferred last
			if n in self.multi_instanced:
				self.passive_additional.append(n)
				self.active_multi.append(n)
				self.num_parents.append(environment.object_instanced[n])
			else:
				self.padi_single.append(n)
				self.active.append(n)
		self.additional = copy.deepcopy(self.active + self.active_multi)
		if self.names.parents[0] in self.multi_instanced:
			self.active_multi = [self.names.parents[0]] + self.active_multi
			self.num_parents = [environment.object_instanced[n]] + self.num_parents			
		else:
			self.active = [self.names.parents[0]] + self.active
		self.max_parent_multi = np.sum(self.num_parents) - int(self.names.target in self.names.parents) # assumes self removal
		self.num_parents = [1 for p in self.names.parents if p in self.single_instanced] + self.num_parents
		self.names.parents = copy.deepcopy(self.active + self.active_multi)
		if self.names.target in self.multi_instanced:
			self.passive_additional.append(self.names.target)
			self.active_multi.append(self.names.target)
		else:
			self.padi_single.append(self.names.target)
			self.active.append(self.names.target)
		self.passive_additional = self.padi_single + self.passive_additional
		self.active = self.active + self.active_multi

		self.max_additional = [environment.object_instanced[n] for n in self.additional]
		self.max_target = environment.object_instanced[object_names.target]
		self.max_parent = environment.object_instanced[object_names.primary_parent]

		self.inter_selector = construct_object_selector(self.active, environment)
		self.full_parent_selector = construct_object_selector(self.names.parents, environment)
		self.parent_selector = construct_object_selector([self.names.parents[0]], environment)
		self.additional_selector = construct_object_selector(self.names.additional, environment)
		self.additional_selectors = [construct_object_selector([a], environment) for a in self.names.additional]
		self.padi_selector = construct_object_selector(self.passive_additional, environment)
		self.padi_single_selector = construct_object_selector(self.padi_single, environment)
		self.single_selector = construct_object_selector(self.single_instanced, environment)
		self.multi_selectors = [construct_object_selector([ms], environment) for ms in self.multi_instanced]
		self.target_selector = construct_object_selector([object_names.target], environment)

		self.complete_instances = [int(environment.object_instanced[n]) for n in environment.object_names]
		self.pad_dim = max(list(environment.object_sizes.values()))
		self.append_dim =len(list(environment.object_sizes.keys()))
		self.expand_dim = self.pad_dim + self.append_dim
		self.complete_object_sizes = [int(environment.object_instanced[n] * self.expand_dim) for n in environment.object_names]

		self.total_target_size = int(self.max_target * environment.object_sizes[self.names.target])
		self.total_inter_size = int(np.sum([environment.object_instanced[n] * environment.object_sizes[n] for n in self.names.parents]) + self.total_target_size )


		# padi stands for passive_additonal
		self.padi_first_obj_dim, self.first_obj_dim, \
			self.target_dim, self.object_dims, self.padi_object_dims = self._get_dims(environment)
		self.total_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in environment.object_names]
		self.complete_object_sizes = [int(environment.object_instanced[n] * environment.object_sizes[n]) for n in environment.object_names]
		self.parent_size = environment.object_sizes[self.names.parents[0]]
		self.additional_sizes = [environment.object_sizes[n] for n in self.additional]

	def _get_dims(self, environment):
		first_obj_dim = self.single_selector.output_size()
		padi_first_obj_dim = self.padi_single
		target_dim = environment.object_sizes[self.names.target]
		object_dims = [environment.object_sizes[n] for n in self.names.parents if n in self.multi_instanced] 
		padi_object_dims = [environment.object_sizes[n] for n in self.names.parents[1:] if n in self.multi_instanced]

		if len(object_dims) > 0: assert np.all([od == object_dims[0] for od in object_dims]) # all multi-instanced object sizes should be the same for pointnets to work
		# returns padi first, first, target, object_dim, padi_object_dim
		return padi_first_obj_dim, first_obj_dim, target_dim, object_dims[0] if len(object_dims) > 0 else 0, padi_object_dims[0] if len(padi_object_dims) > 0 else 0

	def get_selectors(self):
		return self.target_selector, self.full_parent_selector, self.additional_selector, self.additional_selectors, self.padi_selector, self.parent_selector, self.inter_selector