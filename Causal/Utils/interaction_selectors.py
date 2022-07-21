# interact selectors

def construct_selectors(object_names, environment):
    target_select = construct_object_selector([object_names.target], environment)
    parent_selectors = {p: construct_object_selector([p], environment) for p in object_names.parents}
    additional_select = construct_object_selector(object_names.additional, environment) if len(object_names.additional) > 0 else None
    parent_select = construct_object_selector(object_names.parents, environment)
    inter_select = construct_object_selector(object_names.parents + [object_names.target], environment)
    return target_select, parent_selectors, additional_select, parent_select, inter_select


class CausalExtractor():
	def __init__(self, object_names, environment):
		all_names = object_names.parents + [object_names.target]
		self.names = object_names
		self.multi_instanced = [n for name in all_names if environment.object_instanced[name] > 1]
		self.single_instanced = [n for name in all_names if environment.object_instanced[name] == 1]
		self.target_instanced = environment.object_instanced[object_names.target] > 1

		self.single_selector = construct_object_selector(self.single_instanced, environment)
		self.multi_selectors = [construct_object_selector(ms, environment) for ms in self.multi_instanced]
		self.target_selector = construct_object_selector(object_names.target, environment)

		self.first_obj_dim, self.target_dim, self.object_dims, self.object_dim = self._get_dims()

	def _get_dims(self):
		first_obj_dim = self.single_selector.output_size()
		target_dim = environment.

	def get_passive(self):
		return
		
	def get_active(self):
		return