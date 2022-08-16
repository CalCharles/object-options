import numpy as np
from State.feature_selector import broadcast

_SINGLE_NAMES = ["param", "parent", "additional", "target", "inter", "diff"]
_RELATIVE_NAMES = ["parent_relative", "parent_additional", "additional_relative", "parent_param", "param_relative"]
COMPONENT_NAMES = _SINGLE_NAMES + _RELATIVE_NAMES

def get_norm_form(name):
    _norm_forms = {"param": "target", "parent": "parent", "additional": "additional", "target": "target", "inter": "inter", "diff": "dyn", 
                        "parent_relative": "rel", "additional_relative":"taddi", "parent_additional":"paddi", "parent_param": "rel", "param_relative": "diff"}
    if name.find("additional_relative") != -1:
        return _norm_forms["additional_relative"] + name[len("additional_relative"):]
    if name.find("parent_additional") != -1:
        return _norm_forms["parent_additional"] + name[len("parent_additional"):]
    if name.find("additional") != -1:
        return name
    return _norm_forms[name]


class ObservationExtractor(): 
# an internal class for state extractor to handle the observation-related components
    def __init__(self, state_extractor):
        # class variables for the order of elements to append, normalizations to apply
        self._component_names = COMPONENT_NAMES # TODO: add additional state
        
        # environment hyperparameters
        self.target_size = state_extractor.target_size
        self.parent_size = state_extractor.parent_size
        self.inter_size = state_extractor.inter_size
        self.additional_sizes = state_extractor.additional_sizes
        self.max_target_objects = state_extractor.max_target_objects
        self.max_parent_objects = state_extractor.max_parent_objects
        self.max_additional_objects = state_extractor.max_additional_objects
        self.max_partar = max(self.max_target_objects, self.max_parent_objects)
        self.max_paraddi = max(np.max(self.max_additional_objects), self.max_parent_objects)
        self.parent_select = state_extractor.parent_select
        self.target_select = state_extractor.target_select
        self.inter_select = state_extractor.inter_select
        self.additional_select = state_extractor.additional_select
        self.additional_selectors = state_extractor.additional_selectors
        self.state_extractor = state_extractor

        # observation values
        self.single_obs_setting = state_extractor.single_obs_setting
        self.relative_obs_setting = state_extractor.relative_obs_setting
        self.obs_setting = self.single_obs_setting + self.relative_obs_setting
        self.norm = state_extractor.norm
        self.size_index = {"param": self.target_size, "inter": self.inter_size, "parent": self.parent_size * self.max_parent_objects, 
                    "additional": [self.additional_sizes[i] * self.max_additional_objects[i] for i in range(len(self.additional_sizes))],
                    "target": self.target_size * self.max_target_objects, "inter": self.inter_size, "diff": self.target_size * self.max_target_objects,
                    "parent_relative": self.target_size * self.max_partar, "additional_relative": [self.target_size * max(self.max_target_objects, mao) for mao in self.max_additional_objects],
                    "parent_additional": [self.parent_size * max(self.max_parent_objects, mao) for mao in self.max_additional_objects], "parent_param": self.target_size * self.max_parent_objects, "param_relative": self.target_size * self.max_target_objects}
        self.single_size_index = {"param": self.target_size, "inter": self.inter_size, "parent": self.parent_size, 
                    "additional": [self.additional_sizes[i] for i in range(len(self.additional_sizes))],
                    "target": self.target_size, "inter": self.inter_size, "diff": self.target_size,
                    "parent_relative": self.target_size, "additional_relative": [self.target_size for mao in self.max_additional_objects],
                    "parent_additional": [self.parent_size for mao in self.max_additional_objects], "parent_param": self.target_size, "param_relative": self.target_size}
        param, parent, additional, target, inter, diff = self.single_obs_setting
        parent_relative, parent_additional, additional_relative, parent_param, param_relative = self.relative_obs_setting
        self.name_include = {"param": param, "parent": parent, "additional": additional, "target": target, "inter": inter, "diff": diff,
                        "parent_relative": parent_relative, "parent_additional": parent_additional, "additional_relative": additional_relative, "parent_param": parent_param, "param_relative": param_relative}

        self.param_mask = state_extractor.param_mask
        self.combine_param_mask = state_extractor.combine_param_mask
        self.first_obj_dim, self.parent_obj_dim, self.additional_obj_dim, self.target_obj_dim, self.rel_obj_dim, self.obj_dim, self.post_dim = self._get_dims()
        self.multiobject_order = int(self.target_multi) + int(np.any(self.additional_multi)) + int(self.parent_multi)
        self.single_multiobject = (int(self.target_multi) + int(np.any(self.additional_multi)) + int(self.parent_multi)) == 1

        self.max_taddi_objects = [self.max_target_objects for i in range(len(self.additional_sizes))] if self.single_multiobject and self.target_multi else self.max_additional_objects
        self.construct_name_order()

    def _get_dims(self): # gets the first object dimension and object dimension for multi object-networks
        param, parent, additional, target, inter, diff = self.single_obs_setting
        parent_relative, parent_additional, additional_relative, parent_param, param_relative = self.relative_obs_setting
        self.parent_multi = self.max_parent_objects > 1
        self.target_multi = self.max_target_objects > 1
        either_multi = (self.parent_multi or self.target_multi)
        self.additional_multi = (self.max_additional_objects > 1)

        # first_obj_dim is all single element components
        first_obj_dim =  int(param * self.target_size 
                    + parent * int(not self.parent_multi) * self.parent_size
                    + additional * np.sum(self.additional_sizes * (self.max_additional_objects == 1).astype(int))
                    + target * int(not self.target_multi) * self.target_size
                    + inter * self.inter_size # in general, don't use inter because it does not handle multiinstanced
                    + diff * int(not self.target_multi) * self.target_size
                    + parent_relative * int(not either_multi) * self.target_size
                    + parent_additional * int(not self.parent_multi) * np.sum((self.max_additional_objects == 1).astype(int) * self.parent_size)
                    + additional_relative * int(not self.target_multi) * np.sum((self.max_additional_objects == 1).astype(int) * self.target_size)
                    + parent_param * int(not self.parent_multi) * self.target_size # parent_size == target size for this to work
                    + param_relative * int(not self.target_multi) * self.target_size)

        parent_obj_dim = int(parent * int(self.parent_multi) * self.parent_size
                    + parent_param * int(self.parent_multi) * self.parent_size)
        additional_obj_dims = np.array([int(additional * int(self.additional_multi[i]) * self.additional_sizes[i]) for i in range(len(self.additional_multi))]) 
        target_obj_dim = int(target * int(self.target_multi) * self.target_size
                    + diff * int(self.target_multi) * self.target_size
                    + param_relative * int(self.target_multi) * self.target_size)
        rel_obj_dim = int(np.max([parent_relative * either_multi * self.target_size]
                    + [additional_relative * (self.additional_multi[i] or self.target_multi) * self.target_size for i in range(len(self.additional_multi))]
                    + [parent_additional * (self.additional_multi[i] or self.parent_multi) * self.parent_size for i in range(len(self.additional_multi))]))
        obj_dim = int(np.max([target_obj_dim + rel_obj_dim] + [parent_obj_dim + rel_obj_dim] + [a + rel_obj_dim for a in additional_obj_dims])) # obj_dim is invalid for both_multi
        post_dim = self.target_size # only param in post dim for now
        print(first_obj_dim, obj_dim)
        return first_obj_dim, parent_obj_dim, (np.max(additional_obj_dims) if len(additional_obj_dims) > 0 else 0), target_obj_dim, rel_obj_dim, obj_dim, post_dim

    def construct_name_order(self):
        # TODO: only 3 multi-object supported at the same time
        self.parent_multi_list = list()
        self.single_order = ["param", "inter"] # param is always first, followed by parent components, additional components, and target components
        if self.parent_multi:
            self.parent_multi_list = ["parent", "parent_param"]
        else:
            self.single_order += ["parent", "parent_param"]

        if self.multiobject_order == 0:
            self.single_order += ["parent_additional" + str(i) for i in range(len(self.additional_multi))]

        self.additional_multi_list = list() # should be at most length 1
        maidx = -1
        for i in range(len(self.additional_multi)):
            if self.additional_multi[i]:
                self.additional_multi_list = ["additional"+str(i)]
                maidx = i
            else:
                self.single_order +=["additional"+str(i)]


        self.target_multi_list = list()
        if self.target_multi:
            self.target_multi_list += ["target", "diff", "param_relative"] 
        else:
            self.single_order += ["target", "diff", "param_relative"]

        self.multi_order = list()
        self.multi_order_count = 0
        self.multi_second_order = list()
        self.multi_second_order_count = 0
        self.multi_third_order = list()
        self.multi_third_order_count = 0
        if self.single_multiobject: 
            # relative values are only usable with a single number value
            if self.target_multi:
                self.multi_order = self.target_multi_list + ["parent_relative"] + ["additional_relative" + str(i) for i in range(len(self.additional_multi))]
                self.multi_order_count = self.max_target_objects
                self.single_order += ["parent_additional" + str(i) for i in range(len(self.additional_multi))]
            elif self.parent_multi:
                self.multi_order = self.parent_multi_list + ["parent_relative"] + ["parent_additional" + str(i) for i in range(len(self.additional_multi))]
                self.multi_order_count = self.max_parent_objects
                self.single_order += ["additional_relative" + str(i) for i in range(len(self.additional_multi))]
            else: # additional_relative
                self.multi_order = self.additional_multi_list + ["additional_relative" + str(maidx)] + ["parent_additional" + str(maidx)]
                self.multi_order_count = int(np.max(self.max_additional_objects))
                self.single_order += ["additional_relative" + str(i) for i in range(len(self.additional_multi)) if maidx != i]
        elif self.multiobject_order == 0:
            self.single_order += ["parent_relative"] + ["additional_relative" + str(i) for i in range(len(self.additional_multi))]
        elif self.multiobject_order == 2:
            self.multi_order = self.target_multi_list if self.target_multi else self.parent_multi_list
            self.multi_order_count = max(self.max_parent_objects, self.max_target_objects)
            self.multi_second_order =  self.parent_multi_list if self.parent_multi else self.additional_multi_list
            self.multi_order_count = max(self.max_parent_objects, int(np.max(self.max_additional_objects)))
        elif self.multiobject_order == 3:
            self.multi_order = self.target_multi_list
            self.multi_order_count = self.max_target_objects
            self.multi_second_order = self.parent_multi_list
            self.multi_second_order_count = self.max_parent_objects
            self.multi_third_order = self.additional_multi_list
            self.multi_third_order_count = int(np.max(self.max_additional_objects))

    def reverse_obs_norm(self, obs, mask=None):
        def add_names(obs, bins, order, combine, start, multi_ordered, num_instances):
            # adds the reverse-norm components of obs, corresponding to the nonzero values in bins, in the order of order
            # appends the outcomes to combine, and starts from start
            size_to = start
            use_size_index = self.single_size_index if multi_ordered else self.size_index
            num_instances = num_instances if multi_ordered else 1
            for i in range(num_instances):
                for use, name in zip(bins, order):
                    if use:
                        if name.find("additional_relative") != -1:
                            increment = use_size_index["additional_relative"][int(name[len("additional_relative"):])]
                        elif name.find("parent_additional") != -1:
                            increment = use_size_index["parent_additional"][int(name[len("parent_additional"):])]                            
                        elif name.find("additional") != -1:
                            increment = use_size_index["additional"][int(name[len("additional"):])]
                        else:
                            increment = use_size_index[name]
                        reversed_val = self.norm.reverse(obs[...,int(size_to):int(size_to + increment)], form=get_norm_form(name))
                        if name.find("param") != -1: reversed_val = reversed_val * mask if mask is not None else reversed_val
                        combined.append(reversed_val)
                        size_to += increment
            return size_to

        combined = list()
        size_to = 0
        size_to = add_names(obs, [self.name_include[s.strip("0123456789")] for s in self.single_order], self.single_order, combined, size_to, False, 1)
        if len(self.multi_order) > 0: add_names(obs, [self.name_include[s.strip("0123456789")] for s in self.multi_order], self.multi_order, combined, size_to, True, self.multi_order_count)
        if len(self.multi_second_order) > 0: add_names(obs, [self.name_include[s.strip("0123456789")] for s in self.multi_second_order], self.multi_second_order, combined, size_to, True, self.multi_second_order_count)
        if len(self.multi_third_order) > 0: add_names(obs, [self.name_include[s.strip("0123456789")] for s in self.multi_third_order], self.multi_third_order, combined, size_to, True, self.multi_third_order_count)
        return np.concatenate(combined, axis=-1)

    def get_state_index(self, last_state, full_state, param, mask, raw=False):
        param_idx, parent, additional, target, inter, diff = self.single_obs_setting
        if len(self.relative_obs_setting) == 4: # TODO: a hack to make the relative obs work
            parent_relative, additional_relative, parent_param, param_relative = self.relative_obs_setting
            parent_additional = 0
        else:
            parent_relative, parent_additional, additional_relative, parent_param, param_relative = self.relative_obs_setting
        state_index = dict()
        if additional or additional_relative or parent_additional: state_index["additional_raw"] = self.state_extractor.get_additional(full_state, partial=True)
        state_index["parent_raw"] = self.state_extractor.get_parent(full_state)
        state_index["target_raw"] = self.state_extractor.get_target(full_state)
        if parent: state_index["parent_norm"] = self.state_extractor.get_parent(full_state, norm=True)
        if additional: state_index["additional_norm"] = self.state_extractor.get_additional(full_state, partial=True, norm=True)
        if target: state_index["target_norm"] = self.state_extractor.get_target(full_state, norm=True)
        if inter: state_index["inter_norm"] = self.state_extractor.get_target(full_state, )
        if diff: state_index["diff_norm"] = differ = self.norm(state_index["target_raw"] - self.target_select(last_state["factored_state"]), form="diff")
        if param_idx: state_index["param"] = param
        if param_idx: state_index["mask"] = mask
        if parent_param: 
            if self.parent_multi:
                broad_param, broad_mask = broadcast(param, self.max_parent_objects, axis=-1), broadcast(mask, self.max_parent_objects, axis=-1)
                state_index["parent_param"] = self.norm((state_index["parent_raw"] - broad_param), form="rel") * broad_mask
            else: state_index["parent_param"] = self.norm(state_index["parent_raw"] - param, form="rel") * mask
        if param_relative:
            if self.target_multi:
                broad_param, broad_mask = broadcast(param, self.max_parent_objects, axis=-1), broadcast(mask, self.max_parent_objects, axis=-1)
                state_index["target_param"] = self.norm((state_index["target_raw"] - broad_param), form="diff") * broad_mask
            else: state_index["target_param"] = self.norm(state_index["target_raw"] - param, form="diff") * mask
        return state_index

    def add_obs(self, form, state_index, idx=-1):
        if form == "param":
            return self.param_mask(state_index["param"], state_index["mask"])
        elif form == "parent":
            if idx == -1: return state_index["parent_norm"]
            return state_index["parent_norm"][...,idx*self.parent_size: (idx+1)*self.parent_size]
        elif form.find("additional_relative") != -1:
            aidx = int(form[len("additional_relative"):])
            if idx == -1: return self.norm(state_index["additional_raw"][aidx] - state_index["target_raw"], form="taddi"+str(aidx))
            if self.additional_multi[aidx]:
                return self.norm(state_index["additional_raw"][aidx][...,idx*self.target_size:(idx+1)*self.target_size] - state_index["target_raw"], form="taddi"+str(aidx))
            elif self.target_multi:
                return self.norm(state_index["additional_raw"][aidx] - state_index["target_raw"][...,idx*self.target_size:(idx+1)*self.target_size], form="taddi"+str(aidx))
        elif form.find("parent_additional") != -1:
            aidx = int(form[len("parent_additional"):])
            if idx == -1: return self.norm(state_index["additional_raw"][aidx] - state_index["parent_raw"], form="paddi" + str(aidx))
            if self.additional_multi[aidx]:
                return self.norm(state_index["additional_raw"][aidx][...,idx*self.target_size:(idx+1)*self.target_size] - state_index["parent_raw"], form="paddi" + str(aidx))
            elif self.target_multi:
                return self.norm(state_index["additional_raw"][aidx] - state_index["parent_raw"][...,idx*self.target_size:(idx+1)*self.target_size], form="paddi" + str(aidx))
        elif form.find("additional") != -1:
            additional_idx = int(form[len("additional"):])
            if idx == -1: return state_index["additional_norm"][additional_idx]
            return state_index["additional_norm"][additional_idx][...,idx*self.target_size: (idx+1)*self.target_size]
        elif form == "target":
            if idx == -1: return state_index["target_norm"]
            return state_index["target_norm"][...,idx*self.target_size: (idx+1)*self.target_size]
        elif form == "inter":
            return state_index["inter_norm"]
        elif form == "diff":
            if idx == -1: return state_index["diff_norm"]
            return state_index["diff_norm"][...,idx*self.target_size: (idx+1)*self.target_size]
        elif form == "parent_relative":
            if idx == -1: return self.norm(state_index["parent_raw"] - state_index["target_raw"], form="rel")
            if self.parent_multi:
                return self.norm(state_index["parent_raw"][...,idx*self.target_size:(idx+1)*self.target_size] - state_index["target_raw"], form="rel")
            elif self.target_multi:
                return self.norm(state_index["parent_raw"] - state_index["target_raw"][...,idx*self.target_size:(idx+1)*self.target_size], form="rel")
        elif form == "parent_param": 
            if idx == -1: return self.norm((state_index["parent_raw"] - state_index["param"]) * state_index["mask"], form="rel")
            return state_index["parent_param"][...,idx*self.target_size:(idx+1)*self.target_size]
        elif form == "param_relative":
            if idx == -1: return self.norm((state_index['target_raw'] - state_index['param']) * state_index["mask"] , form="diff")
            return state_index["target_param"][...,idx*self.target_size:(idx+1)*self.target_size]

    # TODO: should be  a static function
    def convert_additional_name(self, name):
        inc_name = "additional_relative" if name.find("additional_relative") != -1 else ("parent_additional" if name.find("parent_additional") != -1 else ("additional" if name.find("additional") != -1 else name))
        if inc_name == "additional": aidx = int(name[len("additional"):])
        elif inc_name == "additional_relative": aidx = int(name[len("additional_relative"):])
        elif inc_name == "parent_additional": aidx = int(name[len("parent_additional"):])
        else: aidx = -1
        return inc_name, aidx

    def get_obs(self, last_state, full_state, param, mask, raw = False):
        '''
        gets an observation, as described by the binary string obs_setting
        if diff is not used, then last_state can be None
        '''
        combined = list()
        state_index = self.get_state_index(last_state, full_state, param, mask, raw = False)
        for name in self.single_order:
            inc_name, aidx = self.convert_additional_name(name)
            if self.name_include[inc_name]: 
                combined.append(self.add_obs(name, state_index))
        def combine_multi(num_obj, order, combined):
            for i in range(num_obj):
                for name in order:
                    inc_name, aidx = self.convert_additional_name(name)
                    if self.name_include[inc_name]:
                        combined.append(self.add_obs(name, state_index, i))
        if self.multiobject_order == 1:
            combine_multi(max(self.max_target_objects, self.max_parent_objects, max(self.max_additional_objects)), self.multi_order, combined)
        elif self.multiobject_order == 2:
            combine_multi(max(self.max_target_objects, self.max_parent_objects), self.multi_order, combined)
            combine_multi(max(self.max_parent_objects, max(self.max_additional_objects)), self.multi_order, combined)
        elif self.multiobject_order == 3:
            combine_multi(self.max_target_objects, self.multi_order, combined)
            combine_multi(self.max_parent_objects, self.multi_order, combined)
            combine_multi(max(self.max_additional_objects), self.multi_order, combined)
        return np.concatenate(combined, axis=-1)

    def get_where(self, name):
        def up_to(name, order, size_index):
            at = 0
            for n in order:
                if n == name:
                    break
                inc_name, aidx = self.convert_additional_name(n)
                if aidx >= 0: at += size_index[inc_name][aidx] * self.name_include[inc_name]
                else: at += size_index[inc_name] * self.name_include[inc_name]
            return at
        if name in self.single_order:
            return up_to(name, self.single_order, self.size_index), 0, 0
        single_order_len = up_to(self.single_order[-1], self.single_order, self.size_index) + self.single_size_index[self.single_order[-1]]
        if name in self.multi_order:
            return single_order_len, up_to(name, self.multi_order, self.single_size_index), self.target_obj_dim
        multi_order_len = single_order_len + up_to(self.multi_order[-1], self.multi_order, self.size_index) + self.size_index[self.multi_order[-1]]
        if name in self.multi_second_order:
            return multi_order_len, up_to(name, self.multi_second_order, self.single_size_index), self.parent_obj_dim
        multi_second_order_len = multi_order_len + up_to(self.multi_second_order[-1], self.multi_second_order, self.size_index) + self.size_index[self.multi_second_order[-1]]
        if name in self.multi_third_order:
            return multi_second_order_len, up_to(name, self.multi_third_order, self.single_size_index), self.additional_obj_dims
        print("invalid name")


    def assign_param(self, full_state, obs, param, mask):
        '''
        obs may be 1 or 2 dimensional vector. param should be 1d vector
        full state is needed for relative param, as is mask
        since we are assigning to obs, make sure this is normalized
        '''
        if len(obs.shape) == 2 and len(param.shape) == 1:
            # batch of observations
            param = broadcast(param, obs.shape[0], cat=False)
        norm_param = self.norm(param) * mask if self.combine_param_mask else self.norm(param)
        if self.name_include["param"]: obs[...,:self.target_size] = norm_param
        if self.name_include["param_relative"]: # handling param relative, full_state must have the same batch size as obs
            target_raw = self.target_select(full_state["factored_state"])
            n_target = target_raw.shape[-1] // self.target_size
            first_dim, pre_relative, obj_dim = self.get_where("param_relative")
            for i in range(n_target):
                at = int(first_dim + i * obj_dim)
                obs[...,at + pre_relative:at + pre_relative + self.target_size] =  self.norm((target_raw[...,i*self.target_size:(i+1)*self.target_size] - param), form="diff") * mask
        if self.name_include["parent_param"]: # handling param relative, full_state must have the same batch size as obs
            parent_raw = self.parent_select(full_state["factored_state"])
            n_parent = parent_raw.shape[-1] // self.target_size
            first_dim, pre_relative, obj_dim = self.get_where("parent_param")
            for i in range(n_parent):
                at = int(first_dim + i * obj_dim)
                obs[...,at + pre_relative:at + pre_relative + self.target_size] =  self.norm((parent_raw[...,i*self.target_size:(i+1)*self.target_size] - param), form="rel") * mask
        return obs