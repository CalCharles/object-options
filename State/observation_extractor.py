import numpy as np
from State.feature_selector import broadcast

class ObservationExtractor(): 
# an internal class for state extractor to handle the observation-related components
    def __init__(self, state_extractor):
        # class variables for the order of elements to append, normalizations to apply
        self._component_names = ["param", "additional", "inter", "parent", "relative", "target", "param_relative", "diff"] # TODO: add additional state
        self._multiparent_order = ["param", "target", "param_relative", "diff", "additional", "inter", "parent", "relative", ] # TODO: add additional state
        self._multitarget_order = ["param", "additional", "inter", "parent", "relative", "target", "param_relative", "diff"] # TODO: add additional state
        self._norm_forms = {"param": "target", "additional": "additional", "inter": "inter", "parent": "parent", "relative": "rel", "target": "target", "param_relative": "diff", "diff": "diff"}
        
        # environment hyperparameters
        self.target_size = state_extractor.target_size
        self.parent_size = state_extractor.parent_size
        self.inter_size = state_extractor.inter_size
        self.max_target_objects = state_extractor.max_target_objects
        self.max_parent_objects = state_extractor.max_parent_objects
        self.max_instanced = state_extractor.max_instanced
        self.parent_select = state_extractor.parent_select
        self.target_select = state_extractor.target_select
        self.inter_select = state_extractor.inter_select
        self.additional_select = state_extractor.additional_select

        # observation values
        self.obs_setting = state_extractor.obs_setting
        self.norm = state_extractor.norm
        self.size_index = {"param": self.target_size, "inter": self.inter_size, "parent": self.parent_size * self.max_parent_objects,
                    "relative": self.target_size * self.max_instanced, "target": self.target_size * self.max_target_objects,
                    "param_relative": self.target_size * self.max_target_objects, "diff": self.target_size * self.max_target_objects}

        self.param_mask = state_extractor.param_mask
        self.combine_param_mask = state_extractor.combine_param_mask
        self.first_obj_dim, self.obj_dim = state_extractor.first_obj_dim, state_extractor.obj_dim

    def reverse_obs_norm(self, obs):
        combined = list()
        def add_names(obs, bins, order, combine, start):
            # adds the reverse-norm components of obs, corresponding to the nonzero values in bins, in the order of order
            # appends the outcomes to combine, and starts from start
            size_to = start
            for use, name in zip(bins, order):
                if use:
                    increment = self.size_index[name]
                    combine.append(self.norm.reverse(obs[...,int(size_to):int(size_to + increment)], form=self._norm_forms[name]))
                    size_to += increment
            return size_to

        param, additional, inter, parent, relative, target, param_relative, diff = self.obs_setting
        size_to = 0
        if param: # the param is always the first part
            combined += [self.norm.reverse(obs[...,:self.target_size])]
            size_to = self.target_size * param
        if self.max_parent_objects > 1: # if multiple parents, first reverse the non-instanced components 
            size_to = add_names(obs, [target, param_relative, diff, additional, inter], self._multiparent_order[1:6], combined, size_to)
            for i in self.max_parent_objects:
                size_to = add_names(obs, [parent, relative], self._multiparent_order[6:], combined, size_to)
        elif self.max_target_objects > 1: # if multiple targets, the order changes
            size_to = add_names(obs, [additional, inter, parent], self._multitarget_order[1:4], combined, size_to)
            for i in self.max_parent_objects:
                size_to = add_names(obs, [target, param_relative, diff, relative], self._multitarget_order[4:], combined, size_to)
        else: # if no multi-instancing, just add
            size_to = add_names(obs, self.obs_setting[1:], self._multitarget_order[1:], combined, size_to)
        return np.concatenate(combined, axis=-1)

    def get_obs(self, last_state, full_state, param, mask, raw = False):
        '''
        gets an observation, as described by the binary string obs_setting
        if diff is not used, then last_state can be None
        '''
        if raw: return self.get_raw(full_state, norm=True)
        combined_state = list()
        p, additional, inter, parent, relative, target, param_relative, diff = self.obs_setting
        if p: combined_state.append(self.param_mask(param, mask)) # assumes param is already normalized
        if self.max_parent_objects > 1: # add the target components first (so they can be used for first-object networks)
            combined_state += self._add_target_param_diff(target, param_relative, diff, 0, last_state, full_state, param, mask)
            combined_state += self._add_addi_inter_par_rel(additional, inter, parent, relative, full_state)
        elif self.max_target_objects >= 1: # add the interaction components first
            combined_state += self._add_addi_inter_par_rel(additional, inter, parent, 0, full_state)
            combined_state += self._add_target_param_diff(target, param_relative, diff, relative, last_state, full_state, param, mask)
        return np.concatenate(combined_state, axis=-1)

    def _add_target_param_diff(self, target, param_relative, diff, relative, last_state, full_state, param, mask):
        '''
        adds the following state components: target state, target state relative to parameter, difference between current and last target state, relative interaction-target state
        only performs relative state information when the number of parent objects is equal to 1
        '''
        target_raw = self.target_select(full_state["factored_state"])
        n_target = target_raw.shape[-1] // self.target_size
        tar, parent_relative, param_rel, differ = list(), list(), list(), list()
        if target: tar = [self.norm(target_raw)]
        if param_relative: # get target - param, broadcasted since there may be multiple targets
            broad_param, broad_mask = broadcast(param, n_target), broadcast(mask, n_target)
            param_rel = [self.norm((target_raw - broad_param) * broad_mask , form="diff")]
        if relative:
            parent_raw = self.parent_select(full_state["factored_state"])
            broad_parent = broadcast(parent_raw, n_target)
            parent_relative = [self.norm(broad_parent - target_raw, form="rel")]
        if diff: differ = [self.norm(target_raw - self.target_select(last_state["factored_state"]), form="diff")]
        if n_target > 1: # interleaving
            combined = list()
            for i in range(n_target):
                # param relative is first so that assignment is easier (it is the only param dependent component)
                if target: combined += target[0][...,i*self.target_size:(i+1)*self.target_size]
                if param_relative: combined += param_rel[0][...,i*self.target_size:(i+1)*self.target_size]
                if diff: combined += differ[0][...,i*self.target_size:(i+1)*self.target_size]
                if relative: combined += parent_relative[0][...,i*self.target_size:(i+1)*self.target_size]
            return combined
        else: # no interleaving means easy combination
            return parent_relative + tar + param_rel + differ
    
    def _add_addi_inter_par_rel(self, additional, inter, parent, relative, full_state):
        '''
        adds the parent state and parent state-target state
        only applys parent to target state when parent object num > 1
        '''
        parent_raw = self.parent_select(full_state["factored_state"])
        n_parent = parent_raw.shape[-1] // self.parent_size
        additional_state, inter_state, par, parent_relative, tarrel = list(), list(), list(), list(), list()
        combined = list()
        if additional: additional_state = [self.norm(self.additional_select(full_state["factored_state"]), form="additional")]
        if inter: inter_state = [self.norm(self.inter_select(full_state["factored_state"]), form="inter")]  # inter is not interleaved, use parent, additional if interleaving
        if parent: par = [self.norm(parent_raw)]
        if relative:
            target_raw = self.target_select(full_state["factored_state"])
            broad_target = broadcast(target_raw, n_parent)
            parent_relative = [self.norm(parent_raw - broad_target, form="rel")]
        if n_parent > 1: # interleaving
            combined += additional_state
            combined += inter_state
            for i in range(n_parent):
                if parent: combined += par[0][...,i*self.parent_size: (i+1)*self.parent_size]
                if relative: combined += parent_relative[0][...,i*self.parent_size: (i+1)*self.parent_size]
            return combined
        else: # no interleaving means easy combination
            return additional_state + inter_state + par + tarrel

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
        obs[...,:self.target_size] = norm_param
        if self.obs_setting[6]: # handling param relative, full_state must have the same batch size as obs
            target_raw = self.target_select(full_state["factored_state"])
            n_target = target_raw.shape[-1] // self.target_size
            pre_relative = int(self.obs_setting[4] * self.target_size + self.obs_setting[5] * self.target_size)
            for i in range(n_target):
                at = int(self.first_obj_dim + i * self.obj_dim)
                obs[...,at + pre_relative:at + pre_relative + self.target_size] =  self.norm((target_raw[...,i*self.target_size:(i+1)*self.target_size] - param), form="diff") * mask
        return obs