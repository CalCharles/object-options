


# state extractor
class StateExtractor():
    def __init__(self, args, option_selector, full_state, param, mask):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''
        # state selectors, parent selectors is currently not used
        self.inter_select = kwargs['inter_select']
        self.target_select = kwargs['target_select']
        self.parent_selectors = kwargs['parent_selectors']
        self.parent_select = kwargs['parent_select']

        self.norm = kwargs["norm"]

        # object dimensions for interleaving
        self.target_size = kwargs['target_size']
        self.parent_size = kwargs['parent_size'] # primary parent size

        # the maximum number of objects for the target, primary parent, one of these must be 1, since multi-instanced parent and object is not implemented
        self.max_target_objects = kwargs["max_target_objects"]
        self.max_parent_objects = kwargs["max_parent_objects"]

        # sizes for first_object_dim, object_size for obs
        param, inter, relative, target, param_relative, diff = self.obs_setting
        self.first_obj_dim, self.obj_dim = self.get_dims 


    def get_raw(self, full_state):
        return self.norm(full_state["raw_state"], form = "raw")

    def get_target(self, full_state):
        return self.norm(self.target_select(full_state["factored_state"]))

    def get_inter(self, full_state):
        return self.norm(self.inter_select(full_state["factored_state"]), form="inter")

    def get_diff(self, last_full, full_state):
        return self.norm(self.target_select(full_state["factored_state"]) - self.target_select(last_full["factored_state"]), form="diff")

    def get_obs(self, last_state, full_state, param, mask):
        '''
        gets an observation, as described by the binary string obs_setting
        if diff is not used, then last_state can be None
        '''
        combined_state = list()
        p, inter, relative, target, param_relative, diff = self.obs_setting
        if p: combined_state.append(self.norm(param)) # assumes param is already normalized
        if self.max_parent_objects > 1: # add the target components first (so they can be used for first-object networks)
            combined_state += self._add_target_param_diff(target, param_relative, diff, 0, last_state, full_state, param, mask)
            combined_state += self._add_inter_rel_diff(inter, relative, full_state)
        elif self.max_target_objects >= 1: # add the interaction components first
            combined_state += self._add_inter_rel(inter, 0, full_state)
            combined_state += self._add_target_param_diff(target, param_relative, diff, last_state, full_state, param, mask)
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
            param_rel = [self.norm(target * broad_mask - broad_param, form="diff")]
        if relative:
            parent_raw = self.parent_select(full_state["factored_state"])
            broad_parent = broadcast(parent_raw, n_target)
            parent_relative = [self.norm(broad_parent - target, form="rel")]
        if diff: differ = [self.norm(target_raw - self.target_select(last_state["factored_state"]), form="diff")]
        if n_target > 1: # interleaving
            combined = list()
            for i in range(n_target):
                # param relative is first so that assignment is easier (it is the only param dependent component)
                if param_relative: combined += param_rel[i*self.target_size, (i+1)*self.target_size]
                if target: combined += target[i*self.target_size, (i+1)*self.target_size]
                if relative: combined += parent_relative[i*self.target_size, (i+1)*self.target_size]
                if diff: combined += differ[i*self.target_size, (i+1)*self.target_size]
            return combined
        else: # no interleaving means easy combination
            return diff + param_rel + tar
    
    def _add_inter_rel_diff(self, parent, relative, full_state):
        '''
        adds the parent state and parent state-target state
        only applys parent to target state when parent object num > 1
        '''
        parent_raw = self.parent_select(full_state["factored_state"])
        n_parent = parent_raw.shape[-1] // self.parent_size
        par, tarrel = list(), list()
        if parent: par = [self.norm(parent_raw)]
        if relative:
            target_raw = self.target_select(full_state["factored_state"])
            broad_target = broadcast(target_raw, n_parent)
            parent_relative = [self.norm(parent_raw - broad_target, form="rel")]
        if n_target > 1: # interleaving
            combined = list()
            for i in range(n_target):
                if parent: combined += par[i*self.target_size, (i+1)*self.target_size]
                if relative: combined += parent_relative[i*self.target_size, (i+1)*self.target_size]
            return combined
        else: # no interleaving means easy combination
            return par + tarrel

    def assign_param(self, full_state, obs, param, mask):
        '''
        obs may be 1, 2 or 3 dimensional vector. param should be 1d vector
        full state is needed for relative param, as is mask
        since we are assigning to obs, make sure this is normalized
        '''
        if len(obs.shape) == 1:
            # batch of observations
            obs[:,:self.target_size] = broadcast(param, obs.shape[0], cat=False)
        elif len(obs.shape) == 0:
            # singleton observation
            obs[:self.target_size] = param
        if self.obs_setting[4]: # handling param relative, full_state must have the same batch size as obs
            target_raw = self.target_select(full_state["factored_state"])
            n_target = target_raw.shape[-1] // self.target_size
            for i in range(n_target):
                at = self.first_obj_dim + i * self.obj_dim
                obs[at:at + self.target_size] =  self.norm(target_raw[...,i*self.target_size:(i+1)*self.target_size] * mask - param, form="diff")