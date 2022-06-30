import numpy as np
from State.feature_selector import broadcast
from State.observation_extractor import ObservationExtractor

# state extractor
class StateExtractor():

    def __init__(self, args):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''        
        # state selectors, parent selectors is currently not used
        self.inter_select = args.inter_select
        self.target_select = args.target_select
        self.parent_selectors = args.parent_selectors
        self.additional_select = args.additional_select
        self.parent_select = args.parent_select
        self.has_additional = len(args.object_names.additional) > 0

        self.norm = args.norm

        # object dimensions for interleaving
        self.target_size = args.norm.target_norm[0].shape[0]
        self.inter_size = args.norm.inter_norm[0].shape[0] # inter does not handle multiple instances
        self.parent_size = args.norm.parent_norm[0].shape[0] # primary parent size

        # the maximum number of objects for the target, primary parent, one of these must be 1, since simultanious multi-parent and multi-object is not implemented
        self.max_target_objects = args.max_target_objects
        self.max_parent_objects = args.max_parent_objects
        self.max_instanced = max (self.max_target_objects, self.max_parent_objects)

        # sizes for first_object_dim, object_size for obs
        self.obs_setting = args.obs_setting # TODO: add additional state to obs
        self.first_obj_dim, self.obj_dim = self._get_dims()
        self.total_size = self._get_size()

        # combines the mask with the param
        self.combine_param_mask = args.combine_param_mask

        self.observation_extractor = ObservationExtractor(self)

    def _get_dims(self): # gets the first object dimension and object dimension for multi object-networks
        param, additional, inter, parent, relative, target, param_relative, diff = self.obs_setting
        if self.max_parent_objects > 1:
            return int(target * self.target_size + param * self.target_size + param_relative * self.target_size + diff * self.target_size), int(relative * self.parent_size + parent * self.parent_size)
        else: # multi-instanced targets or normal target numbers
            return int(param * self.target_size + inter * self.inter_size + parent * self.parent_size),  int(relative * self.target_size + target * self.target_size + param_relative * self.target_size)

    def _get_size(self): # gets the size of the observation
        param, additional, inter, parent, relative, target, param_relative, diff = self.obs_setting
        return int(param * self.target_size + inter * self.inter_size + parent * self.parent_size * self.max_parent_objects +
            relative * self.target_size * self.max_instanced + target * self.target_size * self.max_target_objects +
            param_relative * self.target_size * self.max_target_objects + diff * self.target_size * self.max_target_objects)

    def expand_param(self, mapped_action, mask): # expands the dimension of a mapped action by filling in the nonzero mask values, only handles 2d or less
        expanded = np.zeros((*mapped_action.shape[:len(mapped_action.shape) - 1], mask.shape[-1]))
        # broadcast can only handle 2d mapped action at most
        if len(mapped_action.shape) > len(mask.shape): mask = broadcast(mask, mapped_action.shape[0], cat=False)
        mask = mask.astype(bool)
        if len(expanded.shape) > 1: # assign each value separately
            for i in range(expanded.shape[0]):
                expanded[i, mask[i]] = mapped_action[i]
        else: expanded[mask] = mapped_action
        return expanded

    def param_mask(self, param, mask): # combines the param with the mask, if necessary, and normalizes
        return self.norm(param) * mask if self.combine_param_mask else self.norm(param)

    def get_raw(self, full_state, norm=False):
        if norm: return self.norm(full_state["raw_state"], form = "raw")
        else: return full_state["raw_state"]

    def get_target(self, full_state, norm=False):
        if norm: return self.norm(self.target_select(full_state["factored_state"]))
        return self.target_select(full_state["factored_state"])

    def get_inter(self, full_state, norm=False):
        if norm: self.norm(self.inter_select(full_state["factored_state"]), form="inter")
        return self.inter_select(full_state["factored_state"])

    def get_diff(self, last_full, full_state, norm=False):
        if norm: self.norm(self.target_select(full_state["factored_state"]) - self.target_select(last_full["factored_state"]), form="diff")
        return self.target_select(full_state["factored_state"]) - self.target_select(last_full["factored_state"])

    def get_additional(self, full_state, norm=False):
        # if there is not additional state, returns a dummy vector
        if not self.has_additional: return np.zeros(self.get_parent(full_state)[...,:1].shape)
        if norm: return self.norm(self.additional_select(full_state["factored_state"]), form="additional") 
        else: return self.additional_select(full_state["factored_state"])

    def get_parent(self, full_state, norm=False):
        if norm: self.norm(self.parent_select(full_state["factored_state"]), form="parent")
        return self.parent_select(full_state["factored_state"])

    def get_obs(self, last_state, full_state, param, mask, raw = False):
        if not hasattr(self, "observation_extractor"): self.observation_extractor = ObservationExtractor(self) #TODO: REMOVE THIS
        return self.observation_extractor.get_obs(last_state, full_state, param, mask, raw)

    def reverse_obs_norm(self, obs):
        return self.observation_extractor.reverse_obs_norm(obs)

    def assign_param(self, full_state, obs, param, mask):
        return self.observation_extractor.assign_param(full_state, obs, param, mask)