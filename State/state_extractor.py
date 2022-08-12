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
        self.single_obs_setting = args.single_obs_setting # TODO: add additional state to obs
        self.relative_obs_setting = args.relative_obs_setting # TODO: add additional state to obs
        self.obs_setting = self.single_obs_setting + self.relative_obs_setting
        
        self.has_additional = len(args.object_names.additional) > 0

        # combines the mask with the param
        self.combine_param_mask = args.combine_param_mask

        self.assign_extractor_norm(args.inter_extractor, args.norm)
        # sizes for first_object_dim, object_size for obs


    def assign_extractor_norm(self, inter_extractor, norm):
        self.inter_extractor = inter_extractor
        self.target_select, self.full_parent_select, self.additional_select, self.additional_selectors, self.padi_selector, self.parent_select, self.inter_select = self.inter_extractor.get_selectors()
        self.norm = norm

        # object dimensions for interleaving
        self.target_size = self.norm.target_norm[0].shape[0]
        self.inter_size = self.norm.inter_norm[0].shape[0]
        self.parent_size = self.norm.parent_norm[0].shape[0] # primary parent size

        # the maximum number of objects for the target, primary parent, one of these must be 1, since simultanious multi-parent and multi-object is not implemented
        self.max_target_objects = inter_extractor.max_target
        self.max_parent_objects = inter_extractor.max_parent
        self.max_additional_objects = np.array(inter_extractor.max_additional)
        self.additional_sizes = np.array(inter_extractor.additional_sizes)
        self.max_paddi_objects = inter_extractor.max_parent_multi
        self.max_partar = max (self.max_target_objects, self.max_parent_objects)
        self.observation_extractor = ObservationExtractor(self)
        self.first_obj_dim, self.target_obj_dim, self.parent_obj_dim, self.additional_obj_dim, self.rel_obj_dim, self.obj_dim, self.post_dim = self.observation_extractor._get_dims()
        self.total_size = self._get_size()

    def _get_size(self): # gets the size of the observation
        param, parent, additional, target, inter, diff = self.single_obs_setting
        parent_relative, additional_relative, parent_param, param_relative = self.relative_obs_setting
        return int(param * self.target_size + parent * self.parent_size * self.max_parent_objects 
            + additional * np.sum([self.max_additional_objects[i] * self.additional_sizes[i] for i in range(len(self.additional_sizes))])
            + target * self.target_size * self.max_target_objects + inter * self.inter_size + diff * self.target_size * self.max_target_objects
            + parent_relative * self.target_size * self.max_partar + additional_relative * self.target_size * np.sum(self.max_additional_objects * self.max_target_objects)
            + parent_param * self.target_size * self.max_parent_objects + param_relative * self.target_size * self.max_target_objects)

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

    def param_mask(self, param, mask, normalize=True): # combines the param with the mask, if necessary, and normalizes
        if normalize: return self.norm(param) * mask if self.combine_param_mask else self.norm(param)
        return param * mask if self.combine_param_mask else param

    # Most getters are rarely used, since ObservationExtractor handles gets internally
    def get_raw(self, full_state, norm=False):
        if norm: return self.norm(full_state["raw_state"], form = "raw")
        else: return full_state["raw_state"]

    def get_target(self, full_state, norm=False):
        if norm: return self.norm(self.target_select(full_state["factored_state"]))
        return self.target_select(full_state["factored_state"])

    def get_inter(self, full_state, norm=False):
        if norm: return self.norm(self.inter_select(full_state["factored_state"]), form="inter")
        return self.inter_select(full_state["factored_state"])

    def get_diff(self, last_full, full_state, norm=False):
        if norm: return self.norm(self.target_select(full_state["factored_state"]) - self.target_select(last_full["factored_state"]), form="diff")
        return self.target_select(full_state["factored_state"]) - self.target_select(last_full["factored_state"])

    def get_additional(self, full_state, norm=False, partial=False):
        # if there is not additional state, returns a dummy vector
        if not self.has_additional: return np.zeros(self.get_parent(full_state)[...,:1].shape)
        if partial:
            if norm: return [self.norm(self.additional_selectors[i](full_state["factored_state"]), form="additional" + str(i)) for i in range(len(self.additional_selectors))] 
            else: return [addisel(full_state["factored_state"]) for addisel in self.additional_selectors]         
        if norm: return self.norm(self.additional_select(full_state["factored_state"]), form="additional") 
        else: return self.additional_select(full_state["factored_state"])

    def get_parent(self, full_state, norm=False):
        if norm: return self.norm(self.parent_select(full_state["factored_state"]), form="parent")
        return self.parent_select(full_state["factored_state"])

    # Observation extractor getters
    def get_obs(self, last_state, full_state, param, mask, raw = False):
        return self.observation_extractor.get_obs(last_state, full_state, param, mask, raw)

    def reverse_obs_norm(self, obs, mask=None):
        return self.observation_extractor.reverse_obs_norm(obs, mask=mask)

    def assign_param(self, full_state, obs, param, mask):
        return self.observation_extractor.assign_param(full_state, obs, param, mask)