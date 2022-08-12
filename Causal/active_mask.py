import numpy as np
from State.feature_selector import broadcast
from Network.network_utils import pytorch_model

class ActiveMasking():
    def __init__(self, rollouts, interaction_model, min_sample_difference, var_cutoff, parent_active, parent_max_num, num_samples=30, sample_grid=True, dynamics_difference=False):
        
        # uses the limit values of the norm to form a convex set of possible values
        self.tar_name, self.par_name = interaction_model.names.target, interaction_model.names.primary_parent
        self.limits = interaction_model.norm.lim_dict[self.tar_name]
        self.range = interaction_model.norm.lim_dict[self.tar_name][1] - interaction_model.norm.lim_dict[self.tar_name][0]

        parent_limits = interaction_model.norm.lim_dict[self.par_name]

        self.num_samples= num_samples
        self.sample_grid = sample_grid
        self.min_sample_difference = min_sample_difference
        self.var_cutoff = np.array([var_cutoff[0] for i in range(len(self.limits[0]))]) if len(var_cutoff) == 1 else np.array(var_cutoff)
        self.active_mask = self.determine_active_set(rollouts, interaction_model, parent_limits, parent_active, parent_max_num, dynamics_difference)
        print("active_mask", self.active_mask)
        self.active_set = self.collect_samples(rollouts, interaction_model)
        self.filtered_active_set = self.filter_active()
        print("filtered active set", self.filtered_active_set)

    def regenerate_norm(self, norm):
        self.limits = norm.lim_dict[self.tar_name]
        self.range = norm.lim_dict[self.tar_name][1] - norm.lim_dict[self.tar_name][0]

    def compute_var_cutoffs(self, rollouts):
        # not used, a possible alternative to hardcoded variance cutoffs
        return np.std(rollouts.target_diff[interaction_model.test(rollouts.inter)], axis=-1)

    def determine_active_set(self, rollouts, interaction_model, parent_limits, parent_active, parent_max_num, dynamics_difference):
        # generates a mask over which components of state change with different values of the parent object
        # parent_active is the active mask for the parent object
        diffs = list()
        total_interactions = 0
        for i in range(len(rollouts)):
            batch = rollouts[i]
            if interaction_model.test(batch.inter): # assumes interactions are already computed
                inter, (active_mean, av), (pm, pv) = interaction_model.hypothesize((batch.inter_state, batch.target))

                if self.sample_grid:
                    # sample alternative states based on a meshgrid, where min(num_samples, parent_max_num (which is num_actions when discrete)) is used at each dimension
                    n_dim = int(np.ceil(np.power(self.num_samples, 1/np.sum(parent_active))))
                    # generate a linspace for every active attribute of the parent, where parent_max_step prevents oversampling an attribute (for actions)
                    lin_spaces = [np.linspace(parent_limits[0][j ], parent_limits[1][j], min(self.num_samples, parent_max_num)) for j in range(len(parent_limits[0])) if parent_active[j] != 0]
                    # create the values to sample
                    spaces_set = np.meshgrid(*lin_spaces)
                    parent_sample = np.vstack([s.flatten() for s in spaces_set]).T
                else:
                    parent_sample = np.stack([np.random.rand(*parent_limits[0].shape) * parent_limits[1] + parent_limits[0] for j in range(self.num_samples)])
                # assign parent, and then 
                sampled_states = list()
                for sample in parent_sample:
                    inter_state = batch.inter_state.copy()
                    sample = interaction_model.norm(sample, idxes = np.nonzero(parent_active)[0], form = "parent")
                    sampled_states.append(interaction_model.inter_select.reverse(batch.inter_state.copy(), sample, names=[self.par_name], mask=parent_active))
                difference = interaction_model.predict_dynamics and dynamics_difference
                tar_val = batch.target_diff if difference else batch.next_target
                sampled_states, broad_target = np.stack(sampled_states, axis=0), broadcast(tar_val, len(sampled_states), cat=False) 

                # evaluate on the new values
                inter_sam, pred = interaction_model.predict_next_state((sampled_states, broad_target), normalized=True, difference = difference)
                # print(pred, broad_target, difference)
                # if at least one interaction occurs, appends the maximum change seen
                if inter + inter_sam.sum() > 1:
                    # print(np.max(np.abs(pred - broad_target), axis=0), pred, tar_val, interaction_model.norm(pred, form="dyn"), interaction_model.norm(tar_val, form="dyn"))
                    diffs.append(np.max(np.abs(pred - broad_target), axis=0))
                    total_interactions += 1

        # gets the difference variance for each dimension, and compares against the minimum variance for active interest
        std = np.mean(np.stack(diffs, axis=0), axis=0)
        print("std", std)
        mask = np.array([1 if std[i] > self.var_cutoff[i] else 0 for i in range(len(std))])
        return mask

    def collect_samples(self, rollouts, interaction_model):
        # generate a full sampling over the values the object can take on post-interaction
        # allows redundant values
        return interaction_model.norm.reverse(rollouts.next_target[interaction_model.test(rollouts.inter)]) * self.active_mask

    def filter_active(self):
        '''
        filters self.masking.active_set based on the active mask
        if states are the same after masking, they are only counted once
        this removes duplicates of the state
        '''
        active_filtered = list()
        active_count = dict()
        for state in self.active_set:
            masked_state = state * self.active_mask
            failed = False
            for i, val in enumerate(active_filtered):
                if np.linalg.norm(masked_state - val, ord=1) < self.min_sample_difference:
                    failed = True
                    active_count[i] += 1
            if not failed:
                active_count[len(active_filtered)] = 1
                active_filtered.append(masked_state)
        # remove low occurrance values when we have a small active set
        if len(active_filtered) < 10:
            remove_keys = [i for i in active_count.keys() if active_count[i] < 5]
            remove_keys.sort()
            remove_keys.reverse()
            for i in remove_keys:
                active_filtered.pop(i)
                del active_count[i]
        return active_filtered
