
class ActiveMasking():

    def __init__(self, rollouts, interaction_model, min_variance, num_samples=30):
        
        # uses the limit values of the norm to form a convex set of possible values
        self.tar_name, self.par_name = interaction_model.names.target, interaction_model.names.primary_parent
        self.limits = interaction_model.norm.lim_dict[self.tar_name]
        self.range = interaction_model.norm.lim_dict[self.tar_name][1] - interaction_model.norm.lim_dict[self.tar_name][0]

        parent_limits = interaction_model.norm.lim_dict[self.par_name]

        self.num_samples= num_samples
        self.min_variance = min_variance
        self.active_mask = self.determine_active_set(rollouts, interaction_model, parent_limits)
        self.active_set = self.collect_samples(rollouts, interaction_model, active_mask)

    def compute_var_cutoffs(self, rollouts):
        return np.std(rollouts.target_diff[interaction_model.test(rollouts.inter)], axis=-1)

    def determine_active_set(self, rollouts, interaction model, parent_limits):
        # generates a mask over which components of state change with different values of the parent object
        diffs = list()
        for batch in rollouts.sample(0):
            if interaction_model.test(batch.inter): # assumes interactions are already computed
                inter, active_mean, av, pm, pv = interaction_model.hypothesize((batch.inter_state, batch.target))

                # sample random different alternative states for the parent
                parent_sample = np.sample(low=parent_limits[0], high=parent_limits[1], num_sample=self.num_samples)
                sampled_states = list()
                for sample in parent_sample:
                    sampled_states.append(interaction_model.inter_select.reverse(batch.inter_state.copy(), sample, names=[self.par_name]))
                sampled_states, broad_target = np.stack(sampled_states, axis=0), broadcast(batch.target, cat=False) 
                
                # evaluate on the new values
                inter_sam, pred = interaction_model.predict_next_state((batch.inter_state, batch.target))
                diffs.append(pred - active_mean)

        # gets the difference variance for each dimension, and compares against the minimum variance for active interest
        std = np.std(np.concatenate(diffs, axis=0), axis=1)
        mask = [1 if std[i] > self.var_cutoffs[i] else 0 for i in range(len(std))]
        return mask

    def collect_samples(rollouts, interaction_model, active_mask):
        # generate a full sampling over the values the object can take on post-interaction
        # allows redundant values
        return rollouts.next_target[interaction_model.test(rollouts.inter)] * active_mask