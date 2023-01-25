import numpy as np

def check_proximity(proximity, target_states, parent_states): # uses l1 proximity for simplicity
	# print(np.min(proximity - np.abs(target_states - parent_states), axis=-1)[:50], np.abs(target_states - parent_states)[:50], proximity - np.abs(target_states - parent_states)[:50])
	return np.min(proximity - np.abs(target_states - parent_states), axis=-1) > 0 if np.sum(proximity) > 0 else np.ones(target_states.shape[:-1]).astype(bool)

def count_modes(assignments):
	#counts the number of samples in each mode
	mode_counts = dict()
	for a in assignments:
		if a < 0: continue
		if a in mode_counts:
			mode_counts[a] += 1
		else:
			mode_counts[a] = 1
	return mode_counts 


class ChangepointModel():
	def __init__(self, changepoint_model, proximity, cluster_model, traj_dim, skip_one):
		self.changepoint_model = changepoint_model # either CHAMP or some other model for detecting changepoints
		self.proximity = proximity # what constitutes proximity in each relevant dimension
		self.cluster_model = cluster_model # cluster_model.predict determines which post-dynamics mode will be used
		self.traj_dim = traj_dim
		self.valid_modes = None # set this with fit_modes
		self.num_modes = -1 # should be set once fit_modes is run
		self.skip_one = skip_one

	def get_changepoints(self, target_states, parent_states, dones):
		# gets changepoints and returns the correspond segment models, indices of the changepoints, raw changepoints and changepoints filtered for proximity and done flags
		
		# sets the trajectory dimension
		target_states, parent_states = target_states[...,:self.traj_dim], parent_states[...,:self.traj_dim]
		# runs the changpoint model
		models, indices = self.changepoint_model.generate_changepoints(target_states)
		changepoints = np.zeros(target_states.shape[:-1]).astype(bool)
		changepoints[indices] = True

		# will return all True if proximity is not used
		proximity = check_proximity(self.proximity, target_states, parent_states)

		# changepoints at dones filtered out
		done_flags = 1-dones
		if self.skip_one: done_flags[1:] = done_flags[:-1] * done_flags[1:]

		filtered_changepoints = (changepoints.squeeze() * proximity.squeeze() * done_flags.squeeze()).astype(bool) # make sure the very last state is not counted as a changepoint
		if len(filtered_changepoints.shape) == 0:
			filtered_changepoints = np.expand_dims(filtered_changepoints, axis=0)
		return models, indices, changepoints, filtered_changepoints, proximity

	def fit_modes(self, target_diff, target_states, parent_states, dones, min_size=10):
		# fits the DP-GMM

		# gets changepoints
		models, indices, changepoints, filtered_changepoints, proximity = self.get_changepoints(target_states, parent_states, dones)
		# use changepoints to get appropriate state dynamics (post-changepoint dynamics)
		target_diff = target_diff[...,:self.traj_dim]
		mode_data = target_diff[filtered_changepoints]
		print("data", mode_data)
		done_flags = 1-dones
		if self.skip_one: done_flags[1:] = done_flags[:-1] * done_flags[1:]
		for i in range(len(target_states)):
			if changepoints[i]: print(filtered_changepoints[i], changepoints[i], proximity[i], done_flags[i], target_diff[i], target_states[i], parent_states[i])
		self.cluster_model.fit(mode_data)
		assignments, filtered_changepoints = self.get_mode(target_diff, target_states, parent_states, dones)
		mode_counts = count_modes(assignments)
		modes = [a for a in mode_counts.keys() if mode_counts[a] > min_size]
		self.num_modes = len(modes)
		for v in [(a, mode_counts[a], self.cluster_model.model.means_[int(a)]) for a in mode_counts.keys()]:
			print(v)
		print(self.num_modes, modes)
		return modes

	def get_mode(self, target_diff, target_states, parent_states, dones):
		# assigns each of the target_diffs to a mode, where -1 indicates no mode should be used (no changepoint prior)
		target_diff = target_diff[...,:self.traj_dim]
		
		if self.changepoint_model is not None:
			models, indices, changepoints, filtered_changepoints, proximity = self.get_changepoints(target_states, parent_states, dones)
			# print(filtered_changepoints.shape)
			# print("changepoints", changepoints)
			# get the displacements
			# displacement = target_states[np.roll(filtered_changepoints, 1)] - target_states[filtered_changepoints]
			filtered_indices = np.where(filtered_changepoints)[0]
			displacement = target_diff[filtered_changepoints]
			# print("fcp", filtered_changepoints)
		else:
			filtered_changepoints = np.ones(len(target_states)).astype(bool)
			filtered_indices = np.arange(len(target_states))
			displacement = target_diff
		if self.cluster_model is not None:
			if len(displacement) > 0: mode_assignment = self.cluster_model.predict(displacement)
			else: mode_assignment = np.zeros(displacement.shape)
		else:# there is only one mode
			mode_assignment = np.zeros(len(filtered_indices))
		assignments = np.ones(target_states.shape[:-1]) * -1 # assign -1 to all non-assignments
		# print(filtered_indices, mode_assignment, target_states.shape, assignments, target_diff.shape)
		if len(filtered_indices) > 0:
			# print(filtered_indices, mode_assignment)
			assignments[filtered_indices] = mode_assignment
		return assignments, filtered_changepoints
