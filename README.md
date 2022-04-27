HOOD rewrite:

Options:
	Reward, termination, done
	option temporal extension
		Tianshou interface

	Networks:
		Policy

	RL
	collector
		aggregator
		collect
		train

	hindsight management
		hindsight collector

Shared components:
State
	tianshou rollout wrapper: WRITTEN
		getitem(index) -> batch
		sample with weights(length, weights) -> batch
	fill_buffer: WRITTEN
		fill_buffer(list of ordered factored state) -> tianshou buffer
	dataset reader: WRITTEN
		read_obj_dumps(path to directory, start index, number of samples, filename) -> list of factored states
	State management:
		state extractor: WRITTEN
			get_raw
			get_obs
			get_target
			get_inter
			get_diff
		normalizer: WRITTEN
	factored_state: a dict with object name: ndarray, either a single state or a batch of states
	feature selector: WRITTEN
		assign feature(delta value, factored_state) -> mutate factored state to the values in delta value
		__call__(factored state) -> ndarray of desired value
Environment: WRITTEN
	Environment wrapper
		attributes:
			object_sizes
			norm_values
		set_save, save
		step
		reset
		get_state
Networks: WRITTEN
	Networks:
		MLP
		Conv
		Pair

Active passive model rewrite:
	Networks: WRITTEN
		forward(interaction state nparray, target state nparray): mean, variance  
		interaction(interaction state nparray): binary
	interaction model: WRITTEN
		attributes:
			interaction state extractor, target state extractor, parent state extractor(s), controllable state extractor
			testing module
		hypothesize(factored state dict, next factored state dict): binary, active forward, passive forward
		check_interaction(interaction_floats) -> interaction binaries (calls the testing module)
		get_active_mask() -> binary mask (pulls from testing module)
	weighting system: WRITTEN
	training module: WRITTEN
		fill_rollouts(path string): r.inter, r.target, r.next_target
		train(rollout tianshou rollout, interaction model, training parameters object dict/namespace): full model
	testing module: WRITTEN
		interaction binary test(active forward distribution, passive forward distribution, target nparray): binaries
		interaction test(interaction_binary nparray): binaries
	
	mask module: WRITTEN
		determine active set(rollouts, interaction model): active mask nparray
		collect samples(rollouts, interaction model, active mask nparray): active mask nparray
	

	feature explorer: POSTPONED
		search
	sampler: WRITTEN
		attributes:
			mask
			sample_able
		sample

Code run:
	Train Interaction: WRITTEN
	fill active model: WRITTEN
	args
	train option
	test option