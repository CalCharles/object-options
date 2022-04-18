HOOD rewrite:

Options:
	Reward, termination, done
	Networks:
		Policy
	Tianshou interface
	hindsight management
		hindsight collector
	collector
		aggregator
		collect
	train RL
		train

Shared components:
State
	tianshou rollout wrapper
		getitem
	dataset reader:
		read to factored_state
	State management:
		state extractor:
			get_raw
			get_obs
			get_target
			get_inter
			get_diff
		normalizer
	feature selector
		assign feature
		sample feature
		__call__
Environment
	Environment wrapper
		attributes:
			object_sizes
		set_save, save
		step
		reset
		get_state
Networks:
	Networks:
		MLP
		Conv
		Pair

Active passive model rewrite:
	Networks:
		forward(interaction state nparray, target state nparray): mean, variance  
		interaction(interaction state nparray): binary
	interaction model:
		attributes:
			interaction state extractor, target state extractor, parent state extractor(s), controllable state extractor
			testing module
		hypothesize(factored state dict, next factored state dict): binary, active forward, passive forward
		check_interaction(interaction_floats) -> interaction binaries (calls the testing module)
		get_active_mask() -> binary mask (pulls from testing module)
	Auxilary module:
		weighting system
	training module
		fill_rollouts(path string): r.inter, r.target, r.next_target
		train(rollout tianshou rollout, interaction model, training parameters object dict/namespace): interaction model
	testing module
		interaction binary test(active forward distribution, passive forward distribution, target nparray): binaries
		interaction test(interaction_binary nparray): binaries
	mask module:
		determine active set(rollouts, interaction model): active mask nparray
		collect samples(rollouts, interaction model, active mask nparray): active mask nparray
	feature explorer:
		search
	sampler:
		attributes:
			mask
			sample_able
		sample

