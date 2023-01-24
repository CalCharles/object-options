Option Chain:

	Reward function: MultipleCluster, determiners
		Initialize the changepoint detector with the changepoint determiner into the changepoint reward
		changepoint detector
		changepointReward (proximity)

	option:
		reward function 
		policy
		Learning algorithm

	train_HyPE:
		RewardFunction():
			attributes:
				changepoint model
				proximity_distance
			functions:
				train(changepoint_args, rollouts): trains the CHAMP model
				set_reward(reward_args): Assign proximity
				check_proximity(factored_state): 
				assign_reward(factored_states): 
		Option:
			attributes:
				network
				reward_functions
				optimizer
			function:
				get_action()
		CMAES optimizer:
			attributes:
				
			function:
				run_optimizer(rollouts): 
