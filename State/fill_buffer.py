

def fill_buffer(environment, data, args, object_names):
    buffer = ParamWeightedReplayBuffer(len(data), stack_num=1)    
    factored_state = data[0]
    for next_factored_state in data[1:]:
    	act = next_factored_state["Action"]
    	factored_state["Action"] = next_factored_state["Action"]
    	done = factored_state["Done"]
    	rew = factored_state["Reward"]

    	# assign selections of the state
    	target = args.target_select(factored_state)
    	next_target = args.target_select(next_factored_state)
    	target_diff = next_target - target
    	inter_state = args.inter_select(factored_state)
    	parent_state = args.parent_selectors[object_names.primary_parent](factored_state)
    	additional_state = args.parent_select(factored_state)

    	# compute trace should give back if there is a true interaction at a state
    	inter = environment.compute_trace(factored_state, act, object_names)

    	# add one step to the buffer
    	buffer.add(Batch(act=act, done=done, rew=rew, target=target, next_target=next_target, target_diff=target_diff,
    					inter=inter, inter_state=inter_state, parent_state=parent_state, additional_state=additional_state))

    	factored_state = next_factored_state
    return buffer