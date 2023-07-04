from Buffer.buffer import InterWeightedReplayBuffer
from tianshou.data import Batch
import numpy as np

def fill_buffer(environment, data, args, object_names, norm, predict_dynamics):
    buffer = InterWeightedReplayBuffer(len(data), stack_num=1)
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        rew = factored_state["Reward"]

        # assign selections of the state
        target = norm(args.target_select(factored_state))
        next_target = norm(args.target_select(next_factored_state))
        target_diff = norm(args.target_select(next_factored_state) - args.target_select(factored_state), form="dyn")
        inter_state = norm(args.inter_select(factored_state), form="inter")
        # print(args.target_select(factored_state), args.target_select(next_factored_state), target, next_target, target_diff)

        # parent and additional states are unnormalized
        parent_state = args.parent_select(factored_state)
        additional_state = args.additional_select(factored_state)

        # compute trace should give back if there is a true interaction at a state
        inst_trace = environment.get_trace(factored_state, act, object_names)
        if len(inst_trace) > 1: trace = [np.sum(inst_trace)] # we don't store per-instance traces
        else: trace = inst_trace.copy()

        # add one step to the buffer
        use_done = factored_state["Done"] if object_names.primary_parent == "Action" else next_factored_state["Done"] # collect shifts dones late
        # use_done = factored_state["Done"] if predict_dynamics else last_done
        # print(i, target, next_target, args.target_select(factored_state), target_diff, use_done, factored_state["Done"], factored_state["Action"])
        # if np.linalg.norm(target) > 0: print(last_done, use_done, next_factored_state["Done"], args.target_select(next_factored_state) - args.target_select(factored_state), target_diff, factored_state["Target"],factored_state["Block"])
        # if np.linalg.norm(target_diff) > 0: print(i, last_done, use_done, factored_state["Action"], next_factored_state["Done"], args.inter_select(factored_state), target)
        # else: print(i, factored_state["Action"])
        # REMOVE THIS LINE
        # if np.max(np.abs(target_diff)) > 1: use_done = True 
        # print(i, use_done, args.target_select(next_factored_state) - args.target_select(factored_state), target_diff)
        # print(factored_state["Done"], target_diff, target, next_target)
        # print(factored_state["Done"], target_diff, args.target_select(next_factored_state) - args.target_select(factored_state), args.target_select(factored_state), args.target_select(next_factored_state))
        # if np.sum(target_diff) > 0: print(inter_state, factored_state["Block"], target_diff,  use_done)
        # print(factored_state["Block"], next_factored_state["Block"], use_done)
        buffer.add(Batch(act=act, done=use_done, terminated=use_done, truncated=False, true_done=use_done, rew=rew, target=target, next_target=next_target, target_diff=target_diff,
                        trace=trace, inst_trace = inst_trace, obs=inter_state, inter_state=inter_state, parent_state=parent_state, additional_state=additional_state,
                        inter=0.0, proximity=False, proximity_inst=np.zeros(np.array(inst_trace).shape).astype(bool), mask=np.ones(target.shape).astype(float), weight_binary=0)) # the last row are dummy placeholders
        # print(buffer.done.shape, factored_state["Done"], buffer.true_done.shape)
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer