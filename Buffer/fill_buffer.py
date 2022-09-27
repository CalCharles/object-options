from Buffer.buffer import InterWeightedReplayBuffer, FullReplayBuffer, ObjectReplayBuffer
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

        # parent and additional states are unnormalized
        parent_state = args.parent_select(factored_state)
        additional_state = args.additional_select(factored_state)

        # compute trace should give back if there is a true interaction at a state
        inst_trace = environment.get_trace(factored_state, act, object_names)
        if len(inst_trace) > 1: trace = [np.sum(inst_trace)] # we don't store per-instance traces
        else: trace = inst_trace.copy()

        # add one step to the buffer
        use_done = factored_state["Done"] if predict_dynamics else last_done
        # if np.linalg.norm(target) > 0: print(last_done, use_done, next_factored_state["Done"], args.target_select(next_factored_state) - args.target_select(factored_state), target_diff, factored_state["Target"],factored_state["Block"])
        # if np.linalg.norm(target_diff) > 0: print(i, last_done, use_done, factored_state["Action"], next_factored_state["Done"], args.inter_select(factored_state), target)
        # else: print(i, factored_state["Action"])
        # REMOVE THIS LINE
        # if np.max(np.abs(target_diff)) > 1: use_done = True 
        # print(i, use_done, args.target_select(next_factored_state) - args.target_select(factored_state), target_diff)
        # print(factored_state["Done"], target_diff, target, next_target)
        # print(factored_state["Done"], target_diff, args.target_select(next_factored_state) - args.target_select(factored_state), args.target_select(factored_state), args.target_select(next_factored_state))
        # if np.sum(target_diff) > 0: print(inter_state, factored_state["Block"], target_diff,  use_done)
        buffer.add(Batch(act=act, done=use_done, true_done=use_done, rew=rew, target=target, next_target=next_target, target_diff=target_diff,
                        trace=trace, inst_trace = inst_trace, obs=inter_state, inter_state=inter_state, parent_state=parent_state, additional_state=additional_state,
                        inter=0.0, proximity=False, proximity_inst=np.zeros(np.array(inst_trace).shape).astype(bool), mask=np.ones(target.shape).astype(float), weight_binary=0)) # the last row are dummy placeholders
        # print(buffer.done.shape, factored_state["Done"])
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer

def fill_full_buffer(environment, data, args, object_names, norm, predict_dynamics):

    buffer = FullReplayBuffer(len(data), stack_num=1)
    object_buffers = {name: ObjectReplayBuffer(len(data), stack_num=1) for name in environment.object_names}
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        rew = factored_state["Reward"]
        use_done = factored_state["Done"] if predict_dynamics else last_done
        state = norm(args.inter_select(factored_state), form="inter")
        next_state = norm(args.inter_select(factored_state), form="inter")
        full_trace = environment.get_full_trace(factored_state, act)

        for target_name in environment.all_names:
            # assign selections of the state
            target = norm(args.target_select(factored_state))
            next_target = norm(args.target_select(next_factored_state))
            target_diff = norm(args.target_select(next_factored_state) - args.target_select(factored_state), form="dyn")
            inter_state = norm(args.inter_select(factored_state), form="inter")

            # parent and additional states are unnormalized
            parent_state = args.parent_select(factored_state)
            additional_state = args.additional_select(factored_state)

            # compute trace should give back if there is a true interaction at a state

            object_buffers.add(Batch(obs=target, obs_next=next_target, act=act, done=use_done, true_done=use_done, rew=rew, target_diff=target_diff,
                            trace=full_trace[target_name], mapped_act=act, option_choice=-1, param=target, terminate=False,
                            policy_mask= np.zeros(len(environment.object_names)), param_mask = np.zeros(len(environment.object_names)), # dummy values
                            inter=np.zeros(len(environment.all_names)), proximity=np.zeros(np.array(inst_trace).shape).astype(bool), mask=np.ones(target.shape).astype(float), weight_binary=0)) # the last row are dummy placeholders

        # add the full states to the buffers
        buffer.add(Batch(obs=state, obs_next = next_state, act=act, done=use_done, rew=rew, true_done=use_done, true_reward=rew, time=1, option_resample=True)) # the last row are dummy placeholders
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer
