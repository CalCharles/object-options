from Buffer.buffer import InterWeightedReplayBuffer
from tianshou.data import Batch
import numpy as np

def fill_buffer(environment, data, args, object_names, norm):
    buffer = InterWeightedReplayBuffer(len(data), stack_num=1)
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for next_factored_state in data[1:]:
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        rew = factored_state["Reward"]

        # assign selections of the state
        target = norm(args.target_select(factored_state))
        next_target = norm(args.target_select(next_factored_state))
        target_diff = norm(args.target_select(next_factored_state) - args.target_select(factored_state), form="dyn")
        inter_state = norm(args.inter_select(factored_state), form="inter")

        # parent and additional states are unnormalized
        parent_state = args.parent_selectors[object_names.primary_parent](factored_state)
        additional_state = args.parent_select(factored_state)

        # compute trace should give back if there is a true interaction at a state
        trace = environment.get_trace(factored_state, act, object_names)

        # add one step to the buffer
        buffer.add(Batch(act=act, done=last_done, rew=rew, target=target, next_target=next_target, target_diff=target_diff,
                        trace=trace, obs=inter_state, inter_state=inter_state, parent_state=parent_state, additional_state=additional_state,
                        inter=0.0, mask=np.ones(target.shape).astype(float), weight_binary=0)) # the last row are dummy placeholders

        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer
