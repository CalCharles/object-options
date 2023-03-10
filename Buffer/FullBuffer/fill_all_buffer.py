from Buffer.FullBuffer.all_buffer import AllReplayBuffer
from Causal.Utils.get_error import get_full_proximity
from tianshou.data import Batch
import numpy as np

def fill_all_buffer(full_model, environment, data, args, object_names, norm, predict_dynamics):
    alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
    buffer = AllReplayBuffer(len(data), stack_num=1)
    object_buffers = {name: ObjectReplayBuffer(len(data), stack_num=1, alpha=alpha, beta=beta) for name in environment.object_names}
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        # print("adding", i)

        # assign general components
        use_done = next_factored_state["Done"]#factored_state["Done"].squeeze() if predict_dynamics else last_done
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        full_state = args.inter_select(factored_state)
        rew = factored_state["Reward"]
        # print(i, use_done, last_done, predict_dynamics)
        # "inter", "obs_diff", "trace", "proximity", "weight_binary"
        inter = np.ones((environment.instance_length, environment.instance_length)) # n x n matrix of interactions
        obs_diff = {name: norm(full_model.target_selectors[name](next_factored_state) - full_model.target_selectors[name](factored_state), name=name, form="dyn") for name in environment.all_names}
        
        full_traces = environment.get_full_trace(factored_state, act)
        trace = np.stack([full_traces[name] for name in environment.all_names], axis=0)
        proximity = np.stack([get_full_proximity(full_model, full_state, full_model.target_selectors[name](factored_state), normalized=False) for name in environment.all_names], axis=0)
        weight_binary = np.ones(environment.instance_length)

        # assign selections of the state
        obs = norm(full_state, form="inter")
        obs_next = norm(args.inter_select(next_factored_state), form="inter")
        info, policy, time, option_choice, option_resample = dict(), dict(), i, 0, False
        buffer.add(Batch(obs = obs, obs_next=obs_next, act=act, done=use_done, true_done=use_done, rew=rew, true_reward=rew,
            info = info, policy=policy, time = time, option_choice=option_choice, option_resample=option_resample,
            inter=inter, trace=full_traces, obs_diff = obs_diff, proximity = proximity, weight_binary=weight_binary))
        # print(buffer.done.shape, factored_state["Done"])
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer, object_buffers
