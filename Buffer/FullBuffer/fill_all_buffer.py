from Buffer.FullBuffer.all_buffer import AllReplayBuffer
from Causal.Utils.get_error import get_full_proximity
from tianshou.data import Batch
import numpy as np

def fill_all_buffer(full_model, environment, data, args, object_names, norm, predict_dynamics):
    alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
    buffer = AllReplayBuffer(len(data), stack_num=1)
    factored_state = data[0]
    for i, next_factored_state in enumerate(data[1:]):
        # print("adding", i)

        # assign general components
        use_done = next_factored_state["Done"]#factored_state["Done"].squeeze() if predict_dynamics else last_done
        next_factored_state["Done"] = np.array([0]) # don't predict dones
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        full_state = args.inter_select(factored_state)
        rew = factored_state["Reward"]
        # print(i, use_done, last_done, predict_dynamics)
        # "inter", "obs_diff", "trace", "proximity", "weight_binary"
        inter = np.ones((environment.instance_length, environment.instance_length)) # n x n matrix of interactions
        obs_diff = args.inter_select({name: norm(full_model.target_selectors[name](next_factored_state) - full_model.target_selectors[name](factored_state), name=name, form="dyn") for name in environment.all_names})
        if "VALID_NAMES" in factored_state: valid = factored_state["VALID_NAMES"][:-2] # don't include reward or done in validity vector
        else: valid = np.ones((len(environment.all_names)))[:-2]

        full_traces = environment.get_full_trace(factored_state, act)
        trace = np.stack([full_traces[name] for name in environment.all_names], axis=0).astype(float)
        trace = np.pad(trace, (0, 2))[:trace.shape[0]]
        proximity = np.stack([get_full_proximity(full_model, full_state, full_model.target_selectors[name](factored_state), normalized=False) for name in environment.all_names], axis=0)
        weight_binary = np.ones(1)

        # assign selections of the state
        obs = norm(full_state, form="inter")
        target = norm(args.target_select(factored_state), name="all")
        next_target = norm(args.target_select(next_factored_state), name="all")
        target_diff = norm(args.target_select({name: full_model.target_selectors[name](next_factored_state) - full_model.target_selectors[name](factored_state) for name in environment.all_names}), name="all", form="dyn")
        obs_next = norm(args.inter_select(next_factored_state), form="inter")
        info, policy, time, option_choice, option_resample = dict(), dict(), i, 0, False
        buffer.add(Batch(obs = obs, obs_next=obs_next, act=act, done=use_done, terminated=False, truncated=use_done, true_done=use_done, rew=rew, true_reward=rew,
            info = info, policy=policy, time = time, option_choice=option_choice, option_resample=option_resample,
            target=target, next_target=next_target, target_diff=target_diff,
            inter=inter, trace=trace, obs_diff = obs_diff, proximity = proximity, weight_binary=weight_binary, valid=valid))
        # print(buffer.done.shape, factored_state["Done"])
        factored_state = next_factored_state
    return buffer
