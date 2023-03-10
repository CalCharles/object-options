from Buffer.FullBuffer.full_buffer import FullReplayBuffer, ObjectReplayBuffer
from Causal.Utils.get_error import get_full_proximity
from tianshou.data import Batch
import numpy as np

def fill_full_buffer(full_model, environment, data, args, object_names, norm, predict_dynamics):
    alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
    buffer = FullReplayBuffer(len(data), stack_num=1)
    object_buffers = {name: ObjectReplayBuffer(len(data), stack_num=1, alpha=alpha, beta=beta) for name in environment.object_names}
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        # assign general components
        use_done = next_factored_state["Done"]#factored_state["Done"].squeeze() if predict_dynamics else last_done
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        full_state = args.inter_select(factored_state)
        rew = factored_state["Reward"]
        # print(i, use_done, last_done, predict_dynamics)
        full_traces = environment.get_full_trace(factored_state, act)
        for name in environment.object_names:
            denorm_target = full_model.target_selectors[name](factored_state)
            target = norm(denorm_target, name=name)
            next_target = norm(full_model.target_selectors[name](next_factored_state), name=name)
            target_diff = norm(full_model.target_selectors[name](next_factored_state) - full_model.target_selectors[name](factored_state), name=name, form="dyn")

            # get the trace for this object class
            if environment.object_instanced[name] > 1: # if there are multiple instances of the object, it is object_instance x other objects for the mask
                full_trace = np.concatenate([full_traces[name + str(i)] for i in range(environment.object_instanced[name])], axis=0)
                inter = np.ones((environment.object_instanced[name] * environment.instance_length))
            else: 
                full_trace = full_traces[name]
                inter = np.ones(environment.instance_length)
            # print(name)
            proximity = get_full_proximity(full_model, full_state, denorm_target, normalized=False)
            object_buffers[name].add(Batch(obs=target, obs_next=next_target, target_diff=target_diff, act=target, param=target,
                rew=0, done=use_done, policy_mask = np.ones(environment.instance_length), param_mask=np.ones(args.pad_size),
                terminate=False, mapped_act=np.ones(args.pad_size), inter=inter, info=dict(), policy=dict(), 
                trace=full_trace, proximity=proximity, weight_binary=0))
            # print(name, full_trace, target_diff, full_model.target_selectors[name](next_factored_state), full_model.target_selectors[name](factored_state))
            
        # assign selections of the state
        obs = norm(full_state, form="inter")
        obs_next = norm(args.inter_select(next_factored_state), form="inter")
        info, policy, time, option_choice, option_resample = dict(), dict(), i, 0, False
        buffer.add(Batch(obs = obs, obs_next=obs_next, act=act, done=use_done, true_done=use_done, rew=rew, true_reward=rew,
            info = info, policy=policy, time = time, option_choice=option_choice, option_resample=option_resample))
        # print(buffer.done.shape, factored_state["Done"])
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer, object_buffers
