from Buffer.buffer import FullReplayBuffer, ObjectReplayBuffer
from tianshou.data import Batch
import numpy as np

def fill_full_buffer(full_model, environment, data, args, object_names, norm, predict_dynamics):
    buffer = FullReplayBuffer(len(data), stack_num=1)
    object_buffers = {name: ObjectReplayBuffer(len(data), stack_num=1) for name in environment.object_names}
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        full_state = args.inter_select(factored_state)
        for name in environment.object_names:
            target = norm(full_model.target_select[name](factored_state), name=name)
            next_target = norm(full_model.target_select[name](next_factored_state), name=name)
            target_diff = norm(full_model.target_select[name](next_factored_state) - full_model.target_select[name](factored_state), name=name, form="dyn")

            # compute trace should give back if there is a true interaction at a state
            full_trace = environment.get_full_trace(full_state, act, name)
            proximity = get_full_proximity(full_model, full_state, target, normalize=False)
            object_buffers[name].add(Batch(obs=target, obs_next=next_target, target_diff=target_diff, act=target,
                rew=0, done=False, policy_mask = np.ones(environment.instance_length), param_mask=np.ones(args.pad_size),
                teriminate=False, mapped_act=np.ones(args.pad_size), inter=np.ones(environment.instance_length),
                trace=full_trace, proximity=proximity, weight_binary=False))

        # add one step to the buffer
        use_done = factored_state["Done"] if predict_dynamics else last_done
        act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
        factored_state["Action"] = next_factored_state["Action"]
        rew = factored_state["Reward"]

        # assign selections of the state
        obs = norm(full_state, form="inter")
        obs_next = norm(args.inter_select(next_factored_state), form="inter")
        info, policy, time, option_choice, option_resample = dict(), dict(), i, 0, False
        buffer.add(Batch(obs = obs, obs_next=obs_next, act=act, done=use_done, true_done=use_done, rew=rew,
            info = info, policy=policy, time = time, option_choice=option_choice, option_resample=option_resample))
        # print(buffer.done.shape, factored_state["Done"])
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer
