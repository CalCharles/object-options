from Buffer.FullBuffer.full_buffer import FullReplayBuffer, ObjectReplayBuffer
from Causal.Utils.get_error import get_full_proximity
from tianshou.data import Batch
import numpy as np
import copy

def get_passive_name(name):
    return "#" + name[1:]

def fill_full_buffer(full_model, environment, data, args, object_names, norm, predict_dynamics, outcome_variable=""):
    alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
    buffer = FullReplayBuffer(len(data), stack_num=1)
    object_buffers = {name: ObjectReplayBuffer(len(data), stack_num=1, alpha=alpha, beta=beta) for name in environment.object_names}
    factored_state = data[0]
    if len(outcome_variable) > 0 and get_passive_name(outcome_variable) in full_model.target_selectors: args.inter_select.names.pop(args.inter_select.names.find(get_passive_name(outcome_variable)))
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    for i, next_factored_state in enumerate(data[1:]):
        # assign general components
        # TODO: unify breakout environment dones vs random vector dones
        if environment.name != "Breakout":
            use_done = next_factored_state["Done"]#factored_state["Done"]#factored_state["Done"].squeeze() if predict_dynamics else last_done
        else:
            use_done = factored_state["Done"]
        if args.full_inter.predict_next_state:
            act = next_factored_state["Action"][-1] if environment.discrete_actions else next_factored_state["Action"]
            factored_state["Action"] = next_factored_state["Action"]
            full_state = args.inter_select(factored_state)
        else: # don't shift, the actions are used for the current state evaluation
            act = factored_state["Action"][-1] if environment.discrete_actions else factored_state["Action"]
            new_factored = copy.deepcopy(factored_state)
            if get_passive_name(outcome_variable) in full_model.target_selectors: new_factored[outcome_variable] = new_factored[get_passive_name(outcome_variable)]
            else: new_factored[outcome_variable] = new_factored[outcome_variable] * 0.0 # block out the outcome variable for this step
            full_state = args.inter_select(new_factored)
        rew = factored_state["Reward"]
        if "VALID_NAMES" in factored_state: valid = factored_state["VALID_NAMES"][:-2] # don't include reward or done in validity vector
        else: valid = np.ones((len(environment.all_names)))[:-2]
        # print(i, use_done, last_done, predict_dynamics)
        # print(factored_state["$B"])
        full_traces = environment.get_full_trace(factored_state, act, outcome_variable=outcome_variable)
        # if args.full_inter.predict_next_state: full_traces = environment.get_full_trace(factored_state, act, outcome_variable=outcome_variable)
        # else: full_traces = environment.get_full_trace(next_factored_state, act, outcome_variable=outcome_variable)
        for name in environment.object_names:
            denorm_target = full_model.target_selectors[name](factored_state)
            target = norm(denorm_target, name=name)
            if args.full_inter.predict_next_state:
                next_target = norm(full_model.target_selectors[name](next_factored_state), name=name)
                target_diff = norm(full_model.target_selectors[name](next_factored_state) - full_model.target_selectors[name](factored_state), name=name, form="dyn")
                unnorm_diff = full_model.target_selectors[name](next_factored_state).astype(float) - full_model.target_selectors[name](factored_state).astype(float)
            else:
                use_done = next_factored_state["Done"]
                target = target * 0.0 # masks out the target so it won't be applied as an input
                if len(outcome_variable) != 0 and get_passive_name(outcome_variable) in full_model.target_selectors: target = norm(full_model.target_selectors[get_passive_name(outcome_variable)](factored_state), name=get_passive_name(outcome_variable))
                next_target = norm(full_model.target_selectors[name](factored_state), name=name)
                target_diff = full_model.target_selectors[name](next_factored_state).astype(float) - full_model.target_selectors[name](factored_state).astype(float)

            # if name == "ybb": print(target_diff, valid)
            # get the trace, valid for this object class
            if environment.object_instanced[name] > 1: # if there are multiple instances of the object, it is object_instance x other objects for the mask
                full_trace = np.concatenate([full_traces[name + str(i)] for i in range(environment.object_instanced[name])], axis=0)
                inter = np.ones((environment.object_instanced[name] * environment.instance_length))
                # get the valid corresponding to this object class, TODO: assumes contiguous names
            else: 
                full_trace = full_traces[name]
                inter = np.ones(environment.instance_length)
            

            # if name == outcome_variable: print(name, full_trace, full_model.target_selectors["$B"](factored_state), next_target, factored_state, use_done)
            proximity = get_full_proximity(full_model, full_state, denorm_target, normalized=False)
            if (args.collect.omit_done and not use_done) or (not args.collect.omit_done):
                object_buffers[name].add(Batch(obs=target, obs_next=next_target, target_diff=target_diff, act=target, param=target,
                    rew=0, done=use_done, policy_mask = np.ones(environment.instance_length), param_mask=np.ones(args.pad_size),
                    terminate=False, terminated=False, truncated=use_done, mapped_act=np.ones(args.pad_size), inter=inter, info=dict(), policy=dict(), 
                    trace=full_trace, proximity=proximity, weight_binary=0, valid=valid))
            #     if name == outcome_variable: 
            #         # hit_block = factored_state["Block" + str(np.nonzero(full_trace[3:])[0][0])] if len(np.nonzero(full_trace[3:])[0]) > 0 else 0
            #         print(name, full_trace,denorm_target, target_diff, target, next_target, use_done)
            # else:
            #     if name == outcome_variable: 
            #         print("SKIPPED", full_trace,denorm_target, target_diff, target, next_target, use_done)
            
            
        # assign selections of the state
        obs = norm(full_state, form="inter")
        obs_next = norm(args.inter_select(next_factored_state), form="inter")
        info, policy, time, option_choice, option_resample = dict(), dict(), i, 0, False
        # print(obs, obs_next, norm.inter_norm, factored_state["Action"])
        if (args.collect.omit_done and not use_done) or (not args.collect.omit_done):
            buffer.add(Batch(obs = obs, obs_next=obs_next, act=act, done=use_done, true_done=use_done, rew=rew, true_reward=rew,
                info = info, policy=policy, time = time, terminated=False, truncated=use_done, option_choice=option_choice, option_resample=option_resample, valid=valid))
        last_done = factored_state["Done"]
        factored_state = next_factored_state
    return buffer, object_buffers
