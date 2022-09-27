from Record.file_management import display_frame, load_from_pickle, save_to_pickle
from Environment.Environments.initialize_environment import initialize_environment
import numpy as np

def test_collect_environment(args):
    on_off = [np.random.randint(2) for i in range(int(args.train.num_iters // 10))]
    np.random.seed(0)
    env, rec = initialize_environment(args.environment, args.record)
    acts, sequence_record, setting_start, setting_acts, setting_record = list(), list(), list(), list(), list()
    on = on_off[0]
    for i in range(args.train.num_iters):
        action = env.action_space.sample() if not args.environment.demonstrate else env.demonstrate()
        full_state, reward, done, info = env.step(action, render=args.environment.render or args.environment.demonstrate)
        display_frame(env.get_state()['raw_state'])
        if rec is not None: rec.save(full_state['factored_state'], full_state["raw_state"], env.toString)
        sequence_record.append(full_state['factored_state'])
        acts.append(action)
        if on: setting_acts.append(action)
        if i % 10 == 0:
            if on:
                setting_record.append(full_state['factored_state'])
            on = on_off[int(i // 10)]
            if on:
                setting_start.append(full_state['factored_state'])
    save_to_pickle("Testing/testing_data/environment_acts_" + args.environment.env + "_" + args.environment.variant + ".pkl", acts)
    save_to_pickle("Testing/testing_data/environment_" + args.environment.env + "_" + args.environment.variant + ".pkl", sequence_record)
    save_to_pickle("Testing/testing_data/environment_setting_acts_" + args.environment.env + "_" + args.environment.variant + ".pkl", setting_acts)
    save_to_pickle("Testing/testing_data/environment_setting_" + args.environment.env + "_" + args.environment.variant + ".pkl", setting_record)
    save_to_pickle("Testing/testing_data/environment_setting_start_" + args.environment.env + "_" + args.environment.variant + ".pkl", setting_start)

def compare_factored(er, state):
    result = list()
    for n in er.keys():
        # print(state['factored_state'][n], er[n])
        if type(state['factored_state'][n][0]) == np.bool_: result.append(state['factored_state'][n] ==  er[n])
        else: result.append(np.linalg.norm(state['factored_state'][n] -  er[n]))
    return result

def test_environment_sequence(args):
    np.random.seed(0)
    expected_results = load_from_pickle("Testing/testing_data/environment_" + args.environment.env + "_" + args.environment.variant + ".pkl")
    actions = load_from_pickle("Testing/testing_data/environment_acts_" + args.environment.env + "_" + args.environment.variant + ".pkl")
    env, rec = initialize_environment(args.environment, args.record)
    result = list()
    for act, er, i in zip(actions, expected_results, range(args.train.num_iters)):
        state, reward, done, info = env.step(act)
        # display_frame(env.get_state()['raw_state'])
        comp = compare_factored(er, state)
        result.append(comp)
        print(comp)
    return result

def test_environment_setting(args):
    np.random.seed(0)
    res = list()
    setting_states = load_from_pickle("Testing/testing_data/environment_setting_start_" + args.environment.env + "_" + args.environment.variant + ".pkl")
    expected_results = load_from_pickle("Testing/testing_data/environment_setting_" + args.environment.env + "_" + args.environment.variant + ".pkl")
    expected_action = load_from_pickle("Testing/testing_data/environment_setting_acts_" + args.environment.env + "_" + args.environment.variant + ".pkl")
    env, rec = initialize_environment(args.environment, args.record)
    counter = 0
    result = list()
    for ss, er in zip(setting_states, expected_results):
        env.set_from_factored_state(ss)
        for i in range(10):
            state, reward, done, info = env.step(expected_action[counter])
            counter += 1 
        comp = compare_factored(er, state)
        result.append(comp)
        print("set", comp)
    return result

# debug notes: Breakout: issue with the ball resets and setting the random seed
