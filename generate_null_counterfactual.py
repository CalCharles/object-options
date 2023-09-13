# generates a batch of counterfactual data, will insert a lot of two step trajectories, so sampling non-done states is preferred
from Causal.Sampling.sampling import samplers
from State.feature_selector import construct_object_selector
from Environment.Environments.initialize_environment import initialize_environment
from Environment.Environments.Breakout.breakout_policies import AnglePolicy
import numpy as np
from generate_random import generate_random
import itertools
import argparse

def _record_counterfactuals(states, environment, record):
    for state, next_state in zip(states, states[1:]):
        if state['factored_state']["Done"]: # ignore done transitions
             continue
        # assumes actions can be nulled
        all_combinations = list()
        for i in range(1, len(environment.all_names[:-2]) + 1):
            all_combinations += list(itertools.combinations(np.arange(len(environment.all_names[:-2])),i))
        all_combinations.pop(all_combinations.index(tuple(np.nonzero(state['factored_state']["VALID_NAMES"])[0]))) # don't counterfactually resample
        for i in np.choice(np.arange(len(all_combinations)), size=(min(len(all_combinations), 10),)): # creates at most 10 counterfactuals
            combination = all_combinations[i]
            valid_names = [environment.all_names[c] for c in combination]
            environment.set_from_factored_state(state, valid_names)
            state = environment.get_state()
            record.save(state['factored_state'], state["raw_state"], environment.toString)
            full_state, reward, done, info = environment.step(next_state['factored_state']["Action"][-1] if environment.discrete_actions else next_state['factored_state']["Action"])
            full_state['factored_state']['Done'] = True
            record.save(full_state['factored_state'], full_state["raw_state"], environment.toString)


def generate_null_counterfactual(args, states= None):
    if states is not None:
        environment, record = initialize_environment(args, args)
        if args.angle: 
                policy = AnglePolicy(4)
                angle = np.random.randint(4)
        if args.variant == "proximity":
                environment.sampler = samplers["exist"](obj_dim=5, target_select=construct_object_selector(["Block"], environment),parent_select=None,additional_select=None,test_sampler=False,mask=None)
                environment.reset()
        _record_counterfactuals(states, environment, record)
    for b in range(int(np.ceil(args.num_frames // 10000))):
        environment, record, states = generate_random(args, save_state=True)
        _record_counterfactuals(states)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random data from an environment')
    parser.add_argument('--record-rollouts', default = "",
                        help='base directory to save results')
    parser.add_argument('--record-recycle', type=int, default=-1,
                        help='max number of frames to keep, default: -1 not used')
    parser.add_argument('--env',
                        help='environment to run on')
    parser.add_argument('--render', action='store_true', default=False,
                        help='run the pushing gripper domain')
    parser.add_argument('--fixed-limits', action='store_true', default=False,
                        help='fixes the norm boundaries between objects')
    parser.add_argument('--display-frame', action='store_true', default=False,
                        help='shows the frame if renderering')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='amount of frameskip, 1=no frameskip')
    parser.add_argument('--variant', default="default",
                        help='environment variant to use')
    parser.add_argument('--load-environment', default="",
                        help='load the environment from here')
    parser.add_argument('--horizon', type=int, default=-1,
                        help='time cutoff for environment resets, defaults -1 (no cutoff)')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--seed', type=int, default=-1,
                        help='number of frames to run')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random actions')
    parser.add_argument('--angle', action='store_true', default=False,
                        help='uses the angle policy if true')
    args = parser.parse_args()
    args.gym_to_gymnasium = False
    args.seed = args.seed if args.seed >= 0 else np.random.randint(10000)
    args.render = args.demonstrate or args.render
    # first argument is num frames, second argument is save path
    generate_null_counterfactual(args)