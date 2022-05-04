from Environments.Pushing.screen import run, RandomPolicy
import sys
import argparse
from Environment.Environments.initialize_environment import initialize_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('--record-rollouts',
                        help='base directory to save results')
    parser.add_argument('--render', action='store_true', default=False,
                        help='run the pushing gripper domain')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='amount of frameskip, 1=no frameskip')
    parser.add_argument('--variant', default="default",
                        help='environment variant to use')
    parser.add_argument('--time-cutoff', type=int, default=-1,
                        help='time cutoff for environment resets, defaults -1 (no cutoff)')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion TODO: implement')
    args = parser.parse_args()
    # first argument is num frames, second argument is save path
    environment, record = initialize_environment(args)
    for i in range(int(sys.argv[1])):
        action = environment.action_space.sample()
        environment.step(action)