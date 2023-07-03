import sys, time, cv2, os
import argparse
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import display_frame, display_param, save_to_pickle
from Causal.Sampling.sampling import samplers
import numpy as np
from State.feature_selector import construct_object_selector
from State.object_dict import ObjDict

def generate_args():
    args = ObjDict()
    args.record_rollouts = ""
    args.record_recycle = -1
    args.env = "Breakout"
    args.render = False
    args.fixed_limits = False
    args.display_frame = False
    args.frameskip = 1
    args.variant = "default"
    args.load_environment = "" 
    args.horizon = -1
    args.num_frames = 10000 
    args.seed = -1
    args.demonstrate = False
    args.angle = False
    args.seed = args.seed if args.seed >= 0 else np.random.randint(10000)
    args.render = args.demonstrate or args.render
    args.gym_to_gymnasium = False
    return args

def generate_random(args):
    environment, record = initialize_environment(args, args)
    if args.angle: 
        policy = AnglePolicy(4)
        angle = np.random.randint(4)
    if args.variant == "proximity":
        environment.sampler = samplers["exist"](obj_dim=5, target_select=construct_object_selector(["Block"], environment),parent_select=None,additional_select=None,test_sampler=False,mask=None)
        environment.reset()
    start = time.time()
    for i in range(args.num_frames):
        if args.angle and environment.ball.paddle: angle = angle = np.random.randint(4)
        action = environment.action_space.sample() if not args.demonstrate else environment.demonstrate()
        # action = np.ones(environment.action_space._shape) * (-0.5 + i/args.num_frames) if not args.demonstrate else environment.demonstrate()
        # action = np.random.rand(*environment.action_space._shape) * (environment.action_space.high - environment.action_space.low) + environment.action_space.low if not args.demonstrate else environment.demonstrate()
        action = action if not args.angle else policy.act(environment, angle=angle)
        full_state, reward, done, info = environment.step(action, render=args.render)
        if args.render and args.display_frame: 
            frame = cv2.resize(full_state["raw_state"], (full_state["raw_state"].shape[0] * 5, full_state["raw_state"].shape[1] * 5), interpolation = cv2.INTER_NEAREST)
            display_frame(frame, waitkey=10)
        if args.render and args.variant == "proximity" and args.display_frame: display_param(full_state['raw_state'], param=environment.sampler.param[:2], waitkey=100, rescale = 10, dot=False)
        elif args.render and args.display_frame: display_frame(full_state['raw_state'], rescale=10, waitkey=30)
        if record is not None: record.save(full_state['factored_state'], full_state["raw_state"], environment.toString)
        if i % 1000 == 0: print(i, "fps", i / (time.time() - start))
    if args.env not in ["RoboPushing", "RoboStick", "AirHockey"]: save_to_pickle(os.path.join(args.record_rollouts, "environment.pkl"), environment)
    print("fps", args.num_frames / (time.time() - start))

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
    args.seed = args.seed if args.seed >= 0 else np.random.randint(10000)
    args.render = args.demonstrate or args.render
    # first argument is num frames, second argument is save path
    generate_random(args)
