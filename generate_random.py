import sys, time, cv2
import argparse
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import display_frame
from Environment.Environments.Breakout.breakout_policies import AnglePolicy
import numpy as np

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
    parser.add_argument('--display-frame', action='store_true', default=False,
                        help='shows the frame if renderering')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='amount of frameskip, 1=no frameskip')
    parser.add_argument('--variant', default="default",
                        help='environment variant to use')
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
    environment, record = initialize_environment(args, args)
    if args.angle: 
        policy = AnglePolicy(4)
        angle = np.random.randint(4)

    start = time.time()
    for i in range(args.num_frames):
        if args.angle and environment.ball.paddle: angle = angle = np.random.randint(4)
        action = environment.action_space.sample() if not args.demonstrate else environment.demonstrate()
        action = action if not args.angle else policy.act(environment, angle=angle)
        full_state, reward, done, info = environment.step(action, render=args.render)
        if args.render and args.display_frame: 
            frame = cv2.resize(full_state["raw_state"], (full_state["raw_state"].shape[0] * 5, full_state["raw_state"].shape[1] * 5), interpolation = cv2.INTER_NEAREST)
            display_frame(frame, waitkey=10)
        if record is not None: record.save(full_state['factored_state'], full_state["raw_state"], environment.toString)
        if i % 1000 == 0: print(i, "fps", i / (time.time() - start))
    print("fps", args.num_frames / (time.time() - start))