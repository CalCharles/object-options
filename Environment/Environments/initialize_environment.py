
from Record.record_state import FullRecord


def initialize_environment(args, record_args):
    # construct an environment specified by args.env
    if args.env == "Breakout":
        from Environment.Environments.Breakout.breakout_screen import Breakout
        environment = Breakout(frameskip = args.frameskip, breakout_variant=args.variant)
        environment.seed(args.seed)
    if args.env == "Asteroids":
        from Environment.Environments.Asteroids.asteroids import Asteroids
        environment = Asteroids(frameskip = args.frameskip, variant=args.variant)
        environment.seed(args.seed)
    if args.env == "Sokoban":
        from Environment.Environments.Sokoban.sokoban import Sokoban
        environment = Sokoban(frameskip = args.frameskip, variant=args.variant)
        environment.seed(args.seed)
    # elif args.env == "Nav2D":
        # environment = Nav2D()
    elif args.env[:6] == "gymenv":
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name= args.env[6:])
        environment.seed(args.seed)
        args.continuous = True
    elif args.env.find("RoboPushing") != -1:
        from Environment.Environments.RoboPushing.robopushing_screen import RoboPushing

        args.continuous = True
        environment = RoboPushing(variant=args.variant, renderable=args.render)
        environment.seed(args.seed)
    elif args.env.find("RoboStick") != -1:
        from Environment.Environments.RoboPushing.robostick_screen import RoboStick

        args.continuous = True
        environment = RoboStick(variant=args.variant, horizon=args.time_cutoff, renderable=args.render)
        environment.seed(args.seed)
    record = FullRecord(0, record_args.record_rollouts, record_args.record_recycle, args.render) if len(record_args.record_rollouts) != 0 else None
    args.environment = environment
    args.record = record
    return environment, record