
from Record.record_state import FullRecord


def initialize_environment(args):
    # construct an environment specified by args.env
    if args.env == "Breakout":
        from Environments.Environment.Breakout.breakout_screen import Breakout
        environment = Breakout(frameskip = args.frameskip, breakout_variant=args.breakout_variant)
        environment.seed(args.seed)
    # elif args.env == "Nav2D":
        # environment = Nav2D()
    elif args.env[:6] == "gymenv":
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name= args.env[6:])
        environment.seed(args.seed)
        args.continuous = True
    elif args.env.find("RoboPushing") != -1:
        from Environments.Environment.RoboPushing.robopushing_screen import RoboPushing

        args.continuous = True
        environment = RoboPushingEnvironment(variant=args.variant, horizon=args.time_cutoff, renderable=args.render)
        environment.seed(args.seed)
    elif args.env.find("RoboStick") != -1:
        from Environments.Environment.RoboPushing.robostick_screen import RoboStick

        args.continuous = True
        environment = RoboStickEnvironment(variant=args.variant, horizon=args.time_cutoff, renderable=args.render)
        environment.seed(args.seed)
    if len(args.record_rollouts) != 0
    record = FullRecord(0, args.record_rollouts, args.record_recycle, args.render)
    args.environment = environment
    args.record = record
    return environment, record, args