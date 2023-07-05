from Record.file_management import load_from_pickle
from Record.record_state import FullRecord
from Environment.gymnasium_wrapper import GymnasiumWrapper
import os
from Environment.gymnasium_wrapper import GymnasiumWrapper


def initialize_environment(args, record_args):
    # construct an environment specified by args.env
    if args.env == "Breakout":
        from Environment.Environments.Breakout.breakout_screen import Breakout
        environment = Breakout(frameskip = args.frameskip, breakout_variant=args.variant, fixed_limits=args.fixed_limits)
        print(args.seed)
        environment.seed(args.seed)
    elif args.env == "Asteroids":
        from Environment.Environments.Asteroids.asteroids import Asteroids
        environment = Asteroids(frameskip = args.frameskip, variant=args.variant, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    elif args.env == "Sokoban":
        from Environment.Environments.Sokoban.sokoban import Sokoban
        environment = Sokoban(frameskip = args.frameskip, variant=args.variant, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    elif args.env == "TaxiCar":
        from Environment.Environments.TaxiCar.taxi_car import TaxiCar
        environment = TaxiCar(frameskip = args.frameskip, variant=args.variant, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    elif args.env == "RandomDistribution":
        from Environment.Environments.RandomDistribution.random_distribution import RandomDistribution
        if len(args.load_environment) > 0: environment = load_from_pickle(os.path.join(args.load_environment, "environment.pkl"))
        else: environment = RandomDistribution(frameskip = args.frameskip, variant=args.variant, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    # elif args.env == "Nav2D":
        # environment = Nav2D()
    elif args.env[:6] == "gymenv":
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name= args.env[6:], fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
        args.continuous = True
    elif args.env.find("RoboPushing") != -1:
        from Environment.Environments.RoboPushing.robopushing_screen import RoboPushing

        args.continuous = True
        environment = RoboPushing(variant=args.variant, horizon=args.horizon, renderable=args.render, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    elif args.env.find("AirHockey") != -1:
        from Environment.Environments.AirHockey.air_hockey import RobosuiteAirHockey

        args.continuous = True
        environment = RobosuiteAirHockey(variant=args.variant, horizon=args.horizon, renderable=args.render, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    elif args.env.find("RoboStick") != -1:
        from Environment.Environments.RoboPushing.robostick_screen import RoboStick

        args.continuous = True
        environment = RoboStick(variant=args.variant, horizon=args.time_cutoff, renderable=args.render, fixed_limits=args.fixed_limits)
        environment.seed(args.seed)
    if args.gym_to_gymnasium:
        environment = GymnasiumWrapper(environment)
    record = FullRecord(0, record_args.record_rollouts, record_args.record_recycle, args.render) if record_args is not None and len(record_args.record_rollouts) != 0 else None
    args.environment = environment
    args.record = record
    return environment, record