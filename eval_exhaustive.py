import numpy as np
from Environment.Environments.Pusher1D.pusher1D import Pusher1D
from Environment.Environments.ACDomains.Domains.forest_fire import ForestFire
from Environment.Environments.ACDomains.Domains.rock_throwing import RockThrowing
from Environment.Environments.ACDomains.Domains.gang_shoot import GangShoot
from Environment.Environments.ACDomains.Domains.halt_charge import HaltCharge
from Environment.Environments.ACDomains.Domains.train import Train
from Environment.Environments.ACDomains.Domains.voting import Voting
from Environment.Environments.ACDomains.Domains.mod_DAG import ModDAG
import sys, os
import time
from Causal.EMFAC.exhaustive_em import compute_possible_efficient

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random data from an environment')
    parser.add_argument('--env', default = "",
                        help='base directory to save results')
    parser.add_argument('--alpha', type=float, nargs=2, default=(-1, -1),
                        help='max number of frames to keep, default: -1 not used')
    parser.add_argument('--use-zero', action ='store_true', default=False,
                        help='uses the zero binary')
    parser.add_argument('--use-counterfactual', action ='store_true', default=False,
                        help='uses counterfactual vs possible splitting')
    parser.add_argument('--variant', default = "",
                        help='the variant form used for the environment')
    args = parser.parse_args()
    env_name = args.env
    use_counterfactual = args.use_counterfactual
    one_constant, zero_constant = args.alpha
    variant = args.variant
    print(env_name, use_counterfactual, variant)
    if env_name == "Pusher1D":
        env = Pusher1D(cf_states = use_counterfactual)
    elif env_name == "ForestFire":
        env = ForestFire(cf_states = use_counterfactual)
    elif env_name == "RockThrowing":
        env = RockThrowing(cf_states = use_counterfactual)
    elif env_name == "GangShoot":
        env = GangShoot(cf_states = use_counterfactual)
    elif env_name == "HaltCharge":
        env = HaltCharge(cf_states = use_counterfactual)
    elif env_name == "Train":
        env = Train(cf_states = use_counterfactual)
    elif env_name == "Voting":
        env = Voting(cf_states = use_counterfactual)
    elif env_name == "ModDAG":
        env = ModDAG(variant=variant,cf_states = use_counterfactual)
    compute_possible_efficient(env, one_constant, zero_constant, use_zero = args.use_zero, save_path=os.path.join("logs", "exhaustive", env_name + "_" + variant + str(one_constant) + "_" + str(zero_constant) + ".txt"), use_invariant=True )
    # python eval_exhaustive.py --env Pusher1D --use-counterfactual --use-zero --alpha 0.01 0.4
    # python eval_exhaustive.py --env ForestFire --use-counterfactual --use-zero --alpha 0.5 0.2
    # python eval_exhaustive.py --env GangShoot --use-counterfactual --use-zero --alpha 0.2 0.2
