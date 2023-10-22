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




if __name__ == '__main__':
    env_name = sys.argv[1]
    use_counterfactual = sys.argv[2]
    use_counterfactual = use_counterfactual == "use"
    one_constant = float(sys.argv[3]) if len(sys.argv) >= 4 else -1
    zero_constant = float(sys.argv[4]) if len(sys.argv) >= 4 else -1
    variant = sys.argv[5] if len(sys.argv) > 5 else ""
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
    compute_possible_efficient(env, one_constant, zero_constant, save_path=os.path.join("logs", "exhaustive", env_name + "_" + variant + str(one_constant) + "_" + str(zero_constant) + ".txt"), use_invariant=True )
    # python eval_exhaustive.py Pusher1D use 0.01 0.4
    # python eval_exhaustive.py ForestFire use 0.5 0.2
    # python eval_exhaustive.py GangShoot use 0.2 0.2
