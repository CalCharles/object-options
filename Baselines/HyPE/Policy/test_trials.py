from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model
from Record.file_management import write_string
import os


def collect_test_trials(logger, skill, test_collector, num_sample, num_repeats, i, trials, random, demonstrate_skill=False):
    '''
    collect trials with the test collector
    the environment is reset before starting these trials
    most of the inputs are used for printing out the results of these trials 
    term_step is a tuple of the max number of episodes, and max number of steps for a testing episode
    '''
    skill.toggle_test(True) # switches skill to testing mode
    test_collector.reset_env()
    # print("starting trials")
    results = list()
    for j in range(trials):
        print("next_trial", j)
        skill.reset(test_collector.data.full_state)
        result = test_collector.collect(num_sample, num_repeats, episodes=1, random=random, demonstrate=demonstrate_skill)
        result['n/tr'] = max(1, result['n/tr']) # at least one (virtual) epsiode occurs before the end, for testing purposes
        result['n/ep'] = max(1, result['n/ep']) # at least one (virtual) epsiode occurs before the end, for testing purposes
        logger.log_results(result)
        # print(result)
        results.append(result)
    skill.toggle_test(False) # switched option back to training mode
    return results
