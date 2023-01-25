import numpy as np
import copy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

class TemporalAggregator():
    def __init__(self, name="", sum_reward=False):
        self.current_data = Batch()
        self.name= name
        self.next_action = False
        self.next_param = False
        self.keep_next = True
        self.temporal_skip = False
        self.sum_reward = False # sums reward for the length of the trajectory
        self.time_counter = 0 # counts the number of time steps in the temporal extension

    def reset(self, data):
        self.current_data = copy.deepcopy(data)
        self.keep_next = True
        self.time_counter = 0

    def update(self, data):
        self.current_data = copy.deepcopy(data)

    def aggregate(self, data):
        # updates "next" values to the current value, and combines dones, rewards
        added = False
        if self.keep_next: 
            self.current_data = copy.deepcopy(data)
            self.keep_next = False
        # update state components
        self.current_data.update(next_full_state = data.next_full_state, next_target=data.next_target, obs_next=data.obs_next)
        # update  termination and resampling components
        self.current_data.done = [np.any(self.current_data.done) + np.any(data.done)]
        self.current_data.true_done = [np.any(self.current_data.true_done) + np.any(data.true_done)] # basically an OR
        self.current_data.resample = data.resample
        self.current_data.info["TimeLimit.truncated"] = data.info["TimeLimit.truncated"] if "TimeLimit.truncated" in data.info else False
        self.current_data.update(time=[self.time_counter])

        # expand dimensions for one-dimensional vectors
        self.current_data.update(next_target=[data.next_target], target=[data.target], target_diff=[data.target_diff],
                                 parent_state = [data.parent_state])

        # if np.sum(np.abs(self.current_data.target[:2] - self.current_data.parent_state[:2])) < 8:
        #     print("close", np.concatenate([self.current_data.target,self.current_data.parent_state], axis=-1))
        
        added = ((np.any(data.ext_term)) or # going to resample a new action
            np.any(data.done))
        next_data = None
        if added:
            self.keep_next = True
            # temporal skip is a chance to flush out done values
            if not self.temporal_skip:
                next_data = copy.deepcopy(self.current_data)
                added = True
            else: added = False
            self.time_counter = 0

        # skip the next value if a done or it would get double counted
        self.temporal_skip = np.any(data.next_true_done) and added
        # if np.any(self.current_data.inter) or data.inter or np.linalg.norm(self.current_data.full_state.factored_state.Paddle - self.current_data.full_state.factored_state.Ball) < 6:
        #     print("add inter", added, self.current_data.inter, self.current_data.terminate, data.inter, np.any(data.terminate), data.target, data.next_target)
        self.time_counter += 1
        return next_data, added
