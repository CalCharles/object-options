import numpy as np
import copy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

class TemporalAggregator():
    def __init__(self, sum_reward=False, only_termination=False,name=""):
        self.current_data = Batch()
        self.name= name
        self.next_action = False
        self.next_param = False
        self.keep_next = True
        self.temporal_skip = False
        self.ptr = 0
        self.sum_reward = sum_reward # sums reward for the length of the trajectory
        self.time_counter = 0 # counts the number of time steps in the temporal extension
        self.only_termination = only_termination # only samples when there is a termination of the current option

    def reset(self, data):
        self.current_data = copy.deepcopy(data)
        self.keep_next = True
        self.time_counter = 0

    def update(self, data):
        self.current_data = copy.deepcopy(data)

    def aggregate(self, data, buffer, ptr, ready_env_ids):
        # updates "next" values to the current value, and combines dones, rewards
        added = False
        skipped = False
        if self.keep_next: 
            self.current_data = copy.deepcopy(data)
            self.keep_next = False
        else: # otherwise, we only update the reward, this is to ensure the reward is NOT updated twice
            if self.sum_reward:
                self.current_data.rew += data.rew.squeeze()
                self.current_data.true_reward += data.true_reward.squeeze()
            else: # keep the last reward
                self.current_data.rew = [data.rew.squeeze().astype(float)]
                self.current_data.true_reward = [data.true_reward.squeeze().astype(float)]
        # update state components
        self.current_data.update(next_full_state = data.next_full_state, next_target=data.next_target, obs_next=data.obs_next, inter_state=data.inter_state)
        # update  termination and resampling components
        self.current_data.done = np.any(self.current_data.done) + np.any(data.done)
        self.current_data.terminate = [np.any(self.current_data.terminate) + np.any(data.terminate)] # basically an OR
        self.current_data.true_done = [np.any(self.current_data.true_done) + np.any(data.true_done)] # basically an OR
        self.current_data.trace = [np.any(self.current_data.trace) + np.any(data.trace)]
        self.current_data.inst_trace = [(self.current_data.inst_trace[0] + data.inst_trace).astype(bool)]
        self.current_data.option_resample = data.option_resample
        self.current_data.info["TimeLimit.truncated"] = data.info["TimeLimit.truncated"] if "TimeLimit.truncated" in data.info else False
        self.current_data.truncated = self.current_data.info["TimeLimit.truncated"]
        self.current_data.terminated = [self.current_data.done]
        self.current_data.inter = [max(data.inter[0], self.current_data.inter[0])]
        self.current_data.update(time=[self.time_counter])

        # expand dimensions for one-dimensional vectors
        self.current_data.update(inter_state=[data.inter_state], next_target=[data.next_target], target=[data.target], target_diff=[data.target_diff], parent_state = [data.parent_state], additional_state=[data.additional_state])
        
        # if a true done happened, the NEXT data point will need to be recorded
        # if we just resampled (meaning temporal extension occurred), or a done or termination
        if self.name == "Ball" and (np.linalg.norm(self.current_data.full_state.factored_state.Paddle - self.current_data.full_state.factored_state.Ball) < 8 or np.any(self.current_data.trace) or np.any(self.current_data.inter)):
        # if np.any(self.current_data.trace) or np.any(self.current_data.inter):
            print(self.current_data.target, self.current_data.next_target, self.current_data.param, self.current_data.trace, self.current_data.inter, self.current_data.rew, self.current_data.action_chain)
        # print((np.any(data.ext_term) and not self.only_termination), # going to resample a new action
        #     np.any(data.done),
        #     np.any(data.terminate), data.true_done, self.temporal_skip, self.current_data.inter, self.current_data.parent_state, self.current_data.obs, self.current_data.obs_next)
        # print("seen", np.any(data.ext_term), self.current_data.time, self.current_data.target, self.current_data.next_target, self.current_data.obs, self.current_data.obs_next)

        added = ((np.any(data.ext_term) and not self.only_termination) or # going to resample a new action
            np.any(data.done)
            or np.any(data.terminate))
        if added:
            next_data = copy.deepcopy(self.current_data)
            self.keep_next = True
            # temporal skip is a chance to flush out done values
            if not self.temporal_skip:
                added = True
                # print(next_data.time, next_data.target, next_data.next_target, next_data.obs, next_data.obs_next)
                # print(next_data.terminated, next_data.truncated, next_data.info, next_data.terminate, next_data.done, next_data.true_done)
                self.ptr, ep_rew, ep_len, ep_idx = buffer.add(
                        next_data, buffer_ids=ready_env_ids)
            else: added = False
            self.time_counter = 0

        # skip the next value if a done or it would get double counted
        self.temporal_skip = np.any(data.next_true_done) and added
        # if np.any(self.current_data.inter) or data.inter or np.linalg.norm(self.current_data.full_state.factored_state.Paddle - self.current_data.full_state.factored_state.Ball) < 6:
        #     print("add inter", added, self.current_data.inter, self.current_data.terminate, data.inter, np.any(data.terminate), data.target, data.next_target)
        self.time_counter += 1
        return self.current_data, skipped, added, self.ptr
