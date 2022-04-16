from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer

class ParamReplayBuffer(ReplayBuffer):
    # obs, obs_next  is the observation form as used by the highest level option (including param, and relative state, if used)
    # act is the action of the highest level option
    # rew is the reward of the highest level option
    # done is the termination of the highest level option
    # param is the param used at the time of input
    # target, next target is the state of the target object, used for reward and termination
    # true_reward, true_done are the actual dones and rewards
    # option_terminate is for temporal extension, stating if the last object terminated
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy", "param", 
        "mask", "target", "next_target", "target_diff", "terminate", "true_reward", "true_done", "option_resample", 
        "mapped_act", "inter", "trace", "inter_state", "parent_state", "additional_state", "time")

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indice = self.sample_index(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indice = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indice, "obs")
        if self._save_obs_next:
            obs_next = self.get(indice, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indice), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indice],
            rew=self.rew[indice],
            done=self.done[indice],
            terminate = self.terminate[indice],
            obs_next=obs_next,
            info=self.get(indice, "info", Batch()),
            policy=self.get(indice, "policy", Batch()),
            param = self.param[indice], # all the below lines differ from replay buffer to handle additional option values
            mask = self.mask[indice], 
            target = self.target[indice], 
            next_target=self.next_target[indice], 
            target_diff=self.target_diff[indice], 
            true_reward=self.true_reward[indice],
            true_done = self.true_done[indice],
            option_resample = self.option_resample[indice],
            mapped_act = self.mapped_act[indice],
            inter = self.inter[indice],
            trace = self.trace[indice],
            inter_state = self.inter_state[indice],
            parent_state = self.parent_state[indice],
            additional_state = self.additional_state[indice],
            time = self.time[indice]
        )

class ParamWeightedReplayBuffer(ParamReplayBuffer):
    def sample_indices(self, batch_size: int, weights: np.ndarray) -> np.ndarray:
        """Get a random sample of index with size = batch_size, with @param weights weighting the selection
        see Tianshou.data.bbase, since most of the code is copied from there

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size, p=weights)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._index, self._size),
                     np.arange(self._index)]
                )
            else:
                return np.array([], int)
        else:
            if batch_size < 0:
                return np.array([], int)
            all_indices = prev_indices = np.concatenate(
                [np.arange(self._index, self._size),
                 np.arange(self._index)]
            )
            for _ in range(self.stack_num - 2):
                prev_indices = self.prev(prev_indices)
            all_indices = all_indices[prev_indices != self.prev(prev_indices)]
            if batch_size > 0:
                return np.random.choice(all_indices, batch_size, p=weights)
            else:
                return all_indices

    def sample(self, batch_size: int, weights: np.ndarray) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size, weights)
        return self[indices], indices



class ParamPriorityReplayBuffer(PrioritizedReplayBuffer): # not using double inheritance so exactly the same as above.
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy", "param", 
        "mask", "target", "next_target", "target_diff", "terminate", "true_reward", "true_done", "option_resample", 
        "mapped_act", "inter", "inter_state", "parent_state", "additional_state", "time")

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indice = self.sample_index(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indice = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indice, "obs")
        if self._save_obs_next:
            obs_next = self.get(indice, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indice), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indice],
            rew=self.rew[indice],
            done=self.done[indice],
            terminate = self.terminate[indice],
            obs_next=obs_next,
            info=self.get(indice, "info", Batch()),
            policy=self.get(indice, "policy", Batch()),
            param = self.param[indice], # all below lines differ from prioritized replay buffer to handle option values
            mask = self.mask[indice],
            target = self.target[indice],
            next_target=self.next_target[indice],
            target_diff=self.target_diff[indice], 
            true_reward=self.true_reward[indice],
            true_done = self.true_done[indice],
            option_resample = self.option_resample[indice],
            mapped_act = self.mapped_act[indice],
            inter = self.inter[indice],
            trace = self.trace[indice],
            inter_state = self.inter_state[indice],
            parent_state = self.parent_state[indice],
            additional_state = self.additional_state[indice],
            time = self.time[indice]
        )