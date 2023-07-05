from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import copy

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
    # TODO: parent is the action that specifies which parent to use
    _all_keys = ["obs", "act", "rew", "done", "obs_next", "info", "policy", "param", "terminated", "truncated",
        "mask", "target", "next_target", "target_diff", "terminate", "true_reward", "true_done", "option_resample", 
        "mapped_act", "inter", "trace", "inst_trace", "proximity", "proximity_inst", "inter_state", "parent_state", "additional_state", "time", "weight_binary"]
    _reserved_keys = tuple(_all_keys)
    _input_keys = copy.copy(_all_keys)
    _input_keys.pop(_input_keys.index("done"))
    _input_keys = tuple(_input_keys)

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
            terminated=self.terminated[indice],
            truncated=self.truncated[indice],
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
            inst_trace = self.inst_trace[indice],
            proximity = self.proximity[indice],
            proximity_inst = self.proximity_inst[indice],
            inter_state = self.inter_state[indice],
            parent_state = self.parent_state[indice],
            additional_state = self.additional_state[indice],
            weight_binary = self.weight_binary[indice],
            time = self.time[indice]
        )

class ParamPriorityReplayBuffer(PrioritizedReplayBuffer): # not using double inheritance so exactly the same as above.
    _all_keys = ["obs", "act", "rew", "done", "obs_next", "info", "policy", "param", "terminated",  "truncated",
        "mask", "target", "next_target", "target_diff", "terminate", "true_reward", "true_done", "option_resample", 
        "mapped_act", "inter", "trace", "inst_trace", "proximity", "proximity_inst", "inter_state", "parent_state", "additional_state", "time", "weight_binary"]
    _reserved_keys = tuple(_all_keys)
    _input_keys = copy.copy(_all_keys)
    _input_keys.pop(_input_keys.index("done"))
    _input_keys = tuple(_input_keys)

    def sample(self, batch_size: int, no_prio=False) -> Tuple[Batch, np.ndarray]:
        """Replace Tianshou Sample to add no-prio parameter
        """
        indices = self.sample_indices(batch_size, no_prio=no_prio)
        return self[indices], indices


    def sample_indices(self, batch_size: int, no_prio=False) -> np.ndarray:
        # added no-prio logic
        if batch_size > 0 and len(self) > 0 and not no_prio:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)  # type: ignore
        else:
            return super().sample_indices(batch_size)


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
            terminated=self.terminated[indice],
            truncated=self.truncated[indice],
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
            inst_trace = self.inst_trace[indice],
            proximity = self.proximity[indice],
            proximity_inst = self.proximity_inst[indice],
            inter_state = self.inter_state[indice],
            parent_state = self.parent_state[indice],
            additional_state = self.additional_state[indice],
            weight_binary = self.weight_binary[indice],
            time = self.time[indice]
        )

class ParamWeightedReplayBuffer(ParamReplayBuffer):
    def sample_indices(self, batch_size: int, weights: np.ndarray=None) -> np.ndarray:
        """Get a random sample of index with size = batch_size, with @param weights weighting the selection
        see Tianshou.data.bbase, since most of the code is copied from there

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if weights is not None: # if given weights, use those
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
        else:
            return super().sample_indices(batch_size) # sample using priority


    def sample(self, batch_size: int, weights: np.ndarray=None) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size, weights)
        return self[indices], indices

class ParamPrioWeightedReplayBuffer(ParamPriorityReplayBuffer): # the same as above except for this line
    def sample_indices(self, batch_size: int, weights: np.ndarray=None) -> np.ndarray:
        """Get a random sample of index with size = batch_size, with @param weights weighting the selection
        see Tianshou.data.bbase, since most of the code is copied from there

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if weights is not None: # if given weights, use those
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
        else:
            return super().sample_indices(batch_size) # sample using priority


    def sample(self, batch_size: int, weights: np.ndarray=None) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size, weights)
        return self[indices], indices

class InterWeightedReplayBuffer(ReplayBuffer):
    # A buffer used for the active-passive models.
    # obs, act, rew, done are required by tianshou, but are not very useful for the models
    # mask is the active mask, which is most useful when using complete states (for end to end)
    # target, next target, target_diff is the state of the target object, used for reward and termination
    # inter, trace record the interaction values in various forms
    # inter_state, parent_state, additional_state record:
    #   inter = parent+additional+target, parent=primary parent, additional=other state
    # TODO: parent is the action that specifies which parent to use
    _all_keys = ["obs", "act", "rew", "done", "true_done", "obs_next", "terminated", "truncated",
        "mask", "target", "next_target", "target_diff",
        "inter", "trace", "inst_trace", "proximity", "proximity_inst", "inter_state", "parent_state", "additional_state", "weight_binary"]
    _reserved_keys = tuple(_all_keys)
    
    _input_keys = copy.copy(_all_keys)
    _input_keys.pop(_input_keys.index("done"))
    _input_keys = tuple(_input_keys)

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
            terminated=self.terminated[indice],
            truncated=self.truncated[indice],
            obs_next=obs_next,
            true_done=self.true_done[indice],
            mask = self.mask[indice], 
            target = self.target[indice], 
            next_target=self.next_target[indice], 
            target_diff=self.target_diff[indice], 
            inter = self.inter[indice],
            trace = self.trace[indice],
            inst_trace = self.inst_trace[indice],
            proximity = self.proximity[indice],
            proximity_inst = self.proximity_inst[indice],
            inter_state = self.inter_state[indice],
            parent_state = self.parent_state[indice],
            additional_state = self.additional_state[indice],
            weight_binary = self.weight_binary[indice],
        )

    def sample_indices(self, batch_size: int, weights: np.ndarray = None) -> np.ndarray:
        # this branch of the conditional is for the priority weights
        if batch_size > 0 and len(self) > 0 and weights is None:
            return np.random.choice(self._size, batch_size)
        else: # below is copied from weighted Replay Buffer, samples with the given weights
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


    def sample(self, batch_size: int, weights: np.ndarray = None) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size, weights)
        return self[indices], indices