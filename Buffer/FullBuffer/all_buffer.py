from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
from Buffer.buffer import ParamPrioWeightedReplayBuffer

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer
class AllReplayBuffer(ReplayBuffer):
    # obs, obs_next contain the flattened full state from the environment
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", 
            "info", "policy", "true_reward", "true_done", 
            "time","option_choice", "option_resample",
            "inter", "obs_diff", "trace", "proximity", "weight_binary")

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
            obs_next=obs_next,
            info=self.get(indice, "info", Batch()),
            policy=self.get(indice, "policy", Batch()),
            true_reward=self.true_reward[indice],
            true_done = self.true_done[indice],
            time = self.time[indice],
            option_choice = self.option_choice[indice],
            option_resample = self.option_resample[indice], # when the option being run is resampled
            inter = self.inter[indice], 
            obs_diff = self.obs_diff[indice], 
            trace = self.trace[indice], 
            proximity = self.proximity[indice], 
            weight_binary = self.weight_binary[indice],
        )

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
