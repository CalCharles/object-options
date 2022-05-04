import numpy as np
import gym
import time
import os
import torch
import warnings
import cv2
import copy
import psutil
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from argparse import Namespace
from collections import deque
from file_management import printframe, saveframe, action_toString
from Networks.network import pytorch_model

from Rollouts.param_buffer import ParamReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer
from Options.option import Option
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from visualizer import visualize

def print_shape(batch, prefix=""):
    print(prefix, {n: batch[n].shape for n in batch.keys() if type(batch[n]) == np.ndarray})

class OptionCollector(Collector): # change to line  (update batch) and line 12 (param parameter), the rest of parameter handling must be in policy
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        exploration_noise: bool = False,
        option: Option = None,
        test: bool = False,
        environment_model = None,
        args: Namespace = None,
    ) -> None:
        self.param_recycle = args.param_recycle # repeat a parameter
        self.option = option
        self.at = 0
        self.env_reset = args.env_reset # if true, then the environment handles resetting
        self.full_at = 0 # the pointer for the buffer without temporal extension
        self.test = test
        self.print_test = args.print_test
        self.full_buffer = copy.deepcopy(buffer) # TODO: make this conditional on usage?
        self.hit_miss_queue = deque(maxlen=2000) # not sure if this is the best way to record hits, but this records when a target position is reached
        self.true_interaction = args.true_interaction
        self.source, self.target = self.option.next_option.name, self.option.name
        self.environment_model = environment_model
        self.save_action = args.save_action # saves the option level action at each time step in option_action.txt in environment.save_dir
        self.save_path = self.environment_model.environment.save_path
        self._keep_proximity = args.keep_proximity
        self.terminate_reset = args.terminate_reset
        self.env_name = args.env
        self.counter = 0
        self.terminate_cutoff = args.terminate_cutoff
        self.no_truncate = args.no_truncate # ignores all timelimit truncated
        option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'w')
        param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'w')
        option_dumps.close()
        param_dumps.close()
        
        # shortcut calling option attributes through option
        self.state_extractor = self.option.state_extractor
        self.sampler = sampler # sampler manages either recalling the param, or getting a new one
        self.exploration_noise = self.option.policy.exploration_noise
        self.temporal_aggregator = TemporalAggregator(sum_reward=args.sum_rewards, only_termination=args.only_termination)
        self.ext_reset = self.option.temporal_extension_manager.reset
        self._aggregate = self.temporal_aggregator.aggregate
        self.policy_collect = self.option.policy.collect
        self._done_check = self.option.done_model.done_check
        
        env = DummyVectorEnv([lambda: env])
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def reset_env(self, keep_statistics: bool = False):
        full_state = self.env.reset()
        self._reset_components(full_state[0])

    def _reset_components(self, full_state):
        # resets internal collector state, option, self.data, temporal aggregator
        self._reset_state(0)
        self.option.reset(full_state)
        param, mask = self._reset_data(full_state)
        self.data.update(obs=[self.state_extractor.get_obs(full_state, full_state, param, mask)]) # self.option.get_state(obs, setting=self.option.input_setting, param=self.param if self.param is not None else None)
        self.temporal_aggregator.reset(self.data)

    def _reset_data(self, full_state):
        # ensure that data has the correct: param, obs, obs_next, full_state, option_resample
        # will always sample a new param
        self.data.update(last_full_state=[full_state], full_state=[full_state])
        param, mask = self.sampler.get_param(full_state, True)
        self.data.update(target=self.state_extractor.get_target(self.data.full_state),
            obs=[self.state_extractor.get_obs(self.data.last_full_state, self.data.full_state, param, mask)],
            obs_next=[self.state_extractor.get_obs(self.data.full_state, self.data.full_state, param, mask)],
            param=[param], mask = [mask], option_resample=[[True]])
        term_chain = self.option.reset(full_state)
        act, chain, policy_batch, state, masks, resampled = self.option.extended_action_sample(self.data, None, term_chain, term_chain[:-1], random=False)
        self.data.update(terminate=[term_chain[-1]], terminations=term_chain, ext_term=[term_chain[-2]], ext_terms=term_chain[:-1])
        self.data.update(done=[False], true_done=[False])
        self.option.update(self.data["full_state"], act, chain, term_chain, masks, not self.test)
        self.temporal_aggregator.update(self.data)
        return param, mask

    def perform_reset(self):
        # artificially create term to sample a new param
        # reset the temporal extension manager
        self.data.update(terminate=[True])
        self.ext_reset()

    def adjust_param(self):
        # either get a new param or recycle the same param as before
        new_param = False
        if np.random.rand() > self.param_recycle: # get a new param
            param, mask = self.sampler.get_param(self.data.full_state[0], self.data.terminate[0])
            self.data.update(param=[param], mask=[mask])
            new_param = True
        else: # use the same param as before
            param, mask = self.data.param.squeeze(), self.data.mask.squeeze()
        self.data.obs = self.state_extractor.assign_param(self.data.full_state, self.data.obs, param, mask)
        self.data.obs_next = self.state_extractor.assign_param(self.data.full_state, self.data.obs_next, param, mask)
        return param, mask, new_param

    def update_statistics(self):
        if self.option.dataset_model.multi_instanced:
            nt = self.option.dataset_model.split_instances(next_target)
            ct = self.option.dataset_model.split_instances(target)
            hit_idx = np.nonzero((nt[...,-1] - ct[...,-1]).flatten())
            inst_hit = nt[hit_idx]
            close = self.option.terminate_reward.epsilon_close
            if self._keep_proximity: close = self.option.policy.learning_algorithm.dist
            hit = ((np.linalg.norm((param-inst_hit), ord=1) <= close and not true_done)
                        or reward_check)
            # print(reward_check, param, next_target, inst_hit, mask)
        else:
            hit = ((np.linalg.norm((param-next_target) * mask) <= self.option.terminate_reward.epsilon_close and not true_done)
                        or reward_check)
        reward_check =  (done and rew > 0)
        hit_count += int(hit)
        miss_count += int(not hit)


    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        n_term: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        visualize_param: str = "",
        no_grad: bool = True,
        force: [np.array, int] = None,
        no_fulls: bool = False,
    ) -> Dict[str, Any]:
        """

        """

        step_count, term_count, episode_count, true_episode_count = 0,0,0,0
        rews = np.float64(0.0)
        term = False
        info = dict()

        while True:
            # set parameter for this run, if necessary
            param, mask, new_param = self.adjust_param()
            if self.test and self.print_test and new_param: print("new param", param)

            # get the action chain
            state_chain = self.data.state_chain if hasattr(self.data, "state_chain") else None
            with torch.no_grad(): act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, self.data.terminations, self.data.ext_terms, random=random, force=force)
            self._policy_state_update(result)
            self.data.update(true_action=[action_chain[0]], act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled])
            
            # step the environment
            # if resampled: print("resampling", act, since_last, action_chain[-1], self.data[0].target[:10], param, self.data[0].obs[:10], pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            since_last = (since_last) * int(not resampled) + 1
            # step in env
            action_remap = self.data.true_action
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)
            next_full_state = obs_next[0] # only handling one environment
            true_done, true_reward = done, rew

            # update the target, next target, obs, next_obs pair, state components
            obs = self.state_extractor.get_obs(self.data.full_state[0], param, mask) # one environment reliance
            obs_next = self.state_extractor.get_obs(next_full_state, param, mask) # one environment reliance
            target = self.state_extractor.get_target(self.data.full_state[0])
            next_target = self.state_extractor.get_target(next_full_state)
            inter_state = self.state_extractor.get_inter(self.data.full_state[0])
            self.data.update(next_target=[next_target], target=[target], obs_next=[obs_next], obs = [obs])

            # get the dones, rewards, terminations and temporal extension terminations
            done, rewards, terminations, ext_terms, inter, time_cutoff = self.option.terminate_reward_chain(self.data.full_state[0], next_full_state, param, action_chain, mask, masks, environment_model=self.environment_model)
            done, rew, term, ext_term = done, rewards[-1], terminations[-1], ext_terms[-1]
            if self.save_action is not None: self.save_action(action_chain[-1], param, resampled, term)
            info[0]["TimeLimit.truncated"] = bool(cutoff + info[0]["TimeLimit.truncated"]) # environment might send truncated itself
            self.option.update(self.buffer, done, self.data.full_state[0], act, action_chain, terminations, param, masks, not self.test)

            # update hit-miss values
            rews += rew
            if term: self.update_statistics()

            # update the current values
            self.data.update(inter_state=[inter_state], next_full_state=[next_full_state], true_done=true_done, true_reward=true_reward, 
                param=[param], mask = [mask], info = info, inter = [inter], time=[1],
                rew=[rew], done=[done], terminate=[term], ext_term = [ext_term], # all prior are stored, after are not 
                terminations= terminations, rewards=rewards, masks=masks, ext_terms=ext_terms)
            # print("post update", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

            # print (self.data) # FDO
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                ))

            # render calls not usually used in our case
            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # we keep a buffer with all of the values
            full_ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(
                self.data, buffer_ids=ready_env_ids)

            # add to the main buffer
            next_data, skipped, added, self.at = self._aggregate(self.data, self.buffer, full_ptr, ready_env_ids)
            if not self.test and self.her_collect is not None: self.her_collect(next_data, self.data, skipped, added)

            # collect statistics
            step_count += len(ready_env_ids)

            # update counters
            term_count += int(np.any(term))
            episode_count += int(np.any(done))
            true_episode_count += int(np.any(true_done))
            if np.any(true_done) or (np.any(term) and self.terminate_reset):
                # if we have a true done, reset the environments and self.data
                if self.env_reset: # the environment might handle resets for us
                    full_state = self.environment_model.get_state()
                    self._reset_components(full_state)
                else:
                    self.reset_env()

            # assign progressive state
            self.data.prev_full_state = self.data.full_state
            self.data.full_state = self.data.next_full_state
            self.data.target = self.data.next_target
            self.data.obs = self.data.obs_next

            # controls breaking from the loop
            if (n_step and step_count >= n_step):
                break
            if (n_episode and episode_count >= n_episode):
                break
            if (n_term and term_count >= n_term):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)
        return { # TODO: some of these don't return valid values
            "n/ep": episode_count,
            "n/tep": true_episode_count,
            "n/tr": term_count,
            "n/st": step_count,
            "n/h": hit_count,
            "n/m": miss_count,
            "rews": rews,
            "terminate": term_end,
            "saved_fulls": saved_fulls,
            "assessment": assessments,
            "info": info,
        }
