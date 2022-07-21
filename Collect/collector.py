import numpy as np
import gym
import time
import os
import torch
import warnings
import cv2
import copy
from typing import Any, Dict, List, Union, Optional, Callable
from argparse import Namespace
from collections import deque
from Record.file_management import action_chain_string
from Network.network_utils import pytorch_model
from Collect.aggregator import TemporalAggregator
from Record.file_management import save_to_pickle, create_directory
from Causal.Utils.instance_handling import split_instances

from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

def print_shape(batch, prefix=""):
    print(prefix, {n: batch[n].shape for n in batch.keys() if type(batch[n]) == np.ndarray})

class OptionCollector(Collector): # change to line  (update batch) and line 12 (param parameter), the rest of parameter handling must be in policy
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        exploration_noise: bool = False,
        option = None,
        hindsight = None,
        test: bool = False,
        multi_instanced: bool = False,
        record: None = None, # a record object to save states
        args: Namespace = None,
    ) -> None:
        self.param_recycle = args.sample.param_recycle # repeat a parameter
        self.param_counter = 0
        self.option = option
        self.at = 0
        self.full_at = 0 # the pointer for the buffer without temporal extension
        self.her_at = 0
        self.env_reset = args.collect.env_reset # if true, then the environment handles resetting
        self.test = test
        self.full_buffer = copy.deepcopy(buffer) # TODO: make this conditional on usage?
        self.her_buffer = copy.deepcopy(buffer)
        self.hit_miss_queue = deque(maxlen=2000) # not sure if this is the best way to record hits, but this records when a target position is reached
        self.save_action = args.record.save_action # saves the option level action at each time step in option_action.txt in environment.save_dir
        self.save_path = args.record.record_rollouts 
        self.record = record
        self.terminate_reset = args.collect.terminate_reset
        self.env_name = env.name
        self.counter = 0
        self.multi_instanced = multi_instanced
        self.names = args.object_names
        if self.save_action:
            option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'w')
            param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'w')
            term_dumps = open(os.path.join(self.save_path, "term_dumps.txt"), 'w')
            option_dumps.close()
            param_dumps.close()
            term_dumps.close()
        
        # shortcut calling option attributes through option
        self.state_extractor = self.option.state_extractor
        self.sampler = self.option.sampler # sampler manages either recalling the param, or getting a new one
        self.exploration_noise = exploration_noise
        self.temporal_aggregator = TemporalAggregator(sum_reward=args.collect.aggregator.sum_rewards, only_termination=args.collect.aggregator.only_termination)
        self.ext_reset = self.option.temporal_extension_manager.reset
        self._aggregate = self.temporal_aggregator.aggregate
        self.hindsight = hindsight
        if hindsight is not None:
            self.her_collect = hindsight.record_state
        
        self.environment = env
        env = DummyVectorEnv([lambda: env])
        super().__init__(policy, env, buffer, None, exploration_noise)

    def save(self, save_path, filename):
        save_to_pickle(os.path.join(create_directory(save_path), filename), self.get_components())

    def load(self, load_path):
        self.restore_components(*load_from_pickle(load_path))

    def restore_components(self, full_buffer, her_buffer, buffer, at, full_at, her_at):
        self.full_buffer = full_buffer
        self.her_buffer = her_buffer
        self.buffer = buffer
        self.at = at
        self.full_at = full_at
        self.her_at = her_at

    def get_components(self):
        return self.full_buffer, self.her_buffer, self.buffer, self.at, self.full_at, self.her_at

    def _save_action(self, action_chain, term_chain, param): # this is handled here because action chains are option dependent
        write_string(os.path.join(self.save_path, "option_dumps.txt"), str(self.environment_model.environment.get_itr() - 1) + ":" + action_chain_string(action_chain) + "\n")
        write_string(os.path.join(self.save_path, "term_dumps.txt"), str(self.environment_model.environment.get_itr() - 1) + ":" + action_chain_string([term_chain]) + "\n")
        write_string(os.path.join(self.save_path, "param_dumps.txt"), str(self.environment_model.environment.get_itr() - 1) + ":" + action_chain_string([param[0]]) + "\n")

    def reset_env(self, keep_statistics: bool = False):
        full_state = self.env.reset()
        self._reset_components(full_state[0])

    def _reset_components(self, full_state):
        # resets internal collector state, option, self.data, temporal aggregator
        self._reset_state(0)
        self.option.reset(full_state)
        param, mask = self._reset_data(full_state)
        if self.hindsight is not None: self.hindsight.reset(self.data)
        self.temporal_aggregator.reset(self.data)

    def sample(self, full_state):
        param, mask = self.sampler.sample(full_state)
        param, mask = np.array([param]), np.array([mask])
        return param, mask

    def _reset_data(self, full_state):
        # ensure that data has the correct: param, obs, obs_next, full_state, option_resample
        # will always sample a new param
        self.data.update(last_full_state=[full_state], full_state=[full_state])
        param, mask = self.sample(full_state)
        self.param_counter = 0
        self.data.update(target=self.state_extractor.get_target(self.data.full_state),
            obs=self.state_extractor.get_obs(self.data.last_full_state, self.data.full_state, param, mask),
            obs_next=self.state_extractor.get_obs(self.data.full_state, self.data.full_state, param, mask),
            param=param, mask = mask, option_resample=[[True]])
        term_chain = self.option.reset(full_state)
        act, chain, policy_batch, state, masks, resampled = self.option.extended_action_sample(self.data, None, term_chain, term_chain[:-1], random=False)
        self.data.update(terminate=[term_chain[-1]], terminations=term_chain, ext_term=[term_chain[-2]], ext_terms=term_chain[:-1])
        self.data.update(done=[False], true_done=[False])
        self.option.update(act, chain, term_chain, masks, not self.test)
        self.temporal_aggregator.update(self.data)
        return param, mask

    def perform_reset(self):
        # artificially create term to sample a new param
        # reset the temporal extension manager
        self.data.update(terminate=[True])
        self.ext_reset()

    def adjust_param(self, force=False):
        # either get a new param or recycle the same param as before
        new_param = False
        param, mask = self.data.param, self.data.mask
        if np.any(self.data.terminate) or np.any(self.data.done) or force:
            new_param = True
            self.param_counter += 1
            if self.param_recycle > 1 and self.param_counter == self.param_recycle:
                param, mask = self.sample(self.data.full_state[0])
                self.data.update(param=param, mask=mask)
                self.param_counter = 0
            if np.random.rand() > self.param_recycle: # get a new param
                param, mask = self.sample(self.data.full_state[0])
                self.data.update(param=param, mask=mask)

            # Otherwise, already uses the same param as before
            # assign the param in the observation
            if self.data.obs is not None: # otherwise these have not been initialized yet and will be later
                self.data.obs = self.state_extractor.assign_param(self.data.full_state, self.data.obs, param, mask)
                self.data.obs_next = self.state_extractor.assign_param(self.data.full_state, self.data.obs_next, param, mask)
        return param, mask, new_param

    def update_statistics(self, hit_count, miss_count):
        reward_check =  (self.data.done[0] and self.data.rew[0] > 0)
        if self.multi_instanced:
            nt = split_instances(self.data.next_target, self.option.interaction_model.obj_dim)
            ct = split_instances(self.data.target, self.option.interaction_model.obj_dim)
            hit_idx = np.nonzero((nt[...,-1] - ct[...,-1]).flatten())
            inst_hit = nt[hit_idx]
            close = self.option.terminate_reward.epsilon_close
            hit = ((np.linalg.norm((self.data.param[0]-inst_hit), ord=1) <= close and not np.any(self.data.true_done))
                        or reward_check)
        else:
            hit = ((np.linalg.norm((self.data.param-self.data.next_target) * self.data.mask) <= self.option.terminate_reward.epsilon_close and not self.data.true_done[0])
                        or reward_check)
        hit_count += int(hit)
        miss_count += int(not hit)
        return hit, hit_count, miss_count

    def _policy_state_update(self, result):
        # update state / act / policy into self.data
        policy = result.get("policy", Batch())
        state = result.get("state", None)
        if state is not None:
            policy.hidden_state = state  # save state into buffer
            self.data.update(state_chain=state_chain, policy=policy)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        n_term: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        force: [np.array, int] = None,
        new_param: bool = False
    ) -> Dict[str, Any]:
        """
        collects the data from the option, or random actions at the top level of the hierarchy
        """
        step_count, term_count, episode_count, true_episode_count = 0,0,0,0
        ready_env_ids = np.arange(1) # env_num is used in ts for multi-threaded environments, which are not used 
        last_true_done, rews, term, info = self.data.true_done, np.float64(0.0), False, dict()
        hit_count, miss_count = 0,0
        start_time = time.time()
        param, mask, new_param = self.adjust_param(new_param)
        while True:
            # set parameter for this run, if necessary
            param, mask, new_param = self.adjust_param()
            # if self.test and new_param: print("new param", param)
            if new_param: print("new param", param)

            # get the action chain
            state_chain = self.data.state_chain if hasattr(self.data, "state_chain") else None
            with torch.no_grad(): act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, self.data.terminations, self.data.ext_terms, random=random, force=force)
            self._policy_state_update(result)
            self.data.update(true_action=[action_chain[0]], act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled], action_chain = action_chain)
            
            # step in env
            action_remap = self.data.true_action
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)
            # print(self.data.full_state.factored_state.Action)
            if self.environment.discrete_actions: self.data.full_state.factored_state.Action = [action_remap] # reassign the action to correspond to the current action taken
            else: self.data.full_state.factored_state.Action = action_remap
            # print(action_remap, self.data.full_state.factored_state.Action)
            next_full_state = obs_next[0] # only handling one environment
            true_done, true_reward = done, rew

            # update the target, next target, obs, next_obs pair, state components
            obs = self.state_extractor.get_obs(self.data.last_full_state[0], self.data.full_state[0], param[0], mask[0]) # one environment reliance
            obs_next = self.state_extractor.get_obs(self.data.full_state[0], next_full_state, param[0], mask[0]) # one environment reliance
            target = self.state_extractor.get_target(self.data.full_state[0])
            next_target = self.state_extractor.get_target(next_full_state)
            inter_state = self.state_extractor.get_inter(self.data.full_state[0])
            parent_state = self.state_extractor.get_parent(self.data.full_state[0])
            target_diff = self.state_extractor.get_diff(self.data.full_state[0], next_full_state)
            additional_state = self.state_extractor.get_additional(self.data.full_state[0])
            self.data.update(next_target=next_target, target=target, target_diff=target_diff, parent_state = parent_state, inter_state=inter_state, additional_state=additional_state, obs_next=[obs_next], obs = [obs])

            # get the dones, rewards, terminations and temporal extension terminations
            done, rewards, terminations, inter, cutoff = self.option.terminate_reward_chain(self.data.full_state[0], next_full_state, param, action_chain, mask, masks, true_done, true_reward)
            done, rew, term, ext_term = done, rewards[-1], terminations[-1], terminations[-2] or resampled
            if self.save_action: self._save_action(action_chain, param, resampled, term)
            info[0]["TimeLimit.truncated"] = bool(cutoff + info[0]["TimeLimit.truncated"]) if "TimeLimit.truncated" in info[0] else cutoff # environment might send truncated itself
            self.option.update(act, action_chain, terminations, masks, update_policy=not self.test)
            # print(inter_state, self.option.interaction_model.predict_next_state(self.data.full_state))

            # update inline training values
            proximity, binaries = self.option.inline_trainer.set_values(self.data)

            # update hit-miss values
            rews += rew
            hit = False
            if term: hit, hit_count, miss_count = self.update_statistics(hit_count, miss_count)

            # update the current values
            self.data.update(next_full_state=[next_full_state], true_done=last_true_done, true_reward=true_reward, 
                param=param, mask = mask, info = info, inter = [inter], time=[1], trace = [np.any(self.environment.current_trace(self.names))], inst_trace=self.environment.current_trace(self.names), proximity=[proximity.squeeze()], weight_binary=[binaries.squeeze()],
                rew=[rew], done=[done], terminate=[term], ext_term = [ext_term], # all prior are stored, after are not 
                terminations= terminations, rewards=rewards, masks=masks, ext_terms=terminations[:len(terminations) - 1])

            # if self.test: print(hit, next_target, self.state_extractor.reverse_obs_norm(obs, mask[0]),
            # pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)), action_chain, act)
            if self.test: print(param, next_target, act)
            # print(ext_term, resampled, self.state_extractor.reverse_obs_norm(obs, mask[0])[6], self.state_extractor.reverse_obs_norm(obs_next, mask[0])[6],
            # pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)), action_chain)
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

            # Record the state
            if self.record is not None: self.record.save(self.data[0].full_state['factored_state'], self.data[0].full_state["raw_state"], self.environment.toString)

            # we keep a buffer with all of the values
            self.data.done = np.array([self.data.done[0].astype(float)])
            full_ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(self.data)
            # print("adding data", self.data.done, self.data.terminate, self.full_buffer.done.shape, self.full_buffer.terminate.shape)

            # add to the main buffer
            next_data, skipped, added, self.at = self._aggregate(self.data, self.buffer, full_ptr, ready_env_ids)
            # print(self.data.target, not self.test, self.hindsight is not None)
            if not self.test and self.hindsight is not None: self.her_at = self.her_collect(self.her_buffer, next_data, self.data, added)

            # collect statistics
            step_count += len(ready_env_ids)

            # update counters
            term_count += int(np.any(term))
            episode_count += int(np.any(done))
            true_episode_count += int(np.any(true_done))
            if np.any(true_done) or (np.any(term) and self.terminate_reset):
                done, true_done = copy.deepcopy(self.data.done), copy.deepcopy(self.data.true_done) # preserve these values for use in new param getting
                # if we have a true done, reset the environments and self.data
                if self.env_reset: # the environment might handle resets for us
                    full_state = self.environment.get_state()
                    self._reset_components(full_state)
                else:
                    self.reset_env()
                self.data.update(done=done, true_done = true_done)

            # assign progressive state
            self.data.prev_full_state = self.data.full_state
            self.data.full_state = self.data.next_full_state
            self.data.target = self.data.next_target
            self.data.obs = self.data.obs_next
            last_true_done = true_done

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
            "n/tim": self.collect_time, 
            "rews": rews,
            "terminate": (not np.any(true_done)) and np.any(term) and self.terminate_reset,
            "info": info,
        }
