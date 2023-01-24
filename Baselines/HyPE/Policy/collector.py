# HyPE collector
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
from Record.file_management import action_chain_string, display_param, write_string
from Network.network_utils import pytorch_model
from Baselines.HyPE.Policy.aggregator import TemporalAggregator
from Record.file_management import save_to_pickle, create_directory
import imageio as imio


from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer, to_torch_as, to_numpy
from typing import Any, Dict, Tuple, Union, Optional, Callable

class HyPECollector():
    def __init__(
        self,
        env: Union[gym.Env, BaseVectorEnv],
        buffers: Optional[ReplayBuffer] = None,
        skill = None,
        extractor = None,
        test: bool = False,
        record: None = None, # a record object to save states
        policy_iters: int = 0,
        merge_data: bool = False
    ) -> None: 
        self.skill = skill
        self.state_extractor = extractor
        self.num_modes = self.skill.reward_model.num_modes
        self.buffers = buffers
        self.temporal_aggregator = TemporalAggregator(name=skill.name) # we need to record the proper number of transitions
        self.single_buffer = buffers is not None and len(buffers) == 1 # instead of multiple buffers for different rewards, uses a single buffer
        self.record = record
        self.test = test
        self.env = env
        self.policy_iters = policy_iters
        self.counter = 0
        self.merge_data = merge_data
        self.reset_env()

    def reset_env(self, keep_statistics: bool = False):
        full_state = self.env.reset()
        self.data = Batch()
        self._reset_components(full_state, self.policy_iters)

    def _reset_components(self, full_state, policy_iters):
        # resets internal collector state, skill, self.data, temporal aggregator
        ext_terms = self.skill.reset(full_state, policy_iters)
        param = self._reset_data(full_state, ext_terms)
        self.temporal_aggregator.reset(self.data)
        return param

    def sample(self):
        return self.skill.sample_first()

    def sample_check(self, param, num_sample, first=False):
        if first:
            self.last_sample = 0
            self.skill.sample_first(reset_policy=True) # param is set by sample_first
            param = self.skill.sample_next()
            self.skill.set_policy(param)
            param = np.array([param])
            new_param = True
        elif self.last_sample == num_sample:
            param = self.skill.sample_next() # sample next uses a random ordering that must go through all skills
            self.skill.set_policy(param)
            param = np.array([param])
            self.last_sample = 0
            new_param = True
        else:
            new_param = False
        return param, new_param

    def _reset_data(self, full_state, ext_terms):
        # ensure that data has the correct: param, obs, obs_next, full_state, skill_resample
        # will always sample a new param
        self.data.update(last_full_state=[full_state], full_state=[full_state])
        param = self.sample()
        self.data.update(target=self.state_extractor.get_target(self.data.full_state),
            obs=self.state_extractor.get_obs(self.data.full_state),
            obs_next=self.state_extractor.get_obs(self.data.full_state),
            assignment=param, skill_resample=[True], info=[{"Timelimit.truncated": False}])
        term_chain = self.skill.reset(full_state)
        act, chain, policy_batch, state, resampled = self.skill.extended_action_sample(self.data, None, term_chain, term_chain[:-1], random=False)
        self.data.update(ext_term=[True],ext_terms = ext_terms, truncated=[False], state_chain=state)
        self.data.update(done=[False], true_done=[False])
        self.skill.update(act, chain, term_chain, not self.test)
        self.temporal_aggregator.update(self.data)
        return param

    def collect(
        self,
        num_sample,
        num_repeats,
        episodes = 0,
        len_changepoint_queue = 10,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        demonstrate: bool = False,
    ) -> Dict[str, Any]:
        """
        collects the data from the option, or random actions at the top level of the hierarchy
        """
        self.last_rec = self.data.full_state
        step_count, term_count, episode_count, true_episode_count = 0,0,0,0
        last_true_done, rews, true_reward_total, term, info = self.data.true_done, np.float64(0.0), 0.0, False, dict()
        start_time = time.time()
        itr = 0
        her_list = None  # debugging variable
        param = self._reset_components(self.env.get_state(), num_sample)
        first = True # this will ensure that we always start with a new param
        assignments = list()
        # full data queue stores the copies of the state for reward computation, data queue handles temporal extension
        full_data_queue = list()
        data_queue = list()
        aggregates = list()
        changepoint_history_queue = [self.data.full_state[0]]
        print(int(num_sample * self.skill.num_skills * num_repeats), num_sample, self.skill.num_skills, num_repeats)
        i = 0
        timer_end = ((i >= int(self.skill.num_skills * num_repeats)) and episodes <= 0)
        episode_end = ((episodes > 0 and true_episode_count == episodes))
        while not (timer_end or episode_end):
        # for i in range(int(self.skill.num_skills * num_repeats)):
            num_sampled = 0
            while num_sampled < num_sample: # we go for num_sample samples in the TRUE buffer
                tc_start = time.time()
                self.counter += 1
                # set parameter for this run, if necessary
                param, new_param = self.sample_check(param, num_sample, first=first)
                assignments.append(param)
                with torch.no_grad(): act, action_chain, result, state_chain, resampled = self.skill.extended_action_sample(self.data, self.data.state_chain, self.data.done, self.data.ext_terms, random=random)
                tc_action = time.time()
                self.data.update(true_action=[action_chain[0]], act=act, mapped_act=[action_chain[-1]], resample=[resampled], action_chain = action_chain, state_chain = state_chain, network_index = [self.skill.get_network_index()])
                
                # step in env
                action_remap = self.data.true_action[0]
                next_full_state, rew, done, info = self.env.step(action_remap)
                # if self.env.discrete_actions: self.data.full_state.factored_state.Action = [action_remap] # reassign the action to correspond to the current action taken
                self.data.full_state.factored_state.Action = [self.data.true_action]
                true_done, true_reward = done, rew
                tc_step = time.time()
                if self.record is not None:
                    self.record.save(self.data.full_state[0]['factored_state'], self.data.full_state[0]["raw_state"], self.env.toString)

                # update the target, next target, obs, next_obs pair, state components
                obs = self.state_extractor.get_obs(self.data.full_state[0]) # one environment reliance
                obs_next = self.state_extractor.get_obs(next_full_state) # one environment reliance
                target = self.state_extractor.get_target(self.data.full_state[0])
                next_target = self.state_extractor.get_target(next_full_state)
                # print (act, action_remap, action_chain[-1], self.data.full_state[0].factored_state.Gripper, self.data.full_state[0].factored_state.Block, self.env_reset, np.any(true_done) or (np.any(term) and self.terminate_reset))
                parent_state = self.state_extractor.get_parent(self.data.full_state[0])
                target_diff = self.state_extractor.get_diff(self.data.full_state[0], next_full_state)
                self.data.update(next_target=next_target, target=target, target_diff=target_diff, parent_state = parent_state, obs_next=[obs_next], obs = [obs])
                # if "logits" in result: print(self.last_sample, act, action_remap, param, result.logits, obs)
                # if "logits" in result: print(self.last_sample, act, action_remap, param, obs)

                # get the dones, rewards, terminations and temporal extension terminations
                # print(self.data.full_state, self.data.true_done)
                changepoint_history_queue.append(next_full_state)
                ext_term_chain = self.skill.terminate_chain(Batch(changepoint_history_queue), self.data.true_done[0], True)
                self.skill.update(act, action_chain, ext_term_chain, update_policy=True)
                self.data.update(ext_terms = ext_term_chain, ext_term=ext_term_chain[-1], skill_resample=[new_param], done=[(new_param and not first) or true_done], truncated=[new_param])
                first = False

                # update the current values TODO: next_full_state is time expensive (.001 sec per iteration). It should be stored separately
                self.data.update(next_full_state=[next_full_state], true_done=last_true_done, next_true_done=[true_done], true_reward=true_reward, 
                    assignment=param, info = info, time=[1], done=[done])

                if render:
                    self.env.render()
                    if render > 0 and not np.isclose(render, 0):
                        time.sleep(render)

                # Record the state
                full_data_copy = copy.deepcopy(self.data)
                full_data_queue.append(full_data_copy)
                if len(changepoint_history_queue) > len_changepoint_queue: changepoint_history_queue.pop(0)
                data, added = self.temporal_aggregator.aggregate(self.data) # if not added data will be None
                num_sampled += int(added)
                self.last_sample += int(added)
                # print("aggregating", len(data_queue), added, self.data.done, self.data.target)
                if added: data_queue.append(data)
                aggregates.append(added)
                # collect statistics
                step_count += 1
                # print(true_done, self.data.true_done, last_true_done)

                # update counters
                self.data.prev_full_state = self.data.full_state
                self.data.full_state = self.data.next_full_state
                self.data.target = self.data.next_target
                self.data.obs = self.data.obs_next
                true_reward_total += true_reward
                if true_reward != 0: print("adding true reward", true_reward, int(np.any(self.data.true_done)))
                last_true_done = [true_done]
                episode_count += int(true_done)
                true_episode_count += int(true_done)
                itr += 1
            i += 1
            timer_end = ((i >= int(self.skill.num_skills * num_repeats)) and episodes <= 0)
            episode_end = ((episodes > 0 and true_episode_count >= episodes))
            print("end", timer_end, episode_end, itr, i,int(self.skill.num_skills * num_repeats), episodes, true_episode_count, true_reward_total)

        data_index = 0
        assignment_dicts = dict()
        for asmt, agg, full_data in zip(assignments, aggregates, full_data_queue):
            asmt = asmt[0]
            assign_val = asmt
            if self.skill.reward_model.one_mode or self.merge_data: asmt = 0
            if asmt in assignment_dicts:
                assignment_dicts[asmt]["assignment"] = assign_val
                assignment_dicts[asmt]['full_data'].append(full_data)
                assignment_dicts[asmt]['target'].append(full_data.target)
                assignment_dicts[asmt]['parent_state'].append(full_data.parent_state)
                assignment_dicts[asmt]['target_diff'].append(full_data.target_diff)
                assignment_dicts[asmt]['done'].append(full_data.done)
                if agg:
                    assignment_dicts[asmt]["data"].append(data_queue[data_index])
                    assignment_dicts[asmt]["data_counts"].append(assignment_dicts[asmt]["ctr"])
                    data_index += 1 
                assignment_dicts[asmt]["ctr"] += 1
            else:
                assignment_dicts[asmt] = {'assignment': [assign_val], 'full_data': [full_data], 'target': [full_data.target], 
                                        'parent_state': [full_data.parent_state], 'target_diff': [full_data.target_diff],
                                        'done': [full_data.done], "ctr": 0}
                if agg:
                    assignment_dicts[asmt]["data"] = [data_queue[data_index]]
                    assignment_dicts[asmt]["data_counts"] = [assignment_dicts[asmt]["ctr"]]
                    data_index += 1
                    assignment_dicts[asmt]["ctr"] += 1
                else:
                    assignment_dicts[asmt]["data"] = list()
                    assignment_dicts[asmt]["data_counts"] = list()
        hit, miss, term_count = 0,0, 0
        assignment_reward = dict()
        # print(assignment_dicts.keys())
        # print(np.concatenate(assignments).squeeze().tolist())
        asmt_keys = assignment_dicts.keys() if not self.merge_data else set(np.concatenate(assignments).squeeze().tolist())
        for asmt in asmt_keys:
            if self.single_buffer: asmt = 0
            asmt_dict = assignment_dicts[asmt] if not self.merge_data else assignment_dicts[0]
            # print(np.stack(asmt_dict['target_diff'], axis=0), np.stack(asmt_dict['target'], axis=0),
            #                             np.stack(asmt_dict['parent_state'], axis=0), np.stack(asmt_dict['done'], axis=0))
            # print("rewarding", asmt, np.stack(asmt_dict['target_diff'], axis=0).shape, np.stack(asmt_dict['target'], axis=0).shape,
            #                             np.stack(asmt_dict['parent_state'], axis=0).shape, np.stack(asmt_dict['done'], axis=0).shape)
            rewards, terminations = self.skill.reward_model.compute_reward(np.stack(asmt_dict['target_diff'], axis=0), np.stack(asmt_dict['target'], axis=0),
                                        np.stack(asmt_dict['parent_state'], axis=0), np.stack(asmt_dict['done'], axis=0))
            count_at = 0
            rewards = np.stack(rewards, axis=-1)
            assignment_reward[asmt] = 0
            # print(asmt, rewards, terminations)
            for ctr, (full_reward, term) in enumerate(zip(rewards, terminations)):
                # print(ctr, asmt_dict["data_counts"][count_at])
                if count_at < len(asmt_dict["data_counts"]) and ctr == asmt_dict["data_counts"][count_at]:
                    asmt_dict["data"][count_at].rew = full_reward[asmt]
                    rews += full_reward[asmt]
                    if np.any(term) or np.any(asmt_dict["data"][count_at].done):
                        hit += int(int(full_reward[asmt]) == self.skill.reward_model.param_reward) # kind of hacky way to check
                        term_count += 1
                        miss += int(int(full_reward[asmt]) != self.skill.reward_model.param_reward)
                    asmt_dict["data"][count_at].full_reward = full_reward
                    asmt_dict["data"][count_at].terminate = term
                    asmt_dict["data"][count_at].done = asmt_dict["data"][count_at].done or term
                    if np.sum(full_reward) > 0: print("assigning", asmt_dict["data"][count_at].done, term, full_reward, asmt_dict["data"][count_at].rew, asmt)
                    assignment_reward[asmt] += full_reward[asmt]
                    # print("reward", asmt_dict["data"][count_at].rew)
                    if self.buffers is not None: self.buffers[asmt].add(asmt_dict["data"][count_at])
                    count_at += 1
        # generate statistics
        # self.collect_step += int(num_sample * self.skill.num_skills * num_repeats)
        # self.collect_episode += episode_count
        # self.collect_time += max(time.time() - start_time, 1e-9)
        print("assignment reward", assignment_reward)
        return { # TODO: some of these don't return valid values
            "n/ep": episode_count,
            "n/tep": true_episode_count,
            "n/tr": term_count,
            "n/st": step_count,
            "n/h": hit,
            "n/dr": true_reward_total,
            "n/m": miss,
            "n/tim": max(time.time() - start_time, 1e-9), 
            "rews": rews,
            "info": info,
        }
