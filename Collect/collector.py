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
from Collect.aggregator import TemporalAggregator
from Record.file_management import save_to_pickle, create_directory
from Option.General.param import check_close
from Causal.Utils.instance_handling import split_instances
import imageio as imio


from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer, to_torch_as, to_numpy
from typing import Any, Dict, Tuple, Union, Optional, Callable

FULL_ENVS = ["Breakout", "Asteroids", "RoboPushing", "AirHockey"]

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
        stream_write: bool = True, # writes by append
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
        self.trunc_true = args.option.trunc_true
        self.env_name = env.name
        self.counter = 0
        self.multi_instanced = multi_instanced
        self.names = args.object_names
        self.display_frame = args.collect.display_frame
        self.stream_print_file = args.collect.stream_print_file
        self.save_display = args.collect.save_display
        self.time_check = args.collect.time_check
        self.stream_write = stream_write
        if len(self.stream_print_file) > 0: 
            create_directory(os.path.split(self.stream_print_file)[0])
            self.stream_str_record = deque(maxlen=1000)
        if self.save_action:
            option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'w')
            param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'w')
            term_dumps = open(os.path.join(self.save_path, "term_dumps.txt"), 'w')
            mask_dumps = open(os.path.join(self.save_path, "mask_dumps.txt"), 'w')
            option_dumps.close()
            param_dumps.close()
            term_dumps.close()
            mask_dumps.close()
            if self.stream_write:
                self.act_dumps = open(os.path.join(self.save_path, "act_dumps.txt"), 'a')
                self.option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'a')
                self.term_dumps = open(os.path.join(self.save_path, "term_dumps.txt"), 'a')
                self.param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'a')
                self.mask_dumps = open(os.path.join(self.save_path, "mask_dumps.txt"), 'a')

        
        # shortcut calling option attributes through option
        self.state_extractor = self.option.state_extractor
        self.sampler = self.option.sampler # sampler manages either recalling the param, or getting a new one
        self.exploration_noise = exploration_noise
        self.temporal_aggregator = TemporalAggregator(name=option.name,sum_reward=args.collect.aggregator.sum_rewards, only_termination=args.collect.aggregator.only_termination)
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

    def stream_print(self):
        # if self.test: print(hit, next_target, self.state_extractor.reverse_obs_norm(obs, mask[0]),
        # pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)), action_chain, act)
        # if self.test: print(param, next_target, action_chain, inter, parent_state, pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)))
        # print(self.data.param[0], self.data.next_target[0], self.data.action_chain, self.data.inter[0] , self.state_extractor.reverse_obs_norm(self.data.obs), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)))
        if self.test:
            if len(self.stream_print_file) == 0:
                if np.any(self.data.terminate[0]): print("new_param", self.data.param[0])
                if np.any(self.data.done): print(self.counter, self.data.param[0], self.data.next_target, self.data.action_chain,self.data.act, self.data.inter[0] , self.data.rew[0], self.data.terminate[0],self.data.done[0], self.state_extractor.reverse_obs_norm(self.data.obs, self.data.mask.squeeze()), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)))
            else:
                stream_str = str(self.counter) + " ".join(map(str, [self.data.param[0], self.data.next_target, self.data.action_chain,self.data.act, self.data.inter[0],self.data.rew[0], self.data.terminate[0],self.data.done[0], self.state_extractor.reverse_obs_norm(self.data.obs, self.data.mask.squeeze()), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False))]))
                if np.any(self.data.terminate[0]): "new_param" + str(self.data.param[0]) + "\n" + stream_str
                self.stream_str_record.append(stream_str)
                print(stream_str)
                option_dumps = open(self.stream_print_file, 'w')
                option_dumps.write("\n".join(self.stream_str_record))
                option_dumps.close()
        # if resampled: print(action_chain[-1])
        # if self.test: print(inter, parent_state, action_chain, pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)))
        # print(ext_term, resampled, self.state_extractor.reverse_obs_norm(obs, mask[0])[6], self.state_extractor.reverse_obs_norm(obs_next, mask[0])[6],
        # pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)), action_chain)

    def restore_components(self, full_buffer, her_buffer, buffer, at, full_at, her_at):
        self.full_buffer = full_buffer
        self.her_buffer = her_buffer
        self.buffer = buffer
        self.at = at
        self.full_at = full_at
        self.her_at = her_at

    def get_components(self):
        return self.full_buffer, self.her_buffer, self.buffer, self.at, self.full_at, self.her_at

    def _save_action(self, act, action_chain, term_chain, param, mask): # this is handled here because action chains are option dependent
        if len(act.shape) == 0: act = [act]
        if self.stream_write:
            self.act_dumps.write(str(self.environment.get_itr() - 1) + ":" + action_chain_string(act) + "\n")
            self.option_dumps.write(str(self.environment.get_itr() - 1) + ":" + action_chain_string(action_chain) + "\n")
            self.term_dumps.write(str(self.environment.get_itr() - 1) + ":" + action_chain_string([term_chain]) + "\n")
            self.param_dumps.write(str(self.environment.get_itr() - 1) + ":" + action_chain_string([param[0]]) + "\n")
            self.mask_dumps.write(str(self.environment.get_itr() - 1) + ":" + action_chain_string([mask[0]]) + "\n")
        else:
            write_string(os.path.join(self.save_path, "act_dumps.txt"), str(self.environment.get_itr() - 1) + ":" + action_chain_string(act) + "\n")
            write_string(os.path.join(self.save_path, "option_dumps.txt"), str(self.environment.get_itr() - 1) + ":" + action_chain_string(action_chain) + "\n")
            write_string(os.path.join(self.save_path, "term_dumps.txt"), str(self.environment.get_itr() - 1) + ":" + action_chain_string([term_chain]) + "\n")
            write_string(os.path.join(self.save_path, "param_dumps.txt"), str(self.environment.get_itr() - 1) + ":" + action_chain_string([param[0]]) + "\n")
            write_string(os.path.join(self.save_path, "mask_dumps.txt"), str(self.environment.get_itr() - 1) + ":" + action_chain_string([mask[0]]) + "\n")

    def reset_env(self, keep_statistics: bool = False):
        full_state, info = self.env.reset()
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

    def adjust_param(self, param_mask=None, force=False):
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
            if param_mask is not None:
                param, mask = param_mask
                self.data.update(param=param, mask=mask)

            # Otherwise, already uses the same param as before
            # assign the param in the observation
            if self.data.obs is not None: # otherwise these have not been initialized yet and will be later
                self.data.obs = self.state_extractor.assign_param(self.data.full_state, self.data.obs, param, mask)
                self.data.obs_next = self.state_extractor.assign_param(self.data.full_state, self.data.obs_next, param, mask)
        return param, mask, new_param

    def update_statistics(self, hit_count, miss_count, drop_count, dropped):
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
            hit = ((check_close(self.option.terminate_reward.epsilon_close, self.option.terminate_reward.norm_p, self.data.next_target, self.data.param, self.data.mask) and not self.data.true_done[0])
                        or reward_check)
            print("hit", hit, self.data.next_target, self.data.param, drop_count, dropped)
        drop_count += int(dropped)
        hit_count += int(hit)
        miss_count += int(not hit and (not dropped or not self.time_check))
        return hit, hit_count, miss_count, drop_count

    def show_param(self, param, frame):
        if self.display_frame == 2:
            param = None
        if self.display_frame == 3: # display angles
            param.squeeze()[:2] = self.data.parent_state.squeeze()[:2]
            # print("param", param)
        frame = display_param(frame, param, rescale=8 if self.env_name in FULL_ENVS else 64, waitkey=10, transpose=self.environment.transpose)
        if len(self.save_display) > 0: imio.imsave(os.path.join(self.save_display, "state" + str(self.counter) + ".png"), frame)


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
        new_param: bool = False,
        demonstrate: bool = False,
        debug_actions: list = None,
        debug_states: list = None,
        debug: bool=False,
    ) -> Dict[str, Any]:
        """
        collects the data from the option, or random actions at the top level of the hierarchy
        """
        self.last_rec = self.data.full_state
        step_count, term_count, episode_count, true_episode_count = 0,0,0,0
        ready_env_ids = np.arange(1) # env_num is used in ts for multi-threaded environments, which are not used 
        last_true_done, rews, term, info = self.data.true_done, np.float64(0.0), False, dict()
        hit_count, miss_count,drop_count = 0,0,0
        start_time = time.time()
        used_new_param = new_param
        perf_times =dict()
        perf_times["action"] = 0
        perf_times["step"] = 0
        perf_times["term"] = 0
        perf_times["inline"] = 0
        perf_times["process"] = 0
        perf_times["record"] = 0
        perf_times["aggregate"] = 0
        perf_times["total"] = 0
        param_mask = None if debug_states is None or debug_states[0] is None else (debug_states[0][0], debug_states[1][0])
        param, mask, new_param = self.adjust_param(param_mask=param_mask, force=new_param)
        if new_param: print("new param start", param)
        itr = 0
        debug_record = list()
        her_list = None  # debugging variable
        while True:
            tc_start = time.time()
            self.counter += 1
            # set parameter for this run, if necessary
            param_mask = None if debug_states is None or debug_states[0] is None else (debug_states[0][itr], debug_states[1][itr])
            param, mask, new_param = self.adjust_param(param_mask=param_mask)
            # if self.test and new_param: print("new param", param)
            if new_param and not used_new_param: print("new param", param)
            # get the action chain
            state_chain = self.data.state_chain if hasattr(self.data, "state_chain") else None
            action = None
            if demonstrate:
                needs_sample, act, chain, policy_batch, state, masks = self.option.temporal_extension_manager.check(self.data.terminations[-1], self.data.ext_terms[-1])
                if needs_sample:
                    frame = self.environment.render()
                    print("target at: ", self.state_extractor.get_target(self.data.full_state[0]))
                    display_param(frame.astype(float) / 256.0, param, rescale=7, transpose=self.environment.transpose)
                    inp = ""
                    while len(inp) == 0:
                        inp = input("value: ")
                        try:
                            action = np.array([float(v) for v in inp.split(' ')])
                        except ValueError as e:
                            inp = ""
                            continue
                    action = np.array([float(v) for v in inp.split(' ')])
            with torch.no_grad(): act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, self.data.terminations, self.data.ext_terms, random=random, force=force, action=action)
            if debug_actions is not None: act, action_chain = debug_actions[itr]
            tc_action = time.time()
            self._policy_state_update(result)
            self.data.update(true_action=[action_chain[0]], act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled], action_chain = action_chain)
            
            # step in env
            action_remap = self.data.true_action
            if debug_states is not None:
                self.environment.set_from_factored_state(debug_states[2][itr]["factored_state"])
                obs_next = Batch([self.environment.get_state()])
                rew, done, info = obs_next["factored_state"]["Reward"], obs_next["factored_state"]["Done"], [self.environment.get_info()]
            else: obs_next, rew, done, trunc, info = self.env.step(action_remap, id=ready_env_ids)
            # print("done after step", done)
            # print(self.data.full_state.factored_state.Action)
            if self.environment.discrete_actions: self.data.full_state.factored_state.Action = [action_remap] # reassign the action to correspond to the current action taken
            else: self.data.full_state.factored_state.Action = action_remap
            # self.data.full_state.factored_state.Done = done # corrects for off by one by getting state after reset?
            # print(action_remap, self.data.full_state.factored_state.Action)
            next_full_state = obs_next[0] # only handling one environment
            true_done, true_reward = done, rew
            tc_step = time.time()

            # update the target, next target, obs, next_obs pair, state components
            obs = self.state_extractor.get_obs(self.data.last_full_state[0], self.data.full_state[0], param[0], mask[0]) # one environment reliance
            obs_next = self.state_extractor.get_obs(self.data.full_state[0], next_full_state, param[0], mask[0]) # one environment reliance
            target = self.state_extractor.get_target(self.data.full_state[0])
            next_target = self.state_extractor.get_target(next_full_state)
            # print (act, action_remap, action_chain[-1], self.data.full_state[0].factored_state.Gripper, self.data.full_state[0].factored_state.Block, self.env_reset, np.any(true_done) or (np.any(term) and self.terminate_reset))
            inter_state = self.state_extractor.get_inter(self.data.full_state[0])
            parent_state = self.state_extractor.get_parent(self.data.full_state[0])
            target_diff = self.state_extractor.get_diff(self.data.full_state[0], next_full_state)
            additional_state = self.state_extractor.get_additional(self.data.full_state[0])
            self.data.update(next_target=next_target, target=target, target_diff=target_diff, parent_state = parent_state, inter_state=inter_state, additional_state=additional_state, obs_next=[obs_next], obs = [obs])

            # get the dones, rewards, terminations and temporal extension terminations
            done, rewards, terminations, inter, cutoff = self.option.terminate_reward_chain(self.data.full_state[0], next_full_state, param, action_chain, mask, masks, true_done, true_reward)
            done, rew, term, ext_term = done, rewards[-1], terminations[-1], terminations[-2]# or resampled
            if self.save_action: self._save_action(act, action_chain, terminations, param, mask)
            info[0]["TimeLimit.truncated"] = bool(cutoff + info[0]["TimeLimit.truncated"]) if "TimeLimit.truncated" in info[0] else cutoff # environment might send truncated itself
            info[0]["TimeLimit.truncated"] = bool(self.trunc_true * true_done + info[0]["TimeLimit.truncated"]) # if we want to treat environment resets as truncations
            truncated = info[0]["TimeLimit.truncated"]
            self.option.update(act, action_chain, terminations, masks, update_policy=not self.test)
            # print(parent_state, target, param, act)
            # print(inter_state, self.option.interaction_model.predict_next_state(self.data.full_state))
            tc_term = time.time()
            # update inline training values
            proximity, proximity_inst, binaries = self.option.inline_trainer.set_values(self.data)
            tc_inline = time.time()

            # update hit-miss values
            rews += rew
            hit = False
            if term or done: hit, hit_count, miss_count, drop_count = self.update_statistics(hit_count, miss_count, drop_count, np.any(true_done) and not np.any(term))

            # update the current values TODO: next_full_state is time expensive (.001 sec per iteration). It should be stored separately
            self.data.update(next_full_state=[next_full_state], true_done=last_true_done, next_true_done=true_done, true_reward=true_reward, 
                param=param, mask = mask, info = info, inter = [inter], time=[1], trace = [np.any(self.environment.current_trace(self.names))],
                truncated=[truncated], terminated=[done],
                inst_trace=self.environment.current_trace(self.names), proximity=[proximity.squeeze()], 
                proximity_inst=[proximity_inst.squeeze()], weight_binary=[binaries.squeeze()],
                rew=[rew], done=[done], terminate=[term], ext_term = [ext_term], # all prior are stored, after are not 
                terminations= terminations, rewards=rewards, masks=masks, ext_terms=terminations[:len(terminations) - 1])
            # print(self.data.inter_state,type(self.data.inter_state))
            # print("term", terminations, inter, self.option.interaction_model.interaction(self.data, prenormalize=True))
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                ))
            tc_process = time.time()
            # render calls not usually used in our case
            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # Record the state
            # diff = self.data.full_state.factored_state.Block - self.last_rec.factored_state.Block
            # if np.sum(diff) >.01 or bool(self.data.done) or bool(self.data.true_done): 
            #     print(self.counter, diff, self.data.full_state.factored_state.Done, 
            #         self.data.done, self.data.true_done, true_done, self.data.terminate, cutoff)
            # if not self.test: print(self.counter, diff, self.data.full_state.factored_state.Block, self.data.full_state.factored_state.Done, 
            #     self.data.done, self.data.true_done, true_done, self.data.terminate, cutoff, self.environment.steps)
            # self.last_rec = copy.deepcopy(self.data.full_state)
            if self.record is not None:
                # print("recording", self.record.save_path)
                # print(self.data[0].full_state['factored_state']["Done"], self.data[0].full_state['factored_state']["Ball"], self.data[0].next_full_state['factored_state']["Ball"])
                self.record.save(self.data[0].full_state['factored_state'], self.data[0].full_state["raw_state"], self.environment.toString)
            tc_record = time.time()
            # we keep a buffer with all of the values
            self.data.done = np.array([self.data.done[0].astype(float)])
            full_ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(self.data)
            # print("adding data", self.data.done, self.data.terminate, self.full_buffer.done.shape, self.full_buffer.terminate.shape)
            # print(resampled, self.data.action_chain, self.data.act, self.data.full_state.factored_state.Paddle)
            self.stream_print()
            # print(self.counter, self.data.param[0], self.data.next_target, self.data.action_chain,self.data.act, self.data.inter[0] , self.data.rew[0], self.data.terminate[0],self.data.done[0], self.state_extractor.reverse_obs_norm(self.data.obs, self.data.mask.squeeze()), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, False)))

            # print(self.data[0].mapped_act, self.data[0].full_state.factored_state.Gripper, self.data.terminations[-1])
            # print(self.data[0].mapped_act, self.data.parent_state, self.data[0].param, self.data.terminations[-1])
            # add to the main buffer
            next_data, skipped, added, self.at = self._aggregate(self.data, self.buffer, full_ptr, ready_env_ids, interaction_model= self.option.interaction_model)
            # print(self.data.target, not self.test, self.hindsight is not None)
            if not self.test and self.hindsight is not None: self.her_at, her_list = self.her_collect(self.her_buffer, next_data, self.data, added, debug=debug)
            if self.display_frame > 0: self.show_param(param if self.display_frame < 4 else action_chain[-1], self.environment.render())
            tc_aggregate = time.time()
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
                    # print("handling next state", self.data.next_full_state[0].factored_state["Ball"])
                    self.data.next_full_state = [self.environment.get_state()]
                    self._reset_components(self.data.next_full_state[0])
                else:
                    self.reset_env()
                self.data.update(done=done, true_done = true_done)
            # assign progressive state
            self.data.prev_full_state = self.data.full_state
            self.data.full_state = self.data.next_full_state
            self.data.target = self.data.next_target
            self.data.obs = self.data.obs_next
            last_true_done = true_done
            tc_complete = time.time()
            itr += 1
            # print(f"times: action {tc_action - tc_start}, step {tc_step - tc_action} term {tc_term - tc_step} inline {tc_inline - tc_term} process {tc_process - tc_inline} record {tc_record - tc_process} aggregate {tc_aggregate - tc_complete} total {tc_complete - tc_start}")
            perf_times["action"] = perf_times["action"] + tc_action - tc_start 
            perf_times["step"] = perf_times["step"] + tc_step - tc_action 
            perf_times["term"] = perf_times["term"] + tc_term - tc_step 
            perf_times["inline"] = perf_times["inline"] + tc_inline - tc_term 
            perf_times["process"] = perf_times["process"] + tc_process - tc_inline 
            perf_times["record"] = perf_times["record"] + tc_record - tc_process 
            perf_times["aggregate"] = perf_times["aggregate"] + tc_aggregate - tc_complete 
            perf_times["total"] = perf_times["total"] + tc_complete - tc_start

            # debug record
            if debug: debug_record.append((added, copy.deepcopy(self.data), copy.deepcopy(next_data), her_list))

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
            "n/dr": drop_count,
            "n/tim": self.collect_time, 
            "rews": rews,
            "terminate": (not np.any(true_done)) and np.any(term) and self.terminate_reset,
            "info": info,
            "perf": perf_times,
            "debug": debug_record,
        }
