import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from Networks.network import pytorch_model
import cv2
import time
import tianshou as ts
from Rollouts.param_buffer import ParamReplayBuffer, ParamPriorityReplayBuffer
from collections import deque



from file_management import load_from_pickle


class Hindsight():
    # TODO: adapt to work with TianShou
    def __init__(self, args, option):
        super().__init__(args, option)
        # only sample one other goal (the successful one)
        # self.rollouts = RLRollouts(option.rollouts.length, option.rollouts.shapes)
        if len(args.prioritized_replay) > 0:
            self.replay_buffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
        else:
            self.replay_buffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

        self._hypothesize = option.dataset_model.hypothesize
        self._check_interaction = option.terminate_reward.check_interaction
        self._get_mask_param = option.sampler.get_mask_param
        self.state_extractor = args.state_extractor
        self.done_model = option.done_model
        self.terminate_reward = option.terminate_reward

        self.at = 0
        self.last_done = 0
        self.last_res = 0
        self.sample_timer = 0
        self.sum_rewards = args.sum_rewards
        self.resample_timer = args.resample_timer
        self.select_positive = args.select_positive
        self.use_interact = not args.true_environment and args.use_interact
        self.max_hindsight = args.max_hindsight
        self.replay_queue = deque(maxlen=args.max_hindsight)
        self.multi_instanced = self.option.dataset_model.multi_instanced


        # stores single states for reward summing
        self.between_replay = [0]
        self.between_replay_counter = 0
        self.single_batch_queue = list()

        self.early_stopping = args.early_stopping # stops after seeing early stopping number of termiantions
        self.only_interaction = args.her_only_interact # only resamples if at least one interaction occurs
        self.keep_hits = args.learning_type[3:] == "isl" and args.keep_proximity

        # proximity value
        if self.keep_hits: self.dist = self.option.dataset_model.block_width+ self.option.dataset_model.block_height # TODO: makes ISL only compatible with block targeting
    
    def reset(self):
        self.sample_timer = 0
        del self.replay_queue
        del self.single_batch_queue
        self.replay_queue = deque(maxlen=self.max_hindsight)
        self.single_batch_queue = list()
        self.between_replay = list() # will append zero
        self.between_replay_counter = 0


    def step(self):
        self.sample_timer += 1

    def satisfy_criteria(self, replay_queue):
        return ((self.only_interaction == 1 and total_interaction > 0.5) or
         (self.only_interaction == 2 and total_change > 0.001) or
          (self.only_interaction == 0)) # only keep cases where an interaction occurred in the trajectory TODO: interaction model unreliable, use differences in state instead


    def sum_rewards(rew):
        rew = 0
        between_pair = (self.between_replay[-i-1], self.between_replay[-i])
        # print(between_pair)
        for j in range(*between_pair):
            old_batch = self.single_batch_queue[j]
            # only use the last termination, not if any intermediate terminations occurred
            _, ss_rew, _, _ = self.option.terminate_reward.check(old_batch.full_state[0], old_batch.next_full_state[0], param, mask, inter_state=batch.inter_state[0], use_timer=False, true_inter=old_batch.inter.squeeze())
            rew += ss_rew
            # print(rew, ss_rew, j, old_batch.full_state["factored_state"]["Block"], old_batch.target, old_batch.next_target, old_batch.mapped_act, param)


    def record_state(self, full_batch, single_batch, added):
        '''
        full_batch is (state, next_state) from the aggregator, handling temporal extension
        single_batch is (state, next_state) according to the environment
        '''

        # add the new state for a single set. We need every individual state for reward computation
        self.single_batch_queue.append(copy.deepcopy(single_batch))
        self.replay_queue.append(copy.deepcopy(full_batch))
        self.between_replay_counter += 1 # counts the number of states for each replay queue state (TODO: should equal full_batch.timer)

        # determine if a resampling occurred
        term_resample = np.any(full_batch.done) or np.any(full_batch.terminate)
        timer_resample = self.resample_timer > 0 and (self.sample_timer >= self.resample_timer) and not term_resample
        inter_resample = self._check_interaction(full_batch.inter.squeeze()) and self.use_interact # TODO: uses the historical computation

        if (term_resample or timer_resample or inter_resample): # either end of episode or end of timer, might be good to have if interaction, this relies on termination == interaction in the appropriate place
            mask = full_batch.mask[0]
            self.between_replay.append(self.between_replay_counter)

            # if HER only records interactions or sufficient change, determine here
            satisfied = self.satisfy_criteria(self.replay_queue)

            # get the hindsight target. For multi-instanced, it is the state of the object interacted with
            if self.multi_instanced: # if multiinstanced, param should not be masked, and needs to be defined by the instance, not just the object state
                next_instances = split_instances(self.state_extractor.get_target(single_batch.next_full_state[0]))
                idx = np.argwhere(dataset_model.test(full_batch.inter.squeeze()))[0].squeeze()
                param = self._get_mask_param(next_instances[idx], mask)
            else:
                param = self._get_mask_param(full_batch.next_target[0], mask)# self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
            
            if satisfied: # multi_keep checks if there is a valid parameter, if there is not then we can't run HER
                
                # first, adjust for the new parameter reward, termination, time cutoff and done signals in reverse order
                rv_search = list()
                for i in range(1, len(self.replay_queue) + 1): # go back in the replay queue, but stop if last_done is hit
                    # work backward from the most recent
                    batch = self.replay_queue[-i]
                    her_batch = copy.deepcopy(batch)

                    # update param, obs, obs_next
                    her_batch.update(param=[param.copy()], obs = self.state_extractor.assign_param(batch.full_state[0], batch.obs, param, mask),
                        obs_next = self.state_extractor.assign_param(batch.next_full_state[0], batch.obs_next, param, mask))

                    # get terminate, done and reward terms
                    term, rew, inter, time_cutoff = self.option.term_rew_done(batch.full_state[0], batch.next_full_state[0], param, mask, batch.true_reward[0], batch.true_done[0])
                    rew = self.sum_rewards()

                    her_batch.info["TimeLimit.truncated"] = False # no time cutoff for HER
                    her_batch.update(done=[done], terminate=[term], rew=[rew])
                    rv_search.append(her_batch)

                # Now add all the of the her_batches in (in the reverse order of the reverse order)
                for i in range(1, len(rv_search)+1):
                    her_batch = rv_search[-i]                    
                    self.at, ep_rew, ep_len, ep_idx = self.replay_buffer.add(her_batch, buffer_ids=[0])

            # reset queues and timers
            self.reset()

        # since this was called, a between replay counter must be appended
        if added: self.between_replay.append(self.between_replay_counter)

    def get_buffer_idx(self):
        return self.at

    def sample_buffer(self, buffer):
        if np.random.random() > self.select_positive or len(self.replay_buffer) == 0:
            return buffer
        else:
            return self.replay_buffer
