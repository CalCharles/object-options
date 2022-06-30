import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
import cv2
import time
import tianshou as ts
from collections import deque
from Network.network_utils import pytorch_model


from Record.file_management import load_from_pickle


class Hindsight():
    # TODO: adapt to work with TianShou
    def __init__(self, args, option, interaction_model):
        # only sample one other goal (the successful one)
        # self.rollouts = RLRollouts(option.rollouts.length, option.rollouts.shapes)
        self._hypothesize = interaction_model.hypothesize
        self._check_interaction = interaction_model.test
        self._get_mask_param = option.state_extractor.param_mask
        self.state_extractor = option.state_extractor
        self.terminate_reward = option.terminate_reward

        self.at = 0
        self.last_done = 0
        self.last_res = 0
        self.sample_timer = 0
        self.use_sum_rewards = args.collect.aggregator.sum_rewards
        self.resample_timer = args.hindsight.resample_timer
        self.select_positive = args.hindsight.select_positive
        self.use_interact = args.hindsight.interaction_resample
        self.max_hindsight = args.hindsight.max_hindsight if args.hindsight.max_hindsight > 0 else 10000
        self.replay_queue = deque(maxlen=self.max_hindsight)
        self.multi_instanced = interaction_model.multi_instanced


        # stores single states for reward summing
        self.between_replay = [0]
        self.between_replay_counter = 0
        self.single_batch_queue = list()

        self.early_stopping = args.hindsight.early_stopping # stops after seeing early stopping number of termiantions
        self.interaction_criteria = args.hindsight.interaction_criteria # only resamples if at least one interaction occurs
    
    def reset(self, first_state):
        self.sample_timer = 0
        del self.replay_queue
        del self.single_batch_queue
        self.replay_queue = deque(maxlen=self.max_hindsight)
        self.single_batch_queue = deque(maxlen=self.max_hindsight)
        self.between_replay = [0] # will append zero
        self.between_replay_counter = 0
        self.single_batch_queue.append(first_state)

    def step(self):
        self.sample_timer += 1

    def satisfy_criteria(self, replay_queue):
        return ((self.interaction_criteria == 1 and total_interaction > 0.5) or
         (self.interaction_criteria == 2 and total_change > 0.001) or
          (self.interaction_criteria == 0)) # only keep cases where an interaction occurred in the trajectory TODO: interaction model unreliable, use differences in state instead


    def sum_rewards(self, i, param, mask):
        rew = 0.0
        between_pair = (self.between_replay[-i-1], self.between_replay[-i])
        for j in range(*between_pair):
            old_batch = self.single_batch_queue[j]
            # only use the last termination, not if any intermediate terminations occurred
            _, ss_rew, _, _, _ = self.terminate_reward(old_batch.full_state[0], old_batch.next_full_state[0], param, mask, true_done=old_batch.true_done, true_reward=old_batch.true_reward)
            rew += ss_rew
        return rew

    def record_state(self, her_buffer, full_batch, single_batch, added):
        '''
        full_batch is (state, next_state) from the aggregator, handling temporal extension
        single_batch is (state, next_state) according to the environment
        '''

        # add the new state for a single set. We need every individual state for reward computation
        self.single_batch_queue.append(copy.deepcopy(single_batch))
        if added: self.replay_queue.append(copy.deepcopy(full_batch))
        self.between_replay_counter += 1 # counts the number of states for each replay queue state (TODO: should equal full_batch.timer)

        # determine if a resampling occurred
        resetted = False
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
                param = next_instances[idx]
            else:
                param = full_batch.next_target[0]# self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
            
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
                    full_state, next_full_state = self.single_batch_queue[self.between_replay[-i]].full_state, self.single_batch_queue[self.between_replay[-i]].next_full_state # index errors if you terminate after a single step
                    term, rew, done, inter, time_cutoff = self.terminate_reward(full_state, next_full_state, param, mask, batch.true_reward[0], batch.true_done[0])
                    rew = rew if self.use_sum_rewards else self.sum_rewards(i, param, mask)

                    her_batch.info["TimeLimit.truncated"] = [False] # no time cutoff for HER
                    her_batch.update(done=[done], terminate=[term], rew=[rew])
                    rv_search.append(her_batch)

                # Now add all the of the her_batches in (in the reverse order of the reverse order)
                for i in range(1, len(rv_search)+1):
                    her_batch = rv_search[-i]
                    self.at, ep_rew, ep_len, ep_idx = her_buffer.add(her_batch, buffer_ids=[0])

            # reset queues and timers
            self.reset(self.single_batch_queue[-1])
            resetted = True

        # since this was called, a between replay counter must be appended
        if added and not resetted: self.between_replay.append(self.between_replay_counter)
        return self.at

    def get_buffer_idx(self):
        return self.at

    def sample_buffer(self, buffer, her_buffer):
        if np.random.random() > self.select_positive or len(her_buffer) == 0:
            return buffer
        else:
            return her_buffer
