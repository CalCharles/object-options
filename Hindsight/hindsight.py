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

def mean_replay_difference(replay_queue):
    mean = np.mean( np.stack([b.target[0] for b in replay_queue]), axis = 0)
    return np.sum(np.linalg.norm(np.stack([b.target[0] - mean for b in replay_queue]), ord=1))

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
        self.min_replay_len = args.hindsight.min_replay_len

    def reset(self, first_state):
        self.sample_timer = 0
        del self.replay_queue
        del self.single_batch_queue
        self.replay_queue = deque(maxlen=self.max_hindsight)
        self.single_batch_queue = list() #[copy.deepcopy(first_state)]
        self.between_replay = deque(maxlen=self.max_hindsight + 1)
        self.between_replay.append(0) # will append 0 since between_replay is always +1 of replay queue
        self.between_replay_counter = 0

    def step(self):
        self.sample_timer += 1


    def satisfy_criteria(self, replay_queue):
        return len(replay_queue) > 0 and ((self.interaction_criteria == 1 and np.sum(np.stack([b.inter for b in replay_queue])) > 0.5) or # keep if interaction happened
         (self.interaction_criteria == 2 and mean_replay_difference(replay_queue) > 0.001) or # keep if displacement happened
          (self.interaction_criteria == 0)) # just keep any termination


    def sum_rewards(self, i, param, mask):
        rew = 0.0
        between_pair = (max(1, self.between_replay[i]), self.between_replay[i+1]) if self.between_replay[i+1] != 1 else (self.between_replay[i],self.between_replay[i+1])
        for j in range(*between_pair):
            old_batch = self.single_batch_queue[j]
            # only use the last termination, not if any intermediate terminations occurred
            inter_state, target, next_target = self.state_extractor.get_inter(old_batch.full_state[0], norm=True), self.state_extractor.get_target(old_batch.full_state[0]), self.state_extractor.get_target(old_batch.next_full_state[0])
            _, ss_rew, _, _, _ = self.terminate_reward(inter_state, target, next_target, param, mask, true_done=old_batch.true_done, true_reward=old_batch.true_reward, reset=False)
            rew += ss_rew
        return rew

    def record_state(self, her_buffer, full_batch, single_batch, added):
        '''
        full_batch is (state, next_state) from the aggregator, handling temporal extension
        single_batch is (state, next_state) according to the environment
        '''

        # add the new state for a single set. We need every individual state for reward computation
        self.single_batch_queue.append(copy.deepcopy(single_batch))
        # print(single_batch.target, single_batch.next_target, single_batch.inter)
        if added: self.replay_queue.append(copy.deepcopy(full_batch))
        self.between_replay_counter += 1 # counts the number of states for each replay queue state (TODO: should equal full_batch.timer)

        # determine if a resampling occurred
        resetted = False
        term_resample = np.any(full_batch.done) or np.any(full_batch.terminate)
        timer_resample = self.resample_timer > 0 and (self.sample_timer >= self.resample_timer) and not term_resample
        inter_resample = self._check_interaction(full_batch.inter.squeeze()) and self.use_interact # TODO: uses the historical computation
        if (term_resample or timer_resample or inter_resample): # either end of episode or end of timer, might be good to have if interaction, this relies on termination == interaction in the appropriate place
            mask = full_batch.mask[-1] # TODO: the mask must be fixed through a trajectory
            self.between_replay.append(self.between_replay_counter)

            # if HER only records interactions or sufficient change, determine here
            # if len(self.replay_queue) > 0:
            #     print("queue check", len(self.replay_queue),added,np.sum(np.stack([b.inter for b in self.replay_queue])),  ((self.interaction_criteria == 1 and np.sum(np.stack([b.inter for b in self.replay_queue])) > 0.5) , # keep if interaction happened
         # (self.interaction_criteria == 2 and mean_replay_difference(self.replay_queue) > 0.001), # keep if displacement happened
         #  (self.interaction_criteria == 0)))
            satisfied = self.satisfy_criteria(self.replay_queue)

            # get the hindsight target. For multi-instanced, it is the state of the object interacted with
            if self.multi_instanced: # if multiinstanced, param should not be masked, and needs to be defined by the instance, not just the object state
                next_instances = split_instances(self.state_extractor.get_target(single_batch.next_full_state[0]))
                idx = np.argwhere(dataset_model.test(full_batch.inter.squeeze()))[0].squeeze()
                param = self.state_extractor.param_mask(next_instances[idx], mask, normalize = False)
            else:
                param = self.state_extractor.param_mask(full_batch.next_target[0], mask, normalize = False)# self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
            
            if satisfied: # multi_keep checks if there is a valid parameter, if there is not then we can't run HER
                # first, adjust for the new parameter reward, termination, time cutoff and done signals in reverse order
                add_queue = list()
                # print("single_batch_queue", len(self.single_batch_queue), [(s.target, s.next_target) for s in self.single_batch_queue])
                for i in range(len(self.replay_queue)): # iterate forward through the replay queue, restarting if there is a done
                    batch = self.replay_queue[i]
                    her_batch = copy.deepcopy(batch)

                    # update param, obs, obs_next
                    her_batch.update(param=[param.copy()], obs = self.state_extractor.assign_param(batch.full_state[0], batch.obs, param, mask),
                        obs_next = self.state_extractor.assign_param(batch.next_full_state[0], batch.obs_next, param, mask))

                    # get terminate, done and reward terms
                    # print("single_batch", self.between_replay[i+1], len(self.single_batch_queue))
                    full_state, next_full_state = self.single_batch_queue[self.between_replay[i+1]-1].full_state, self.single_batch_queue[self.between_replay[i+1]-1].next_full_state # index errors if you terminate after a single step
                    # print(self.between_replay[i+1]-1, full_state.factored_state.Ball, next_full_state.factored_state.Ball, her_batch.inter_state, her_batch.target, her_batch.next_target)
                    # print(self.single_batch_queue[self.between_replay[i+1]-1].target, self.single_batch_queue[self.between_replay[i+1]-1].next_target)
                    inter_state, target, next_target = self.state_extractor.get_inter(full_state, norm=True), self.state_extractor.get_target(full_state), self.state_extractor.get_target(next_full_state)
                    term, rew, done, inter, time_cutoff = self.terminate_reward(inter_state, target, next_target, param, mask, batch.true_reward[0], batch.true_done[0], reset=False)
                    rew = rew if not self.use_sum_rewards else self.sum_rewards(i, param, mask)

                    her_batch.info["TimeLimit.truncated"] = [False] # no time cutoff for HER
                    her_batch.update(done=[done], terminate=[term], rew=[rew], old_inter=copy.deepcopy(her_batch.inter), inter=[inter])
                    add_queue.append(her_batch)
                    if np.any(batch.done) and i != len(self.replay_queue) - 1:
                        add_queue = list()

                # Now add all the of the her_batches in (in the reverse order of the reverse order)
                if len(add_queue) > self.min_replay_len:
                    for i in range(len(add_queue)):
                        her_batch = add_queue[i]
                        # print(self.terminate_reward.inter_extract(full_state, norm=True)), pytorch_model.unwrap(self.terminate_reward.interaction_model.interaction(self.terminate_reward.inter_extract(full_state, norm=True))))
                        # print("adding", her_batch.target, her_batch.next_target, her_batch.inter, her_batch.old_inter, her_batch.rew, her_batch.done, her_batch.param)
                        self.at, ep_rew, ep_len, ep_idx = her_buffer.add(her_batch, buffer_ids=[0])
                # else:
                #     print("queue len", len(add_queue))
            # reset queues and timers
            # print("resetting", len(self.single_batch_queue))
            # print("reset", len(self.single_batch_queue), len(self.replay_queue), single_batch.target, single_batch.next_target, full_batch.target, full_batch.next_target, term_resample, timer_resample, inter_resample)
            self.reset(single_batch)
            resetted = True
            # print("resetting")

        # since this was called, a between replay counter must be appended
        if added and not resetted: 
            self.between_replay.append(self.between_replay_counter)
        return self.at

    def get_buffer_idx(self):
        return self.at

    def sample_buffer(self, buffer, her_buffer):
        if np.random.random() > self.select_positive or len(her_buffer) == 0:
            return buffer
        else:
            return her_buffer
