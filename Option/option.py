import numpy as np
import os, cv2, time, copy
import torch
import gym
from Network.network_utils import pytorch_model
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

def load_option(load_dir, name, interaction_model=None, device=-1):
    load_name = os.path.join(load_dir, name + "_option.opt")
    try: 
        option = torch.load(load_name)
    except FileNotFoundError as e:
        return None
    if interaction_model is not None:
        option.assign_interaction_model(interaction_model)
    if device != -1:
        option.cuda(device=device)
    return option

class Option():
    def __init__(self, args, policy, models, next_option):
        # parameters for saving
        self.name = args.object_names.target
        self.train_epsilon = args.policy.epsilon_random

        # primary models
        self.policy = policy # policy to run during opion
        self.next_option = next_option # the option which controls the actions
        self.next_option_name = next_option.name
        self.save_inline = args.inline.save_inline
        self.next_option.zero_epsilon()

        self.interaction_model = models.interaction_model
        self.sampler = models.sampler # samples params
        self.temporal_extension_manager = models.temporal_extension_manager # manages when to temporally extend
        self.state_extractor = models.state_extractor # extracts the desired state
        self.terminate_reward = models.terminate_reward # handles termination, reward and temporal extension termination
        self.action_map = models.action_map # object dict with action spaces
        self.initiation_set = None # TODO: handle initiation states
        self.test_sampler = models.test_sampler # a specialized sampler for testing
        self.inline_trainer = models.inline_trainer

    def reassign_norm(self, environment):
        norm = self.interaction_model.regenerate_norm()
        self.state_extractor.norm = norm

    def set_device(self, device_no):
        device = 'cpu' if not self.iscuda else 'cuda:' + str(device_no)
        if self.policy is not None:
            self.policy.to(device)
            self.policy.assign_device(device)
        if self.next_option is not None:
            self.next_option.set_device(device_no)

    def cuda(self, device=None):
        self.iscuda = True
        if device is not None: self.device=device
        if self.policy is not None: self.policy.cuda(device=device)
        if self.next_option is not None: self.next_option.cuda(device=device)
        if self.interaction_model is not None: self.interaction_model.cuda(device=device)

    def cpu(self): # does NOT set device
        self.iscuda = False
        if self.policy is not None: self.policy.cpu()
        if self.next_option is not None: self.next_option.cpu()
        if self.interaction_model is not None: self.interaction_model.cpu()

    def toggle_test(self, test):
        if test:
            self.zero_epsilon()
            self.sampler, self.test_sampler = self.test_sampler, self.sampler # use test_sampler to keep train sampler
        else:
            self.set_epsilon(self.train_epsilon)
            self.sampler, self.test_sampler = self.test_sampler, self.sampler

    def zero_epsilon(self):
        if self.policy is not None:
            self.policy.set_eps(0.0)
        if self.next_option is not None:
            self.next_option.zero_epsilon()
        return self.train_epsilon

    def set_epsilon(self, eps):
        if self.policy is not None:
            self.train_epsilon = eps
            self.policy.set_eps(eps)

    def _set_next_option(self, batch, mapped_act):
        next_batch = copy.deepcopy(batch)
        next_batch["mask"] = [self.next_option.sampler.mask.active_mask]
        next_batch["param"] = self.state_extractor.expand_param(mapped_act, self.next_option.sampler.mask.active_mask)
        next_batch['obs'] = self.next_option.state_extractor.get_obs(batch["last_full_state"], batch["full_state"], next_batch["param"], next_batch["mask"])
        return next_batch

    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False, force=None):
        '''
        get a new action (resample) or not based on the result of TEM.check. If we don't, check downstream options
        batch must contain full_state and termination_chain
        '''
        needs_sample, act, chain, policy_batch, state, masks = self.temporal_extension_manager.check(term_chain[-1], ext_terms[-1])
        if needs_sample: result_tuple = self.sample_action_chain(batch, state_chain, random=random)
        else: # if we don't need a new sample
            next_batch = self._set_next_option(batch, chain[-1])
            new_act, rem_chain, pol_batch, rem_state, rem_masks, last_resmp = self.next_option.extended_action_sample(next_batch, state_chain, term_chain[:-1], ext_terms[:-1], random=False, use_model=use_model)
            result_tuple = (act, rem_chain + [chain[-1]], policy_batch, rem_state + [state[-1]] if state is not None else None, rem_masks + [masks[0]])
        return (*result_tuple, needs_sample)

    def sample_action_chain(self, batch, state_chain, random=False): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
        use_model is only for model based search
        if the batch object contains a partial flag (key with PARTIAL=1), then treat the state as a partial
        @param force forces the action to be a particular value
        '''
        if random:
            act = self.action_map.sample_policy_space()
            mapped_act = self.action_map.map_action(act, batch[0])
            state, policy_batch = None, Batch()
        else:
            policy_batch = self.policy.forward(batch, state_chain[-1] if state_chain is not None else None)
            state = policy_batch.state
            act, mapped_act = pytorch_model.unwrap(policy_batch.act[0]), self.action_map.map_action(policy_batch.act[0], batch[0])
        chain = [mapped_act.squeeze()]
        # recursively propagate action up the chain
        if self.next_option is not None:
            next_batch = self._set_next_option(batch, mapped_act)
            next_policy_act, rem_chain, result, rem_state_chain, last_masks = self.next_option.sample_action_chain(next_batch, state_chain[-1] if state_chain is not None else None) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain, state, masks = rem_chain + chain, rem_state_chain + [state], last_masks + [batch.mask[0]] # TODO: mask is not set from the policy
        return act, chain, policy_batch, state, masks

    def reset(self, full_state):
        # reset the timers for temporal extension, termination
        init_terms = self.next_option.reset(full_state)
        self.temporal_extension_manager.reset()
        return init_terms + [False]

    def update(self, act, chain, term_chain, masks, update_policy=True):
        # updates internal states of the option, and asssociated components
        if self.next_option is not None:
            self.next_option.update(chain[len(chain)-1], chain[:len(chain)-1], term_chain[:len(term_chain)-1], masks[:len(masks)-1], update_policy = False)
        self.temporal_extension_manager.update(act, chain, term_chain[-2], masks)
        self.terminate_reward.update()
        self.sampler.update()
        if update_policy:
            self.policy.update_epsilon()

    def terminate_reward_chain(self, full_state, next_full_state, param, chain, mask, mask_chain, true_done= None, true_reward=None):
        # recursively get all of the dones and rewards
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            next_param, next_mask = chain[-1], mask_chain[-2]
            next_param = self.state_extractor.expand_param(next_param, self.next_option.sampler.mask.active_mask)
            last_done, last_rewards, last_termination, _, _ = self.next_option.terminate_reward_chain(full_state, next_full_state, next_param, chain[:len(chain)-1], next_mask, mask_chain[:len(mask_chain)-1], true_done=true_done, true_reward=true_reward)
            # print(self.name, last_done, last_rewards, last_termination)
        # slightly confusing: inter_state is normalized, but next_target is not, because of how goal state is handled
        inter_state, target, next_target = self.state_extractor.get_inter(full_state, norm=True), self.state_extractor.get_target(full_state), self.state_extractor.get_target(next_full_state)
        term, reward, done, inter, time_cutoff = self.terminate_reward(inter_state, target, next_target, param, mask, true_done, true_reward) # TODO: true_inter = ?
        # print(self.name, term, reward, done, inter, time_cutoff)
        rewards, terminations = last_rewards + [reward], last_termination + [term]
        return done, rewards, terminations, inter, time_cutoff

    def assign_interaction_model(self, interaction_model):
        inter_model = self.interaction_model
        if interaction_model is not None: self.sampler.mask = interaction_model.mask # reassigns the mask in case it changed
        self.terminate_reward.interaction_model = interaction_model
        self.inline_trainer.interaction_model = interaction_model
        self.interaction_model = interaction_model
        return inter_model

    def save(self, save_dir):
        # checks and prepares for saving option as a pickle
        save_name = os.path.join(save_dir, self.name + "_option.opt") # TODO: have separate save/load for terminate_reward
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            inter_model = None
            self.cpu()
            if self.interaction_model is not None:
                inter_model = self.assign_interaction_model(None) # don't save interaction model in option
                if inter_model is not None and self.save_inline: inter_model.save(save_dir)
            torch.save(self, save_name)
            self.assign_interaction_model(inter_model)
            self.cuda(device=self.device)
            return self
        return None