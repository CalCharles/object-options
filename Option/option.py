import numpy as np
import os, cv2, time, copy
import torch
import gym
from Networks.network import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from EnvironmentModels.environment_model import FeatureSelector, cpu_state
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from file_management import suppress_stdout_stderr

class Option():
    def __init__(self, args, policy, next_option):
        # parameters for saving
        self.name = args.object

        # primary models
        self.sampler = models.sampler # samples params
        self.policy = policy # policy to run during opion
        self.next_options = next_options # the option which controls the actions
        self.next_option.zero_epsilon()

    def assign_models(self, models):
        # assigns state extractor, rew_term_done, action_map, 
        self.state_extractor = models.state_extractor # extracts the desired state
        self.terminate_reward = models.terminate_reward # handles termination, reward and temporal extension termination
        self.action_map = models.action_map # object dict with action spaces
        self.temporal_extension_manager = models.temporal_extension_manager # manages when to temporally extend
        self.initiation_set = None # TODO: handle initiation states

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
        if self.sampler is not None: self.sampler.cuda()
        if self.next_option is not None: self.next_option.cuda(device=device)

    def cpu(self): # does NOT set device
        self.iscuda = False
        if self.policy is not None: self.policy.cpu()
        if self.sampler is not None: self.sampler.cpu()
        if self.next_option is not None: self.next_option.cpu()

    def zero_epsilon(self):
        if self.policy is not None:
            self.policy.set_eps(0.0)
            self.action_map.assign_policy_map(self.policy.map_action, self.policy.reverse_map_action, self.policy.exploration_noise)
        if type(self.next_option) != PrimitiveOption:
            self.next_option.zero_epsilon()

    def set_epsilon(self, eps):
        if self.policy is not None:
            self.policy.set_eps(eps)
            self.action_map.assign_policy_map(self.policy.map_action, self.policy.reverse_map_action, self.policy.exploration_noise)

    def _set_next_option(self, batch, mapped_act):
        next_batch = copy.deepcopy(batch)
        next_batch["mask"] = self.next_option.sampler.mask
        next_batch["param"] = [mapped_act]
        next_batch['obs'] = self.next_option.state_extractor.get_obs(batch["full_state"], next_batch["param"], next_batch["mask"])
        return next_batch, param, obs, mask

    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False, force=None):
        '''
        get a new action (resample) or not based on the result of TEM.check. If we don't, check downstream options
        batch must contain full_state and termination_chain
        '''
        needs_sample, act, chain, policy_batch, state, masks = self.temporal_extension_manager.check(term_chain[-1], ext_terms[-1])
        if needs_sample: result_tuple = self.sample_action_chain(batch, state_chain, random=random)
        else: # if we don't need a new sample
            next_batch, param, obs, mask = self._set_next_option(batch, chain[-1])
            new_act, rem_chain, pol_batch, rem_state, rem_masks, last_resmp = self.next_option.extended_action_sample(next_batch, state_chain, term_chain[:-1], ext_terms[:-1], random=False, use_model=use_model)
            result_tuple = (act, rem_chain + [chain[-1]], policy_batch, rem_state + [state[-1]] if state is not None else None, rem_masks + [mask[0]])
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
            act, mapped_act = self.action_map.sample_policy_space()
            state, policy_batch = None, Batch()
        else:
            policy_batch = self.policy.forward(batch, state_chain[-1] if state_chain is not None else None) # uncomment this
            state = policy_batch.state
            act, mapped_act = self.action_map.map_action(policy_batch.act, batch)
        chain = [mapped_act]
        # recursively propagate action up the chain
        if self.next_option is not None:
            next_batch, param, obs, mask = self._set_next_option(batch, mapped_act)
            next_policy_act, rem_chain, result, rem_state_chain, last_masks = self.next_option.sample_action_chain(next_batch, state_chain[-1] if state_chain is not None else None) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain, state, masks = rem_chain + chain, rem_state_chain + [state], last_masks + [mask[0]] # TODO: mask is one dimension too large because batch has multiple environments
        return act, chain, policy_batch, state, masks

    def reset(self, full_state):
        # reset the timers for temporal extension, termination
        init_terms = self.next_option.reset(full_state)
        self.temporal_extension_manager.reset()
        return init_terms + [init_term]

    def update(self, act, chain, term_chain, masks, update_policy=True):
        # updates internal states of the option, and asssociated components
        if self.next_option is not None:
            self.next_option.update(chain[:len(chain)-1], chain[:len(chain)-1], term_chain[:len(term_chain)-1], masks[:len(masks)-1], update_policy = False)
        self.temporal_extension_manager.update(act, chain, term_chain[-2], masks)
        self.rew_term_done.update(term_chain[-1])
        self.sampler.update() # TODO: sampler also handles its own param, mask
        if update_policy:
            self.policy.update()

    def terminate_reward_chain(self, full_state, next_full_state, param, chain, mask, mask_chain, environment_model=None):
        # recursively get all of the dones and rewards
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            next_param, next_mask = self.next_option.sampler.convert_param(chain[-1]), mask_chain[-2]
            last_done, last_rewards, last_termination, last_ext_term, _, _ = self.next_option.terminate_reward_chain(full_state, next_full_state, next_param, chain[:len(chain)-1], next_mask, mask_chain[:len(mask_chain)-1])
        term, reward, inter, done, time_cutoff = self.rew_term_done(full_state, next_full_state, param, mask, true_inter=true_inter)
        rewards, terminations = last_rewards + [reward], last_termination + [termination]
        return done, rewards, terminations, inter, time_cutoff

    def save(self, save_dir):
        # checks and prepares for saving option as a pickle
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.policy.cpu()
            return self
        return None