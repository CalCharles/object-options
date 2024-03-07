import numpy as np
import os, cv2, time, copy
import torch
import gym
from Network.network_utils import pytorch_model
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

def load_option(load_dir, name, interaction_model=None, device=-1, use_cuda = True):
    load_name = os.path.join(load_dir, name + "_option.opt")
    try: 
        option = torch.load(load_name)
    except FileNotFoundError as e:
        return None
    if interaction_model is not None:
        option.assign_interaction_model(interaction_model)
    if device != -1:
        option.cuda(device=device)
    else: # device is -1
        if not use_cuda:
            option.cpu()
    return option

class Option():
    def __init__(self, args, policy, models, next_option, debug=False):
        # parameters for saving
        self.name = args.object_names.target if len(args.train.override_name) <= 0 else args.train.override_name
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
        self.inline_trainer = models.inline_trainer # a trainer for the interaction model
        self.reset_timer = 0 # storage for the temporal extension timer
        self.zero_epsilon_test = args.option.zero_epsilon_test

        # debugging mode stores intermediate state
        self.set_debug(debug)
        self.debug_stack = list()

        self.match_temporal_extension()

    def get_depth(self):
        if self.next_option is not None:
            return 1 + self.next_option.get_depth()
        return 1

    def set_debug(self, debug):
        self.debug=debug
        self.next_option.set_debug(debug)

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

    def zero_below_grads(self, top=False):
        self.next_option.zero_below_grads(False)
        if not top: self.policy.zero_grads()

    def toggle_test(self, test):
        if test:
            # self.set_epsilon(self.test_epsilon)
            # self.set_epsilon(0.5)
            if self.zero_epsilon_test: self.zero_epsilon()
            self.sampler, self.test_sampler = self.test_sampler, self.sampler # use test_sampler to keep train sampler
            self.reset_timer = self.terminate_reward.compute_done.timer
        else:
            self.set_epsilon(self.train_epsilon)
            self.sampler, self.test_sampler = self.test_sampler, self.sampler
            self.terminate_reward.compute_done.timer = self.reset_timer

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

    def match_temporal_extension(self):
        if self.next_option is not None and hasattr(self.next_option.terminate_reward, "compute_done"): self.next_option.terminate_reward.compute_done.time_cutoff = self.temporal_extension_manager.ext_cutoff

    def _set_next_option(self, batch, mapped_act):
        start = time.time()
        mask, param, obs = batch.mask, batch.param, batch.obs
        batch["mask"] = [self.next_option.sampler.mask.active_mask]
        batch["param"] = self.state_extractor.expand_param(mapped_act, self.next_option.sampler.mask.active_mask)
        batch['obs'] = self.next_option.state_extractor.get_obs(batch["last_full_state"], batch["full_state"], batch["param"], batch["mask"])
        # if self.name == "Block":
        #     print("deciding action",self.name, batch["param"], batch["full_state"]["factored_state"]["Gripper"], self.next_option.state_extractor.reverse_obs_norm(batch['obs'], mask=batch['mask']))
        # print("setting", time.time() - start)
        return batch, mask, param, obs

    def _reset_current_option(self, batch, mask, param, obs):
        start = time.time()
        batch.update(mask=mask, param=param,obs=obs)
        # batch["mask"] = mask
        # batch["param"] = param
        # batch['obs'] = obs
        # print("setting", time.time() - start)
        return batch


    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False, force=None, action=None):
        '''
        get a new action (resample) or not based on the result of TEM.check. If we don't, check downstream options
        batch must contain full_state and termination_chain
        '''
        needs_sample, act, chain, policy_batch, state, masks = self.temporal_extension_manager.check(term_chain[-1], ext_terms[-1])
        if needs_sample: result_tuple = self.sample_action_chain(batch, state_chain, random=random, action=action)
        else: # if we don't need a new sample
            batch, mask, param, obs = self._set_next_option(batch, chain[-1])
            new_act, rem_chain, pol_batch, rem_state, rem_masks, last_resmp = self.next_option.extended_action_sample(batch, state_chain, term_chain[:-1], ext_terms[:-1], random=False, use_model=use_model)
            result_tuple = (act, rem_chain + [chain[-1]], policy_batch, rem_state + [state[-1]] if state is not None else None, rem_masks + [masks[0]])
            batch = self._reset_current_option(batch, mask, param, obs)
        return (*result_tuple, needs_sample)

    def sample_action_chain(self, batch, state_chain, random=False, action=None): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
        use_model is only for model based search
        if the batch object contains a partial flag (key with PARTIAL=1), then treat the state as a partial
        @param action is a mapped action which forces the param to be that action.
        '''
        # if self.name == "Ball": print(self.name, batch.obs)
        start = time.time()
        if action is not None:
            act = self.action_map.reverse_map_action(action, batch[0])
            mapped_act = self.action_map.map_action(act, batch[0])
            state, policy_batch = None, Batch()
        elif random:
            act = self.action_map.sample_policy_space()
            mapped_act = self.action_map.map_action(act, batch[0])
            state, policy_batch = None, Batch()
        else:
            policy_batch = self.policy.forward(batch, state_chain[-1] if state_chain is not None else None)
            state = policy_batch.state
            # print("forward", time.time() - start)
            act, mapped_act = pytorch_model.unwrap(policy_batch.sampled_act[0]), self.action_map.map_action(policy_batch.sampled_act[0], batch[0])
        # if self.name == "Block": print("actions", act, mapped_act)
        chain = [mapped_act.squeeze()]
        # recursively propagate action up the chain
        # print("compute", self.name, time.time() - start)
        compute = time.time()
        if self.next_option is not None:
            next_batch, cur_mask, cur_param, cur_obs = self._set_next_option(batch, mapped_act)
            if self.debug:
                self.debug = copy.deepcopy(next_batch)
            next_policy_act, rem_chain, result, rem_state_chain, last_masks = self.next_option.sample_action_chain(next_batch, state_chain[-1] if state_chain is not None else None) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain, state, masks = rem_chain + chain, rem_state_chain + [state], last_masks + [cur_mask] # TODO: mask is not set from the policy
            batch = self._reset_current_option(batch, cur_mask, cur_param, cur_obs)
        # print("next", self.name, time.time() - compute)
        # print("total", self.name, time.time() - start)
        return act, chain, policy_batch, state, masks

    def reset(self, full_state):
        # reset the timers for temporal extension, termination
        init_terms = self.next_option.reset(full_state)
        self.temporal_extension_manager.reset()
        self.terminate_reward.reset()
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
            next_param = self.state_extractor.expand_param(next_param, self.next_option.sampler.mask.active_mask) # TODO: shouldn't this be next_mask?
            last_done, last_rewards, last_termination, _, _ = self.next_option.terminate_reward_chain(full_state, next_full_state, next_param, chain[:len(chain)-1], next_mask, mask_chain[:len(mask_chain)-1], true_done=true_done, true_reward=true_reward)
            # print(self.name, last_done, last_rewards, last_termination)
        # slightly confusing: inter_state is normalized, but next_target is not, because of how goal state is handled
        inter_state, target, next_target, target_diff = self.state_extractor.get_inter(full_state), self.state_extractor.get_target(full_state), self.state_extractor.get_target(next_full_state), self.state_extractor.get_diff(full_state, next_full_state)
        term, reward, done, inter, time_cutoff = self.terminate_reward(inter_state, target, next_target, target_diff, param, mask, true_done, true_reward) # TODO: true_inter = ?
        # print(self.name, term, reward, done, inter, time_cutoff, self.terminate_reward)
        rewards, terminations = last_rewards + [reward], last_termination + [term]
        return done, rewards, terminations, inter, time_cutoff

    def assign_interaction_model(self, interaction_model):
        inter_model = self.interaction_model
        if interaction_model is not None: 
            self.sampler.mask = interaction_model.mask # reassigns the mask in case it changed
            self.sampler.target_selector = interaction_model.target_select
            self.sampler.additional_selector = interaction_model.additional_selectors[-1] if len(interaction_model.additional_selectors) > 0 else interaction_model.additional_select
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