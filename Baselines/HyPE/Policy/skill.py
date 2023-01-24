import numpy as np
import os, cv2, time, copy
import torch
import gym
from Network.network_utils import pytorch_model
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

def load_skill(load_dir, name, device=-1):
    load_name = os.path.join(load_dir, name + "_skill.opt")
    try: 
        skill = torch.load(load_name)
    except FileNotFoundError as e:
        return None
    if device != -1:
        skill.cuda(device=device)
    return skill

class Extractor():
    def __init__(self, causal_extractor, norm, obs_components, input_scaling, normalized):
        self.extractor = causal_extractor
        self.norm = norm
        self.obs_components = obs_components
        self.input_scaling = input_scaling
        self.normalized = normalized

    def get_target(self, full_state):
        return self.extractor.target_selector(full_state['factored_state'])

    def get_parent(self, full_state):
        return self.extractor.parent_selector(full_state['factored_state'])

    def get_diff(self, full_state, next_full_state):
        return self.extractor.target_selector(next_full_state['factored_state']) - self.extractor.target_selector(full_state['factored_state'])

    def get_obs(self, full_state):
        # print(full_state["factored_state"])
        parent = self.extractor.parent_selector(full_state['factored_state'])
        target = self.extractor.target_selector(full_state['factored_state'])
        rel = parent - target
        # print(parent.shape, target.shape, rel.shape)
        if not hasattr(self, "normalized") or self.normalized: components = [self.norm(parent, form="parent"), self.norm(target), self.norm(rel, form='rel')]
        else: components = [parent, target, rel]
        cat = [c for (c,oc) in zip(components, self.obs_components) if oc == 1]
        # print(cat, self.obs_components)
        if hasattr(self, "input_scaling"):
            return np.concatenate(cat, axis=-1) * self.input_scaling
        else: return np.concatenate(cat, axis=-1)

class Skill():
    def __init__(self, args, policies, models, next_option, debug=False):
        # parameters for saving
        self.name = args.object_names.target
        self.train_epsilon = args.skill.epsilon_random

        # primary models
        self.policies = policies # one policy for each mode
        self.policy_index = 0
        self.policy = self.first_policy() # policy to run during opion
        self.next_option = next_option # the option which controls the actions
        self.next_option_name = next_option.name
        self.next_option.zero_epsilon()

        self.action_space = self.policy.action_space
        self.temporal_extension_manager = models.temporal_extension_manager # manages when to temporally extend
        self.extractor = models.extractor # extracts the desired state
        self.reward_model = models.reward_model # handles termination, reward and temporal extension termination
        self.num_skills = self.reward_model.num_modes # the number of distinct targets
        self.sample_queue = list()
        self.reset_timer = 0 # storage for the temporal extension timer
        self.policy_timer = 0 # on-policy 
        self.policy_iters = args.skill.policy_iters
        self.assignment_mode = False
        self.move_policies = True
        if args.torch.cuda: self.cuda(args.torch.gpu)
        else:
            self.device = 'cpu'
            self.cpu()
        self.set_epsilon(args.skill.epsilon_random)

    def toggle_test(self, on):
        if on:
            self.zero_epsilon()
            self.policy.use_best(True)
        else:
            self.set_epsilon(self.train_epsilon)
            self.policy.use_best(False)

    def toggle_assignment_mode(self):
        self.assignment_mode = not self.assignment_mode

    def set_policy(self, index):
        self.policy_index = index
        self.policy = self.policies[index]
        self.policy.next_policy()
        return self.policy

    def first_policy(self):
        self.policy_index = 0
        self.policy = self.policies[0]
        self.policy.first_policy()
        return self.policy

    def next_policy(self):
        self.policy_index = (self.policy_index + 1) % len(self.policies)
        self.policy = self.policies[self.policy_index]
        self.policy.next_policy()
        return self.policy

    def get_network_index(self):
        return self.policy.get_network_index()

    def sample(self):
        return self.action_space.sample()

    def sample_first(self, reset_policy=False):
        self.sample_queue = np.arange(self.num_skills)
        np.random.shuffle(self.sample_queue)
        self.sample_queue = self.sample_queue.tolist()
        if reset_policy:
            print("policy_reset")
            for policy in self.policies:
                policy.neg_policy()
        return self.sample_queue[0]

    def sample_next(self):
        if len(self.sample_queue) == 0:
            self.sample_first()
        print(self.sample_queue)
        return self.sample_queue.pop(0)

    def get_depth(self):
        if self.next_option is not None:
            return 1 + self.next_option.get_depth()
        return 1

    def set_device(self, device_no):
        self.device = 'cpu' if not self.iscuda else 'cuda:' + str(device_no)
        if self.policies is not None:
            for policy in self.policies:
                policy.to(self.devicedevice)
                policy.assign_device(self.device)
        if self.next_option is not None:
            self.next_option.set_device(device_no)

    def cuda(self, device=None):
        self.iscuda = True
        if device is not None: self.device=device
        if self.policies is not None: 
            for policy in self.policies:
                policy.cuda(device = device)
        if self.next_option is not None: self.next_option.cuda(device=device)

    def cpu(self): # does NOT set device
        self.iscuda = False
        if self.policies is not None: 
            for policy in self.policies:
                policy.cpu()
        if self.next_option is not None: self.next_option.cpu()

    def zero_below_grads(self, top=False):
        self.next_option.zero_below_grads(False)
        if not top: self.policy.zero_grads()

    # def toggle_test(self, test):
    #     if test:
    #         # self.set_epsilon(self.test_epsilon)
    #         # self.set_epsilon(0.5)
    #         self.zero_epsilon()
    #         self.reset_timer = self.terminate_reward.compute_done.timer
    #     else:
    #         self.set_epsilon(self.train_epsilon)
    #         self.terminate_reward.compute_done.timer = self.reset_timer

    def zero_epsilon(self):
        if self.policy is not None:
            for policy in self.policies:
                policy.set_eps(0.0)
        if self.next_option is not None:
            self.next_option.zero_epsilon()
        return self.train_epsilon

    def set_epsilon(self, eps):
        self.train_epsilon = eps
        if self.policy is not None:
            for policy in self.policies:
                policy.set_eps(eps)

    def _set_next_option(self, batch, action):
        start = time.time()
        param, obs = batch.assignment, batch.obs
        batch["assignment"] = action
        batch['obs'] = self.next_option.extractor.get_obs(batch["full_state"])
        # if self.name == "Block":
        #     print("deciding action",self.name, batch["param"], batch["full_state"]["factored_state"]["Gripper"], self.next_option.state_extractor.reverse_obs_norm(batch['obs'], mask=batch['mask']))
        # print("setting", time.time() - start)
        return batch, param, obs

    def extended_action_sample(self, batch, state_chain, term, ext_terms, random=False, force=None, action=None):
        '''
        get a new action (resample) or not based on the result of TEM.check. If we don't, check downstream options
        batch must contain full_state and termination_chain
        '''
        needs_sample, act, chain, policy_batch, state = self.temporal_extension_manager.check(term, ext_terms[-1])
        # print(needs_sample, act)
        if needs_sample: result_tuple = self.sample_action_chain(batch, state_chain, random=random, action=action)
        else: # if we don't need a new sample
            batch, param, obs = self._set_next_option(batch, chain[-1])
            new_act, rem_chain, pol_batch, rem_state, last_resmp = self.next_option.extended_action_sample(batch, state_chain, ext_terms[-1], ext_terms[:-1], random=random)
            result_tuple = (act, rem_chain + [chain[-1]], policy_batch, rem_state + [state[-1]] if state is not None else None)
            batch.update(assignment=param,obs=obs)
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
            act = action
            state, policy_batch = None, Batch()
        elif random:
            act = self.sample_action(random=True)
            state, policy_batch = None, Batch()
        else:
            if self.assignment_mode: policy = self.policies[int(batch.assignment.squeeze())]
            else: policy = self.policy
            # print(self.train_epsilon, self.policy.epsilon)
            policy_batch = policy.forward(batch, state_chain[-1] if state_chain is not None else None)
            state = policy_batch.state
            act = pytorch_model.unwrap(policy_batch.act[0])
            # print(self.name, policy_batch.logits, policy_batch.act)
        chain = [act.squeeze()]
        # recursively propagate action up the chain
        compute = time.time()
        if self.next_option is not None:
            next_batch, cur_param, cur_obs = self._set_next_option(batch, act)
            next_policy_act, rem_chain, result, rem_state_chain = self.next_option.sample_action_chain(next_batch, state_chain[-1] if state_chain is not None else None) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain, state = rem_chain + chain, rem_state_chain + [state] # TODO: mask is not set from the policy
            batch.update(param=cur_param,obs=cur_obs)
        return act, chain, policy_batch, state

    def reset(self, full_state, policy_iters=-1):
        # reset the timers for temporal extension, termination
        init_terms = self.next_option.reset(full_state)
        self.temporal_extension_manager.reset()
        self.policy_timer = 0
        self.first_policy()
        if policy_iters > 0: self.policy_iters = policy_iters
        return init_terms + [False]

    def update(self, act, chain, te_chain, update_policy=True):
        # updates internal states of the option, and asssociated components
        if self.next_option is not None:
            self.next_option.update(chain[len(chain)-1], chain[:len(chain)-1], te_chain[:len(te_chain)-1], update_policy = False)
        self.temporal_extension_manager.update(act, chain, te_chain[-1])
        # if update_policy:
        #     for policy in self.policies:
        #         policy.update_epsilon()
            # self.policy_timer += 1
            # if self.policy_timer == self.policy_iters:
            #     self.policy.next_policy()
            #     self.policy_timer = 0
            #     self.new_policy = True
            # else:
            #     self.new_policy = False

    def terminate_chain(self, full_states, true_done=False, first=False):
        # full states is the last k states, plus the next full state
        # returns the termination chain AFTER the current termination
        # recursively get all of the dones and rewards
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            last_terminations = self.next_option.terminate_chain(full_states, true_done=true_done, first=False)
        if first:
            return last_terminations
        else:
            parent, target, target_diff = self.extractor.get_parent(full_states)[1:], self.extractor.get_target(full_states)[1:], self.extractor.get_diff(full_states[:-1], full_states[1:])
            _, terminations = self.reward_model.compute_reward(target_diff, target, parent, true_done) # TODO: true_inter = ?
            cutoff = self.temporal_extension_manager.is_cutoff()
            terminations = last_terminations + [terminations[-1] or cutoff]
        return terminations

    def save(self, save_dir):
        # checks and prepares for saving option as a pickle
        save_name = os.path.join(save_dir, self.name + "_skill.opt") # TODO: have separate save/load for terminate_reward
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            inter_model = None
            self.cpu()
            torch.save(self, save_name)
            self.cuda(device=self.device)
            return self
        return None