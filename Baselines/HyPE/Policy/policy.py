import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import copy, os, cv2
import tianshou as ts
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
import gym
from typing import Any, Dict, Tuple, Union, Optional, Callable

from Baselines.HyPE.Policy.utils import init_networks, init_algorithm, _assign_device, reassign_optim, reset_params
from gym import spaces
from State.object_dict import ObjDict
from Network.network_utils import reset_parameters, count_layers



class Policy(nn.Module):
    '''
    wraps around a TianShao Base policy, but supports most of the same functions to interface with Option.option.py
    Note that TianShao Policies also support learning

    '''
    def __init__(self, num_actions, input_shape, args, pair_args):
        '''
        @param input shape is the shape of the input to the network
        @param paction space is a gym.space corresponding to the ENVIRONMENT action space
        @param action space is the action space of the agent
        @param max action is the maximum action for continuous
        @param discrete actions is when the action space is discrete
        kwargs includes:
            learning type, lookahead, input norm, sample merged, option, epsilon_schedule, epsilon, object dim, first_obj_dim, parameterized, grad epoch
            network args: max critic, cuda, policy type, gpu (device), hidden sizes, actor_lr, critic_lr, aggregate final, post dim, 
        '''
        super().__init__()
        self.algo_name = args.skill.learning_type # the algorithm being used
        self.MIN_HER = 1000
        
        # initialize networks
        nets_optims = init_networks(args, input_shape, num_actions, pair_args)
        self.critic_lr, self.actor_lr = args.critic_net.optimizer.lr, args.actor_net.optimizer.lr

        # initialize tianshou lower level
        self.action_space = spaces.Discrete(num_actions)
        self.algo_policy = init_algorithm(args, nets_optims, self.action_space)

        # intialize epsilon values for randomness
        # self.epsilon_schedule = args.skill.epsilon_schedule # if > 0, adjusts epsilon from 1->args.epsilon by exp(-steps/epsilon schedule)
        self.epsilon = args.skill.epsilon_random
        self.epsilon_base = args.skill.epsilon_random
        self.set_eps(self.epsilon)

        # other parameters
        self.learning_type = args.skill.learning_type
        self.grad_epoch = args.skill.learn.grad_epoch
        self.device = args.torch.gpu
        # Primacy bias parameters
        self.init_form = args.network.init_form
        reset_params(0, self.algo_policy, self.init_form, self.learning_type) # resets all the parameters first (including the TS ones)

    def use_best(self, use_best):
        self.algo_policy.using_best = use_best

    def neg_policy(self):
        if self.learning_type == 'cmaes': self.algo_policy.neg_policy()

    def first_policy(self):
        if self.learning_type == 'cmaes': self.algo_policy.first_policy()

    def next_policy(self):
        if self.learning_type == 'cmaes': self.algo_policy.next_policy()     

    def get_network_index(self):
        if self.learning_type == 'cmaes': return self.algo_policy.policy_index
        return 0

    def zero_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def cpu(self):
        super().cpu()
        self.device = "cpu"
        _assign_device(self.algo_policy, self.algo_name, True, "cpu")
        if self.learning_type != 'cmaes': reassign_optim(self.algo_policy, self.critic_lr, self.actor_lr)

    def cuda(self, device=None, args=None):
        super().cuda()
        if device is not None:
            self.device=device
            _assign_device(self.algo_policy, self.algo_name, True, device)
            if self.learning_type != 'cmaes': reassign_optim(self.algo_policy, self.critic_lr, self.actor_lr)

    def set_eps(self, epsilon): # not all algo policies have set eps
        self.epsilon = epsilon
        if hasattr(self.algo_policy, "set_eps"):
            self.algo_policy.set_eps(epsilon)
        if hasattr(self.algo_policy, "set_exp_noise"):
            self.algo_policy.set_exp_noise(GaussianNoise(sigma=epsilon))

    def save(self, pth, name):
        torch.save(self, os.path.join(pth, name + ".pt"))

    def compute_Q(
        self, batch: Batch, nxt: bool
    ) -> torch.Tensor:
        comp = batch.obs_next if nxt else batch.obs
        if self.algo_name in ['dqn', 'rainbow']:
            Q_val = self.algo_policy(batch, input="obs_next" if nxt else "obs").logits
        if self.algo_name in ['rainbow']:
            Q_val = self.algo_policy.compute_q_value(Q_val, None)
        elif self.algo_name in ['cmaes']:
            Q_val = self.algo_policy(batch, input="obs_next" if nxt else "obs").probs
        elif self.algo_name in ['ppo']:
            Q_val = self.algo_policy(batch, input="obs_next" if nxt else "obs").probs
        return Q_val

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs", **kwargs: Any):
        '''
        Matches the call for the forward of another algorithm method. Calls 
        '''
        # batch = copy.deepcopy(batch) # make sure input norm does not alter the input batch
        vals = self.algo_policy(batch, state = state, input=input, **kwargs)
        vals.act = self.algo_policy.exploration_noise(vals.act, batch)
        return vals

    # def update_epsilon(self):
    #     self.epsilon_timer += 1
    #     if self.epsilon_schedule > 0 and self.epsilon_timer % self.epsilon_schedule == 0: # only updates every epsilon_schedule time steps
    #         self.epsilon = self.epsilon_base + (1-self.epsilon_base) * np.exp(-max(0, self.epsilon_timer - self.pretrain_iters)/self.epsilon_schedule) 
    #         self.set_eps(self.epsilon)

    def sample_indices(self, sample_size, buffer, sample_last):
        if sample_last:
            indice = buffer.sample_indices(0)
            indice = indice[-sample_size:]
            batch = buffer[indice]
        else:
            batch, indice = buffer.sample(sample_size)
        batch = self.algo_policy.process_fn(batch, buffer, indice)
        return batch, indice

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        '''
        don't call the algo_policy update, but carries almost the same logic
        however, inserting the param needs to be handled.
        '''
        for i in range(self.grad_epoch):
            use_buffer = buffer
            batch, indice = self.sample_indices(sample_size, buffer, self.learning_type == "cmaes")
            # print("num in update", len(batch), sample_size, indice)
            kwargs["batch_size"] = sample_size
            kwargs["repeat"] = 2
            result = self.algo_policy.learn(batch, **kwargs)
            if i == 0: cumul_losses = result
            else: 
                for k in result.keys():
                    cumul_losses[k] += result[k] 
            self.algo_policy.post_process_fn(batch, use_buffer, indice)
            self.algo_policy.updating = False
        return {k: cumul_losses[k] / self.grad_epoch for k in cumul_losses.keys()}
