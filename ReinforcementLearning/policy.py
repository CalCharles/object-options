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

from ReinforcementLearning.ts.utils import init_networks, init_algorithm, _assign_device, reassign_optim, reset_params
from State.object_dict import ObjDict
from Network.network_utils import reset_parameters, count_layers



class Policy(nn.Module):
    '''
    wraps around a TianShao Base policy, but supports most of the same functions to interface with Option.option.py
    Note that TianShao Policies also support learning

    '''
    def __init__(self, discrete_actions, input_shape, policy_action_space, args, preset_policy=None):
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
        self.algo_name = args.policy.learning_type # the algorithm being used
        self.MIN_HER = 1000
        
        if preset_policy is None:
            # initialize networks
            nets_optims = init_networks(args, input_shape, policy_action_space.n if discrete_actions else policy_action_space.shape[0], discrete_actions)
            self.critic_lr, self.actor_lr = args.critic_net.optimizer.lr, args.actor_net.optimizer.lr

            # initialize tianshou lower level
            self.algo_policy = init_algorithm(args, nets_optims, policy_action_space, discrete_actions)
        else:
            self.algo_policy = preset_policy.algo_policy
            self.critic_lr, self.actor_lr = preset_policy.critic_lr, preset_policy.actor_lr

        # intialize epsilon values for randomness
        self.epsilon_schedule = args.policy.epsilon_schedule
        self.epsilon = 1 if self.epsilon_schedule > 0 else args.policy.epsilon_random
        self.epsilon_schedule = args.policy.epsilon_schedule # if > 0, adjusts epsilon from 1->args.epsilon by exp(-steps/epsilon schedule)
        self.epsilon_timer = 0 # timer to record steps
        self.epsilon_base = args.policy.epsilon_schedule
        self.set_eps(self.epsilon)

        # other parameters
        self.discrete_actions = discrete_actions
        self.grad_epoch = args.policy.learn.grad_epoch
        self.select_positive = args.hindsight.select_positive
        self.use_her = args.hindsight.use_her
        self.sample_form = args.policy.learn.sample_form
        self.device = args.torch.gpu
        # Primacy bias parameters
        self.init_form = args.network.init_form
        self.reset_layers = args.policy.primacy.reset_layers
        reset_params(0, self.algo_policy, self.discrete_actions, self.init_form, self.algo_name) # resets all the parameters first (including the TS ones)

    def zero_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def cpu(self):
        super().cpu()
        self.device = "cpu"
        _assign_device(self.algo_policy, self.algo_name, self.discrete_actions, "cpu")
        reassign_optim(self.algo_policy, self.critic_lr, self.actor_lr)

    def cuda(self, device=None, args=None):
        super().cuda()
        if device is not None:
            self.device=device
            _assign_device(self.algo_policy, self.algo_name, self.discrete_actions, device)
            reassign_optim(self.algo_policy, self.critic_lr, self.actor_lr)

    def reset_select_params(self):
        '''
        resets the parameters of the model, last layers are counted as 1
        '''
        reset_params(self.reset_layers, self.algo_policy, self.discrete_actions, self.init_form, self.algo_name)


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
        if self.algo_name in ['sac']:
            if self.discrete_actions: Q_val = self.algo_policy.critic1(comp)
            else: Q_val = self.algo_policy.critic1(comp, batch.act)
        if self.algo_name in ['ddpg']:
            Q_val = self.algo_policy.critic(comp, batch.act)
        if self.algo_name in ['dqn', 'rainbow']:
            Q_val = self.algo_policy(batch, input="obs_next" if nxt else "obs").logits
        if self.algo_name in ['rainbow']:
            Q_val = self.algo_policy.compute_q_value(Q_val, None)
        return Q_val

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs", **kwargs: Any):
        '''
        Matches the call for the forward of another algorithm method. Calls 
        '''
        batch = copy.deepcopy(batch) # make sure input norm does not alter the input batch
        vals = self.algo_policy(batch, state = state, input=input, **kwargs)
        return vals

    def update_epsilon(self):
        self.epsilon_timer += 1
        if self.epsilon_schedule > 0 and self.epsilon_timer % self.epsilon_schedule == 0: # only updates every epsilon_schedule time steps
            self.epsilon = self.epsilon_base + (1-self.epsilon_base) * np.exp(-max(0, self.epsilon_timer - self.pretrain_iters)/self.epsilon_schedule) 
            self.set_eps(self.epsilon)

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], her_buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        '''
        don't call the algo_policy update, but carries almost the same logic
        however, inserting the param needs to be handled.
        '''
        for i in range(self.grad_epoch):
            use_buffer = buffer
            if self.sample_form == "HER":
                if len(her_buffer) < self.MIN_HER: # nothing to sample 
                    return {}
                her_batch, indice = her_buffer.sample(sample_size)
                batch = self.algo_policy.process_fn(her_batch, her_buffer, indice)
            elif self.sample_form == "merged" and len(her_buffer) > self.MIN_HER:
                if buffer is None or her_buffer is None:
                    return {}
                self.algo_policy.updating = True

                # sample from the main buffer and assign returns
                main_batch, main_indice = buffer.sample(int(np.round(sample_size * (1-self.select_positive))))
                main_batch = self.algo_policy.process_fn(main_batch, buffer, main_indice)

                # sample from the HER buffer and assign returns
                her_batch, her_indice = her_buffer.sample(int(np.round(sample_size * self.select_positive)))
                her_batch = self.algo_policy.process_fn(her_batch, her_buffer, her_indice)
                
                batch = main_batch
                batch.cat_([her_batch])
                indice = np.concatenate([main_indice, her_indice]) 
            else:
                use_buffer = her_buffer if np.random.rand() < self.select_positive and self.use_her and len(her_buffer) > self.MIN_HER else buffer
                batch, indice = use_buffer.sample(sample_size)
                batch = self.algo_policy.process_fn(batch, use_buffer, indice)

            kwargs["batch_size"] = sample_size
            kwargs["repeat"] = 2
            result = self.algo_policy.learn(batch, **kwargs)
            if i == 0: cumul_losses = result
            else: 
                for k in result.keys():
                    cumul_losses[k] += result[k] 
            if self.sample_form == "merged" and len(her_buffer) > self.MIN_HER:
                if "weight" in batch: main_batch.weight, her_batch.weight = batch.weight[:int(np.round(sample_size * (1-self.select_positive)))], batch.weight[int(np.round(sample_size * (1-self.select_positive))):] # assign weight values for prioritized buffer
                self.algo_policy.post_process_fn(main_batch, buffer, main_indice)
                self.algo_policy.post_process_fn(her_batch, her_buffer, her_indice)
            else:
                self.algo_policy.post_process_fn(batch, use_buffer, indice)
            self.algo_policy.updating = False
        return {k: cumul_losses[k] / self.grad_epoch for k in cumul_losses.keys()}



# policy_forms = {"basic": BasicPolicy, "image": ImagePolicy, 'grid': GridWorldPolicy, 'actorcritic': BasicActorCriticPolicy}
