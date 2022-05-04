import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import copy, os, cv2
from file_management import default_value_arg
from Networks.network import Network, pytorch_model
from Networks.tianshou_networks import networks, RainbowNetwork
from Networks.critic import BoundedDiscreteCritic, BoundedContinuousCritic
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
cActor, cCritic = Actor, Critic
from tianshou.utils.net.discrete import Actor, Critic
dActor, dCritic = Actor, Critic
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
import tianshou as ts
import gym
from typing import Any, Dict, Tuple, Union, Optional, Callable
from ReinforcementLearning.LearningAlgorithm.iterated_supervised_learner import IteratedSupervisedPolicy
from ReinforcementLearning.LearningAlgorithm.HER import HER
from Rollouts.rollouts import ObjDict


_actor_critic = ['ddpg', 'sac']
_double_critic = ['sac']

# TODO: redo this one
class TSPolicy(nn.Module):
    '''
    wraps around a TianShao Base policy, but supports most of the same functions to interface with Option.option.py
    Note that TianShao Policies also support learning

    '''
    def __init__(self, discrete_actions, input_shape, action_shape, args):
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
        self.algo_name = args.learning_type # the algorithm being used
        
        # initialize networks
        nets_optims = init_networks(args, input_shape, action_shape, discrete_actions)

        # initialize tianshou lower level
        self.algo_policy = init_algorithm(args, nets_optims)

        # intialize epsilon values for randomness
        self.epsilon = 1 if self.epsilon_schedule > 0 else args.epsilon
        self.epsilon_schedule = args.epsilon_schedule # if > 0, adjusts epsilon from 1->args.epsilon by exp(-steps/epsilon schedule)
        self.epsilon_timer = 0 # timer to record steps
        self.epsilon_base = args.epsilon
        self.set_eps(self.epsilon)

        # other parameters
        self.discrete_actions = discrete_actions
        self.grad_epoch = args.grad_epoch
        self.sample_form = args.sample_form

    def cpu(self):
        super().cpu()
        assign_device("cpu")
        reassign_optim(self.algo_policy)

    def cuda(self, device=None, args=None, critic_lr=1e-5, actor_lr=1e-5):
        super().cuda()
        if device is not None:
            assign_device(self.algo_policy, device)
            reassign_optim(self.algo_policy)

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
            Q_val = self.algo_policy.compute_q_value(probs, None)
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
                if len(her_buffer) < 1000: # nothing to sample 
                    return {}
                her_batch, indice = her_buffer.sample(sample_size)
                batch = self.algo_policy.process_fn(her_batch, her_buffer, indice)
            elif self.sample_form == "merged" and len(self.learning_algorithm.replay_buffer) > 1000:
                if buffer is None or her_buffer is None:
                    return {}
                self.algo_policy.updating = True

                # sample from the main buffer and assign returns
                main_batch, main_indice = buffer.sample(int(np.round((1-sample_size) * self.select_positive)))
                main_batch = self.algo_policy.process_fn(main_batch, buffer, main_indice)

                # sample from the HER buffer and assign returns
                her_batch, her_indice = her_buffer.sample(int(np.round(sample_size * self.select_positive)))
                her_batch = self.algo_policy.process_fn(her_batch, her_buffer, her_indice)
                
                batch = main_batch
                batch.cat_([main_batch])
            else:
                use_buffer = her_buffer if np.random.rand() < self.select_positive and self.use_her else buffer
                batch, indice = use_buffer.sample(sample_size)
                batch = self.algo_policy.process_fn(batch, use_buffer, indice)

            kwargs["batch_size"] = sample_size
            kwargs["repeat"] = 2
            result = self.algo_policy.learn(batch, **kwargs)
            if i == 0: cumul_losses = result
            else: 
                for k in result.keys():
                    cumul_losses[k] += result[k] 
            if self.sample_merged and len(self.learning_algorithm.replay_buffer) > 1000:
                self.algo_policy.post_process_fn(main_batch, buffer, main_indice)
                self.algo_policy.post_process_fn(her_batch, her_buffer, her_indice)
            else:
                self.algo_policy.post_process_fn(batch, use_buffer, indice)
            self.algo_policy.updating = False
        return {k: cumul_losses[k] / self.grad_epoch for k in cumul_losses.keys()}



# policy_forms = {"basic": BasicPolicy, "image": ImagePolicy, 'grid': GridWorldPolicy, 'actorcritic': BasicActorCriticPolicy}
