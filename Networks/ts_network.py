import torch
import torch.nn as nn
import numpy as np
from Networks.network import pytorch_model, BasicMLPNetwork, PointNetwork, PairNetwork
import torch.nn.functional as F

class TSNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.iscuda = kwargs["cuda"]
        self.output_dim = int(np.prod(kwargs["num_outputs"]))
        self.use_input_norm = False # set these to true
        self.continuous_critic = False if "continuous_critic" not in kwargs else kwargs["continuous_critic"]
        self.action_dim = 0 if "action_dim" not in kwargs else kwargs["action_dim"]
        self.bound_output = kwargs["bound_output"] # if 0, then not used, otherwise, bound the output to [-bound_output, bound_output]

    def cuda(self):
        super().cuda()
        self.iscuda = True

    def cpu(self):
        super().cpu()
        self.iscuda = False

    def forward(self, obs, state=None, info={}):
        # requires normalized obs: np.ndarray
        # state is the additional state information, such as recurrence
        if self.continuous_critic and self.action_dim > 0: # the action values need to be at the front for pointnet type networks to work properly
            obs = torch.cat([obs[...,-self.action_dim:], obs[...,:obs.shape[-1] - self.action_dim]], dim=-1) 
        if not isinstance(obs, torch.Tensor):
            obs = pytorch_model.wrap(obs, dtype=torch.float, cuda=self.iscuda)
        logits = self.model(obs.reshape(obs.shape[0], -1))
        if self.bound_output != 0:
            logits = torch.tanh(logits) * self.bound_output
        return logits, state


class BasicNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["num_outputs"] = self.output_dim
        self.model = BasicMLPNetwork(**kwargs)
        if self.iscuda:
            self.cuda()

class PairPolicyNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["aggregate_final"] = True
        kwargs["num_outputs"] = self.output_dim
        self.model = PairNetwork(**kwargs)
        if self.iscuda:
            self.cuda()

class RainbowNetwork(TSNet):
    """
    Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
    In particular, this network requires num_atoms handling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs['policy_type'] == "pair":
            kwargs["aggregate_final"] = True
            kwargs["num_outputs"] = kwargs['hidden_sizes'][-1]
            self.model = PairNetwork(**kwargs)
        if kwargs['policy_type'] == "basic":
            kwargs["num_outputs"] = kwargs['hidden_sizes'][-1]
            self.model = BasicMLPNetwork(**kwargs)
        kwargs["num_inputs"] = kwargs['hidden_sizes'][-1]
        self.num_atoms = kwargs['num_atoms']
        kwargs['hidden_sizes'] = [512]
        kwargs["num_outputs"] = self.num_atoms * self.output_dim
        self._is_dueling = kwargs['is_dueling']
        self.Q = BasicMLPNetwork(**kwargs)
        if self._is_dueling:
            kwargs["num_outputs"] = self.num_atoms
            self.V = BasicMLPNetwork(**kwargs)


    def forward(self, obs, state=None, info={}):
        # TODO: make this not hardcoded
        obs = self.input_norm(obs)

        if self.continuous_critic and self.action_dim > 0: # the action values need to be at the front for pointnet type networks to work properly
            obs = torch.cat([obs[...,-self.action_dim:], obs[...,:obs.shape[-1] - self.action_dim]], dim=-1) 
        if not isinstance(obs, torch.Tensor):
            obs = pytorch_model.wrap(obs, dtype=torch.float, cuda=self.iscuda)
        batch = obs.shape[0]
        obs = obs.reshape(batch, -1)
        hidden = self.model(obs)
        q = self.Q(hidden)
        q = q.view(-1, self.output_dim, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2) # not certain how compatible this is
        return probs, state

networks = {'basic': BasicNetwork, 'pixel': PixelNetwork, 'grid': GridWorldNetwork, 'pair': PairPolicyNetwork}