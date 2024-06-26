import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Network.network_utils import pytorch_model
from Network.network import Network, network_type
from Network.General.mlp import MLPNetwork
from Network.General.point import PointNetwork
from Network.General.pair import PairNetwork
from State.object_dict import ObjDict

class TSNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.iscuda = kwargs["cuda"]
        self.output_dim = int(np.prod(kwargs["num_outputs"]))
        self.use_input_norm = False # set these to true
        self.continuous_critic = False if "continuous_critic" not in kwargs else kwargs["continuous_critic"]
        self.action_dim = 0 if "action_dim" not in kwargs else kwargs["action_dim"]
        self.scale_logits = kwargs["scale_logits"] if "scale_logits" in kwargs else -1

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
        # print(logits.shape, self.scale_logits)
        if self.scale_logits > 0: logits = logits * self.scale_logits
        return logits, state

class BasicNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs = ObjDict(kwargs)
        kwargs.pair.aggregate_final = True
        kwargs["num_outputs"] = self.output_dim
        self.model = network_type[kwargs.net_type](kwargs)
        if self.iscuda:
            self.cuda()

class RainbowNetwork(TSNet):
    """
    Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.
    In particular, this network requires num_atoms handling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs = ObjDict(kwargs)
        last_size = kwargs['hidden_sizes'][-1]
        kwargs['pair']["aggregate_final"] = True
        kwargs['hidden_sizes'] = kwargs['hidden_sizes'][:-1]
        kwargs["num_outputs"] = last_size
        self.model = network_type[kwargs.net_type](kwargs)
        kwargs["num_inputs"] = last_size
        self.num_atoms = kwargs['num_atoms']
        kwargs['hidden_sizes'] = [256]
        kwargs["num_outputs"] = self.num_atoms * self.output_dim
        self._is_dueling = kwargs['is_dueling']
        self.Q = MLPNetwork(kwargs)
        if self._is_dueling:
            kwargs["num_outputs"] = self.num_atoms
            self.V = MLPNetwork(kwargs)


    def forward(self, obs, state=None, info={}):
        # TODO: make this not hardcoded

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
            v = self.V(hidden)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2) # not certain how compatible this is
        return probs, state

networks = {'basic': BasicNetwork, 'rainbow': RainbowNetwork}