# implement a conv network which collapses to a fixed size embedding space used for representation learning
# implement an autoencoding hash network
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.discrete import NoisyLinear
# from Baselines.RIDE.common_nets import Conv

def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        action_shape: Sequence[int],
        input_dim:int, # if num_objects is used, this is the object dim
        hidden_sizes: Sequence[int],
        output_dim: int,
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        num_objects: int= -1, # if num_objects, then output_dim is per object
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        sizes = [input_dim] + hidden_sizes + [output_dim]
        self.net = nn.Sequential(
            *sum([[layer_init(nn.Linear(int(d), int(nd))),    
            nn.ReLU(inplace=True)] for d, nd in zip(sizes[:-1], sizes[1:])] , start = list())
        ) if num_objects <= 0 else nn.Sequential(
            *(sum([[nn.Conv1d(d, nd, 1, bias=True),    
            nn.ReLU(inplace=True)] for d, nd in zip(sizes[:-2], sizes[1:])] , start = list()) + [nn.Flatten()])
        )
        print(self)
        final_dim = num_objects * hidden_sizes[-1] if num_objects > 0 else output_dim
        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(final_dim, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(final_dim, output_dim)),
            )
            self.output_dim = output_dim
        self.num_objects = num_objects

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.num_objects > 0: obs = obs.reshape(obs.shape[0], self.num_objects, -1).transpose(-1,-2)
        return self.net(obs), state

class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        action_shape: Sequence[int],
        input_dim:int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: Union[str, int, torch.device] = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
        num_objects: int=-1,
    ) -> None:
        super().__init__(action_shape, input_dim, hidden_sizes, output_dim, device, features_only=True, num_objects=num_objects)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state


class RIDEModule(nn.Module):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param torch.nn.Module feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param int feature_dim: input dimension of the feature net.
    :param int action_dim: dimension of the action space.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(
        self,
        feature_net: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, torch.device] = "cpu",
        num_objects: int = -1,
        discrete_actions: bool = False,
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        self.num_objects = num_objects
        self.discrete_actions = discrete_actions
    
    def get_embedding(self, s1:  Union[np.ndarray, torch.Tensor]):
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        if self.num_objects > 0: s1 = s1.reshape(s1.shape[0], self.num_objects, -1).transpose(-1,-2)
        phi = self.feature_net(s1)
        return phi

    def forward(
        self, s1: Union[np.ndarray, torch.Tensor],
        act: Union[np.ndarray, torch.Tensor], s2: Union[np.ndarray,
                                                        torch.Tensor], **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: s1, act, s2 -> mse_loss, act_hat."""
        s1 = to_torch(s1, dtype=torch.float32, device=self.device)
        s2 = to_torch(s2, dtype=torch.float32, device=self.device)
        # print("pre_embed", s1.shape, s2.shape, self.feature_net)
        if self.num_objects > 0: s1, s2 = s1.reshape(s1.shape[0], self.num_objects, -1).transpose(-1,-2), s2.reshape(s2.shape[0], self.num_objects, -1).transpose(-1,-2)
        phi1, phi2 = self.feature_net(s1), self.feature_net(s2)
        act = to_torch(act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat([phi1, F.one_hot(act, num_classes=self.action_dim) if self.discrete_actions else act], dim=1)
        )
        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return mse_loss, act_hat, phi1