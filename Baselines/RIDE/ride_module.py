# extends the tianshou policy with the RIDE reward in the process_fn, updating rew with the appropriate intrinsic reward

from typing import Any, Dict, Optional, Union
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy
from Baselines.RIDE.ride_network import RIDEModule


class RIDEPolicy(BasePolicy):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param BasePolicy policy: a base policy to add ICM to.
    :param RIDEModule model: the RIDE model.
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float lr_scale: the scaling factor for ICM learning.
    :param float forward_loss_weight: the weight for forward model loss.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        model: RIDEModule,
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        pseudocount_lambda: float, 
        discrete_actions: bool=False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.model = model
        self.optim = optim
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight
        self.pseudocount_lambda = pseudocount_lambda
        self.pseudo_count_dict = Counter()
        self.total_seen = 0
        self.discrete_actions = discrete_actions


    def train(self, mode: bool = True) -> "ICMPolicy":
        """Set the module in training mode."""
        self.policy.train(mode)
        self.training = mode
        self.model.train(mode)
        return self
    
    def _hash_vector(self, phi):
        binned = torch.sigmoid(phi).clone().detach()
        binned[binned > 0.5] = 1
        binned[binned <= 0.5] = 0
        return binned.cpu().numpy()

    def _add_pseudo_count(self, phi):
        binned = self._hash_vector(phi)
        for state in binned: # for each vector in the batch
            self.pseudo_count_dict[tuple(state)] += 1
    
    def _get_hashes(self, phi):
        bins = self._hash_vector(phi)
        counts = list()
        for b in bins: # unfortunately, we have to iterate to get the hashes
            counts.append(self.pseudo_count_dict[tuple(b)] + self.pseudocount_lambda)
        return torch.as_tensor(np.array(counts).astype(float), device = self.model.device)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data by inner policy.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # BELOW: hacked way to add a state to the visited queue based on the size of the batch
        # the size of the batch is equal to 16 (number of enviroment values)
        if len(batch) == 16:
            phi = self.model.get_embedding(batch.obs)
            self._add_pseudo_count(phi)
        return self.policy.forward(batch, state, **kwargs)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        mse_loss, act_hat, phi = self.model(batch.obs, batch.act, batch.obs_next)
        batch.policy = Batch(orig_rew=batch.rew, act_hat=act_hat, mse_loss=mse_loss, phi=phi)
        mse_loss = mse_loss / self._get_hashes(phi)
        batch.rew += to_numpy(mse_loss * self.reward_scale)
        return self.policy.process_fn(batch, buffer, indices)

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        self.policy.post_process_fn(batch, buffer, indices)
        batch.rew = batch.policy.orig_rew  # restore original reward

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        res = self.policy.learn(batch, **kwargs)
        self.optim.zero_grad()
        act_hat = batch.policy.act_hat
        if self.discrete_actions:
            act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
            inverse_loss = F.cross_entropy(act_hat, act).mean()
        else:
            act = to_torch(batch.act, dtype=torch.float32, device=act_hat.device)
            inverse_loss = 0.5 * F.mse_loss(act_hat, act, reduction="none").sum(1).mean()
        forward_loss = batch.policy.mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss +
            self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        loss.backward()
        self.optim.step()
        res.update(
            {
                "loss/icm": loss.item(),
                "loss/icm/forward": forward_loss.item(),
                "loss/icm/inverse": inverse_loss.item()
            }
        )
        return res