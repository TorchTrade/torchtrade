"""Delightful Gradient (DG) loss module.

Implements the Delightful Policy Gradient from arXiv:2603.14608, extended to
distributed settings in arXiv:2603.20521. Gates each policy gradient update by
sigmoid(delight / η), where delight = advantage × surprisal. This suppresses
rare failures and amplifies rare successes without requiring behavior
probabilities (no importance sampling).

Key difference from PPO/GRPO: DG uses only the current policy's log-probabilities,
not importance ratios. This makes it simpler and removes the need for old log-probs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from tensordict import (
    is_tensor_collection,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _maybe_add_or_extend_key,
    _maybe_get_or_select,
    _reduce,
    _sum_td_features,
)


class DGLoss(LossModule):
    """Delightful Gradient loss — gates updates by sigmoid(delight / η).

    The loss is computed as::

        surprisal = -log π(a|s)
        advantage = reward - baseline
        delight   = advantage × surprisal
        gate      = σ(delight / η)
        loss      = -𝔼[log π(a|s) · sg(gate · advantage)]

    where sg() is stop-gradient.

    Args:
        actor_network: The policy network (ProbabilisticTensorDictSequential).
        eta: Temperature for the sigmoid gate. Lower η → sharper gating.
            Default: 1.0.
        baseline: How to compute the reward baseline.
            - ``"mean"``: batch mean of rewards (default).
            - ``"none"``: no baseline (raw rewards as advantage).
        entropy_bonus: Whether to add an entropy bonus. Default: True.
        entropy_coeff: Coefficient for the entropy bonus. Default: 0.01.
        samples_mc_entropy: Number of samples for MC entropy estimation. Default: 1.
        reduction: Reduction mode ("mean", "sum", "none"). Default: "mean".
        functional: Whether to use functional mode. Default: True.
    """

    @dataclass
    class _AcceptedKeys:
        """Default tensordict keys for DGLoss.

        Attributes:
            action: Key for the action tensor. Defaults to ``"action"``.
            reward: Key for the reward tensor. Defaults to ``"reward"``.
        """

        action: NestedKey | list[NestedKey] = "action"
        reward: NestedKey | list[NestedKey] = "reward"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    actor_network: ProbabilisticTensorDictModule
    actor_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        *,
        eta: float = 1.0,
        baseline: str = "mean",
        entropy_bonus: bool = True,
        entropy_coeff: float = 0.01,
        samples_mc_entropy: int = 1,
        reduction: str | None = None,
        functional: bool = True,
        device: torch.device | None = None,
    ):
        if actor_network is None:
            raise TypeError("Missing positional argument actor_network.")

        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta}")

        if baseline not in ("mean", "none"):
            raise ValueError(
                f"baseline must be 'mean' or 'none', got '{baseline}'"
            )

        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._in_keys = None
        self._out_keys = None
        super().__init__()

        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network
            self.actor_network_params = None

        self.eta = eta
        self.baseline_type = baseline
        self.entropy_bonus = entropy_bonus
        self.samples_mc_entropy = samples_mc_entropy
        self.reduction = reduction
        self.register_buffer("entropy_coeff", torch.tensor(float(entropy_coeff)))

        try:
            action_keys = self.actor_network.dist_sample_keys
            if len(action_keys) == 1:
                self.set_keys(action=action_keys[0])
            else:
                self.set_keys(action=action_keys)
        except AttributeError:
            pass

    @property
    def functional(self):
        return self._functional

    def _set_in_keys(self):
        keys = []
        _maybe_add_or_extend_key(keys, self.actor_network.in_keys)
        _maybe_add_or_extend_key(keys, self.tensor_keys.action)
        _maybe_add_or_extend_key(keys, self.tensor_keys.reward, "next")
        self._in_keys = list(dict.fromkeys(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "gate", "advantage"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def reset(self) -> None:
        pass

    def _get_log_prob_and_dist(self, tensordict):
        """Get current policy log-prob and distribution."""
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            with self.actor_network_params.to_module(
                self.actor_network
            ) if self.functional else contextlib.nullcontext():
                dist = self.actor_network.get_dist(tensordict)

            action = _maybe_get_or_select(tensordict, self.tensor_keys.action)
            if action.requires_grad:
                raise RuntimeError(
                    f"tensordict stored {self.tensor_keys.action} requires grad."
                )

            is_composite = isinstance(dist, CompositeDistribution)
            log_prob = dist.log_prob(action)

            if is_composite and is_tensor_collection(log_prob):
                log_prob = _sum_td_features(log_prob)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported."
            )
        return log_prob, dist

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
            if not entropy.isfinite().all():
                del entropy
                raise NotImplementedError
        except NotImplementedError:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)}. Using MC sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            if is_tensor_collection(log_prob):
                log_prob = _sum_td_features(log_prob)
            entropy = -log_prob.mean(0)
            if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
                entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)

    def _compute_baseline(self, reward):
        """Compute reward baseline for advantage estimation."""
        if self.baseline_type == "none":
            return 0.0
        # baseline_type == "mean" (validated in __init__)
        return reward.mean(0, keepdim=True)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)

        # 1. Get current policy log-prob and distribution
        log_prob, dist = self._get_log_prob_and_dist(tensordict)

        # 2. Get reward and compute advantage
        reward = tensordict["next", self.tensor_keys.reward]  # (batch, 1)
        baseline = self._compute_baseline(reward)
        advantage = reward - baseline  # (batch, 1)

        # 3. Compute delight gate
        surprisal = -log_prob.unsqueeze(-1)  # (batch, 1)
        delight = advantage * surprisal  # (batch, 1)
        gate = torch.sigmoid(delight / self.eta)  # (batch, 1)

        # 4. Compute gated policy gradient loss
        loss = surprisal * (gate * advantage).detach()

        td_out = TensorDict({"loss_objective": loss})
        td_out.set("gate", gate.detach().mean())
        td_out.set("advantage", advantage.detach().mean())

        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                td_out.set("entropy", _sum_td_features(entropy).detach().mean())
            else:
                td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out
