# TODO: Implement Simple Policy Optimization paper: https://arxiv.org/pdf/2401.16025

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.objectives.ppo import PPOLoss
from torchrl._utils import _standardize
from torchrl.objectives.utils import _reduce

class SPOLoss(PPOLoss):
    """Simple Policy Optimization (SPO) Loss.

    SPO replaces PPO's clipping with a squared penalty on the importance weights 
    scaled by the advantage and an epsilon hyperparameter.
    
    Formula: L = rho * A - (|A| / 2*eps) * (rho - 1)^2

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.
        eps_clip (float): The trust region bound (epsilon), similar to PPO. Default: 0.2.
    """

    def __init__(
        self,
        actor_network,
        critic_network=None,
        *,
        eps_clip: float = 0.2,
        **kwargs,
    ):
        # We use the existing clip_epsilon from PPOLoss to keep compatibility
        super().__init__(actor_network, critic_network, clip_epsilon=eps_clip, **kwargs)

    @torchrl.objectives.common.dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        
        # 1. Advantage Estimation
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        
        if self.normalize_advantage and advantage.numel() > 1:
            advantage = _standardize(advantage, self.normalize_advantage_exclude_dims)

        # 2. Compute Importance Weights (rho)
        log_weight, dist, kl_approx = self._log_weight(
            tensordict, adv_shape=advantage.shape[:-1]
        )
        rho = log_weight.exp()

        # 3. SPO Objective calculation:
        # Penalty coefficient is |A| / (2 * epsilon)
        penalty_coeff = advantage.abs() / (2 * self.clip_epsilon)
        penalty = penalty_coeff * torch.square(rho - 1)
        
        # L = rho * A - penalty
        loss_objective = -(rho * advantage - penalty)

        # 4. Prepare Output
        td_out = TensorDict({"loss_objective": loss_objective}, batch_size=[])
        
        # Logging metadata
        td_out.set("rho_mean", rho.detach().mean())
        td_out.set("penalty_mean", penalty.detach().mean())
        if kl_approx is not None:
            td_out.set("kl_approx", kl_approx.detach().mean())

        # 5. Handle Entropy and Critic Loss (inherited logic)
        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))

        if self._has_critic:
            loss_critic, _, explained_variance = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if explained_variance is not None:
                td_out.set("explained_variance", explained_variance)

        # 6. Final reduction
        return td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )