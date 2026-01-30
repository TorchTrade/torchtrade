# Loss Functions

TorchTrade provides specialized loss functions for training RL trading agents, built on TorchRL's `LossModule` interface.

## Available Loss Functions

| Loss Function | Type | Use Case |
|---------------|------|----------|
| [**GRPOLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/grpo_loss.py) | Policy Gradient | One-step RL with SLTP environments |
| [**CTRLLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/ctrl.py) | Representation Learning | Self-supervised encoder training |
| [**CTRLPPOLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/ctrl.py) | Combined | Joint policy + representation learning |

For standard multi-step RL (PPO, DQN, SAC, IQL), use TorchRL's built-in loss modules directly.

---

## GRPOLoss

Group Relative Policy Optimization for one-step RL. Designed for `OneStepTradingEnv` where episodes are single decisions with SL/TP bracket orders. Normalizes advantages within each batch: `advantage = (reward - mean) / std`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor_network` | Required | Policy network (ProbabilisticTensorDictSequential) |
| `entropy_coeff` | 0.01 | Entropy regularization coefficient |
| `epsilon_low` / `epsilon_high` | 0.2 | Clipping bounds for policy ratio |

```python
from torchtrade.losses import GRPOLoss

loss_module = GRPOLoss(actor_network=actor, entropy_coeff=0.01)

for batch in collector:
    loss_td = loss_module(batch)
    loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
    loss.backward()
    optimizer.step()
```

**Paper**: [DeepSeekMath (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300) — Section 2.2

---

## CTRLLoss

Cross-Trajectory Representation Learning for self-supervised encoder training. Trains encoders to recognize behavioral similarity across trajectories without rewards, improving zero-shot generalization.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_network` | Required | Encoder that produces embeddings |
| `embedding_dim` | Required | Dimension of encoder output |
| `num_prototypes` | 512 | Learnable prototype vectors |
| `sinkhorn_iters` | 3 | Sinkhorn-Knopp iterations |
| `temperature` | 0.1 | Softmax temperature |
| `myow_coeff` | 1.0 | MYOW loss coefficient |

```python
from torchtrade.losses import CTRLLoss

ctrl_loss = CTRLLoss(
    encoder_network=encoder,
    embedding_dim=128,
    num_prototypes=512,
)

for batch in collector:
    loss_td = ctrl_loss(batch)
    loss_td["loss_ctrl"].backward()
    optimizer.step()
```

**Paper**: [Cross-Trajectory Representation Learning (arXiv:2106.02193)](https://arxiv.org/abs/2106.02193)

---

## CTRLPPOLoss

Combines ClipPPOLoss with CTRLLoss for joint policy and encoder training. The encoder learns useful representations while the policy learns to act.

```python
from torchtrade.losses import CTRLLoss, CTRLPPOLoss
from torchrl.objectives import ClipPPOLoss

combined_loss = CTRLPPOLoss(
    ppo_loss=ClipPPOLoss(actor, critic),
    ctrl_loss=CTRLLoss(encoder, embedding_dim=128),
    ctrl_coeff=0.5,
)

for batch in collector:
    loss_td = combined_loss(batch)
    total_loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_ctrl"]
    total_loss.backward()
    optimizer.step()
```

---

## See Also

- [Examples](../examples/index.md) - Training scripts using these losses
- [Environments](../environments/offline.md) - Compatible environment types
- [TorchRL Objectives](https://pytorch.org/rl/reference/objectives.html) - Built-in PPO, DQN, SAC, IQL losses
