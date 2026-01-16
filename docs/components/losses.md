# Loss Functions

TorchTrade provides specialized loss functions for training RL trading agents, including one-step policy gradient methods (GRPO) and representation learning objectives (CTRL).

## Available Loss Functions

| Loss Function | Type | Use Case | Key Features |
|---------------|------|----------|--------------|
| **GRPOLoss** | Policy Gradient | One-step RL with SLTP environments | Group Relative Policy Optimization, entropy regularization, advantage normalization |
| **CTRLLoss** | Representation Learning | Self-supervised encoder training | Cross-trajectory representation learning, prototype-based contrastive learning |
| **CTRLPPOLoss** | Combined | Joint policy + representation learning | Combines ClipPPOLoss with CTRLLoss for end-to-end training |

---

## GRPOLoss

Group Relative Policy Optimization (GRPO) loss for one-step reinforcement learning. This loss is designed for environments where actions have immediate consequences, such as SLTP (Stop-Loss/Take-Profit) bracket order environments.

### Key Features

- **One-step optimization**: No multi-step rollouts or value estimation required
- **Advantage normalization**: Standardizes advantages within each batch for stable training
- **Clipped policy updates**: Similar to PPO, prevents destructive policy updates
- **Entropy regularization**: Encourages exploration through entropy bonus

### When to Use

- **FuturesOneStepEnv** or **LongOnlyOneStepEnv** - One-step environments where episodes reset after each action
- **SLTP bracket orders** - Trading with predefined stop-loss and take-profit levels
- **Contextual bandit** - Single-step decision problems

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_network` | ProbabilisticTensorDictSequential | Required | Policy network that outputs action distributions |
| `entropy_coeff` | float or dict | 0.01 | Entropy regularization coefficient (supports per-head coefficients for composite actions) |
| `epsilon_low` | float | 0.2 | Lower clipping bound for policy ratio |
| `epsilon_high` | float | 0.2 | Upper clipping bound for policy ratio |
| `samples_mc_entropy` | int | 1 | Number of Monte Carlo samples for entropy estimation |
| `reduction` | str | "mean" | Reduction method for loss aggregation ("mean", "sum", "none") |

### Usage Example

```python
from torchtrade.losses import GRPOLoss
from torchrl.modules import ProbabilisticActor

# Create policy network
actor = ProbabilisticActor(
    module=policy_network,
    in_keys=["observation"],
    out_keys=["action"],
    distribution_class=Categorical,
)

# Create GRPO loss
loss_module = GRPOLoss(
    actor_network=actor,
    entropy_coeff=0.01,
)

# Training loop
for batch in collector:
    loss_td = loss_module(batch)
    loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
    loss.backward()
    optimizer.step()
```

**Reference**: [`torchtrade/losses/grpo_loss.py`](https://github.com/TorchTrade/TorchTrade/blob/main/torchtrade/losses/grpo_loss.py)

---

## CTRLLoss

Cross-Trajectory Representation Learning (CTRL) loss for self-supervised encoder training. CTRL improves zero-shot generalization by training encoders to recognize behavioral similarity across trajectories without using rewards.

### Key Features

- **Self-supervised learning**: No reward labels required
- **Prototype-based clustering**: Uses Sinkhorn algorithm for soft cluster assignments
- **MYOW loss**: Encourages cross-trajectory consistency for similar behaviors
- **Zero-shot transfer**: Representations generalize across market conditions

### When to Use

- **Encoder pre-training** - Train shared encoder before policy learning
- **Multi-environment transfer** - Learn representations that work across different assets
- **Data-efficient RL** - Leverage unlabeled trajectory data for better representations

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_network` | TensorDictModule | Required | Encoder network that produces embeddings |
| `embedding_dim` | int | Required | Dimension of encoder output embeddings |
| `projection_dim` | int | 128 | Dimension of projection space for prototypes |
| `num_prototypes` | int | 512 | Number of learnable prototype vectors |
| `sinkhorn_iters` | int | 3 | Iterations for Sinkhorn-Knopp algorithm |
| `temperature` | float | 0.1 | Temperature for softmax (lower = more peaked assignments) |
| `window_len` | int | 4 | Length of sliding window for trajectory segments |
| `myow_k` | int | 5 | Number of nearest prototypes for MYOW loss |
| `myow_coeff` | float | 1.0 | Coefficient for MYOW loss term |
| `reduction` | str | "mean" | Reduction method for loss aggregation |

### Usage Example

```python
from torchtrade.losses import CTRLLoss
from tensordict.nn import TensorDictModule

# Create encoder network
encoder = TensorDictModule(
    nn.Sequential(
        nn.Linear(observation_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ),
    in_keys=["observation"],
    out_keys=["embedding"]
)

# Create CTRL loss
ctrl_loss = CTRLLoss(
    encoder_network=encoder,
    embedding_dim=128,
    num_prototypes=512,
    myow_coeff=1.0,
)

# Pre-training loop
for batch in collector:
    loss_td = ctrl_loss(batch)
    loss = loss_td["loss_ctrl"]
    loss.backward()
    optimizer.step()
```

**Reference**: ["Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL" (arXiv:2106.02193)](https://arxiv.org/abs/2106.02193)

---

## CTRLPPOLoss

Combined loss module that jointly trains policy (via PPO) and encoder representations (via CTRL). This enables end-to-end training where the encoder learns useful representations while the policy learns to act.

### Key Features

- **Joint optimization**: Trains policy and encoder simultaneously
- **Shared encoder**: Encoder representations used by both actor and critic
- **Weighted combination**: Control relative importance of PPO and CTRL objectives

### When to Use

- **End-to-end training** - Train policy and encoder together from scratch
- **Representation + policy learning** - When both components need to adapt jointly
- **Multi-task RL** - Learn representations that support multiple downstream tasks

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ppo_loss` | LossModule | Required | PPO loss module (e.g., ClipPPOLoss) |
| `ctrl_loss` | CTRLLoss | Required | CTRL loss module for representation learning |
| `ctrl_coeff` | float | 1.0 | Coefficient for CTRL loss term (balances with PPO loss) |

### Usage Example

```python
from torchtrade.losses import CTRLLoss, CTRLPPOLoss
from torchrl.objectives import ClipPPOLoss

# Create PPO loss
ppo_loss = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
)

# Create CTRL loss for encoder
ctrl_loss = CTRLLoss(
    encoder_network=encoder,
    embedding_dim=128,
)

# Combine into joint loss
combined_loss = CTRLPPOLoss(
    ppo_loss=ppo_loss,
    ctrl_loss=ctrl_loss,
    ctrl_coeff=0.5,  # Weight CTRL loss at 50%
)

# Training loop
for batch in collector:
    loss_td = combined_loss(batch)
    # loss_td contains both PPO keys (loss_objective, loss_critic, etc.)
    # and CTRL keys (loss_ctrl, loss_proto, loss_myow)
    total_loss = (
        loss_td["loss_objective"] +
        loss_td["loss_critic"] +
        loss_td["loss_ctrl"]
    )
    total_loss.backward()
    optimizer.step()
```

---

## Quick Reference

### Loss Selection Guide

| Scenario | Recommended Loss | Notes |
|----------|------------------|-------|
| One-step SLTP trading | GRPOLoss | No value estimation needed |
| Encoder pre-training | CTRLLoss | Self-supervised, no rewards |
| Joint policy + encoder | CTRLPPOLoss | End-to-end training |
| Multi-step sequential | ClipPPOLoss (TorchRL) | Use TorchRL's built-in losses |

### Common Patterns

**Pattern 1: GRPO for One-Step Environments**
```python
loss = GRPOLoss(actor, entropy_coeff=0.01)
# Train with: loss_objective + loss_entropy
```

**Pattern 2: CTRL Pre-Training + Fine-Tuning**
```python
# Phase 1: Pre-train encoder with CTRL
ctrl_loss = CTRLLoss(encoder, embedding_dim=128)
# ... train encoder ...

# Phase 2: Freeze encoder, train policy with PPO
ppo_loss = ClipPPOLoss(actor, critic)
encoder.requires_grad_(False)
# ... train policy ...
```

**Pattern 3: Joint Training**
```python
combined = CTRLPPOLoss(ppo_loss, ctrl_loss, ctrl_coeff=0.3)
# Single optimizer updates both policy and encoder
```

---

## Implementation Details

### GRPOLoss Advantage Computation

GRPO normalizes advantages within each batch using the immediate reward:

```python
advantage = (reward - reward.mean()) / (reward.std() + 1e-8)
```

This per-batch normalization makes training stable even when reward scales vary significantly.

### CTRL Loss Components

The total CTRL loss combines two terms:

```python
loss_ctrl = loss_proto + myow_coeff * loss_myow
```

- **loss_proto**: Prototype contrastive loss (cross-entropy between predictions and Sinkhorn targets)
- **loss_myow**: MYOW loss (cosine similarity for trajectories with shared prototypes)

---

## See Also

- [Environments Guide](../environments/offline.md) - Compatible environment types
- [Transforms Guide](transforms.md) - Data augmentation and preprocessing
- [Custom Reward Functions](../guides/reward-functions.md) - Reward engineering patterns
