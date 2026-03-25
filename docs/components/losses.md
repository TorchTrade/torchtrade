# Loss Functions

TorchTrade provides specialized loss functions for training RL trading agents, built on TorchRL's `LossModule` interface.

## Available Loss Functions

| Loss Function | Type | Use Case |
|---------------|------|----------|
| [**DGLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/dg_loss.py) | Policy Gradient | Delight-gated updates — no importance sampling needed |
| [**GRPOLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/grpo_loss.py) | Policy Gradient | One-step RL with SLTP environments |
| [**CTRLLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/ctrl.py) | Representation Learning | Self-supervised encoder training |
| [**CTRLPPOLoss**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/losses/ctrl.py) | Combined | Joint policy + representation learning |

For standard multi-step RL (PPO, DQN, SAC, IQL), use TorchRL's built-in loss modules directly.

---

## DGLoss

Delightful Policy Gradient — gates each update by $\sigma(\text{delight} / \eta)$, where $\text{delight} = \text{advantage} \times \text{surprisal}$. This suppresses rare failures and amplifies rare successes without requiring importance sampling (no old log-probs needed).

$$
\begin{aligned}
\text{surprisal} &= -\log \pi(a|s) \\
\text{advantage} &= r - b \\
\text{delight} &= \text{advantage} \times \text{surprisal} \\
\text{gate} &= \sigma(\text{delight} / \eta) \\
\mathcal{L} &= -\mathbb{E}\left[\log \pi(a|s) \cdot \text{sg}(\text{gate} \cdot \text{advantage})\right]
\end{aligned}
$$

**Key difference from PPO/GRPO:** DG uses only the current policy's log-probabilities. No importance ratios, no behavior probabilities. This makes it simpler and naturally robust to stale or off-policy data.

### Why DG for trading?

Trading rewards are heavy-tailed: most trades produce small P&L, but a few outliers (flash crashes, breakouts) dominate the gradient. Standard PG and GRPO weight updates purely by advantage magnitude, so one catastrophic trade can overwhelm training even when the policy assigned near-zero probability to that action.

The delight gate fixes this. A large loss on an unlikely action produces negative delight, pushing the gate toward 0 and shrinking its gradient contribution. Conversely, an unexpected win produces positive delight, pushing the gate toward 1 so the policy learns faster from it. The net effect: black-swan losses don't destabilize training, and rare profitable signals get amplified.

DG also drops the importance sampling machinery that GRPO needs. GRPO requires old log-probs (`sample_log_prob`) to compute policy ratios, which means storing behavior policy state and dealing with ratio clipping when data gets stale. DG only needs the current policy's log-probs, so it works cleanly with replay buffers, asynchronous collection, and distributed setups where the collecting policy drifts from the learner.

**When to use DG vs GRPO:**

| Scenario | Recommendation |
|----------|---------------|
| One-step SLTP with fresh on-policy batches | Either works; GRPO is well-tested |
| Sequential environments (multi-step episodes) | DG, no need to track old log-probs across steps |
| Replay buffer / offline data | DG, no importance sampling artifacts |
| Distributed collection with stale actors | DG, designed for this (see arXiv:2603.20521) |
| Heavy-tailed reward distributions | DG, gate suppresses outlier-driven gradient noise |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor_network` | Required | Policy network (ProbabilisticTensorDictSequential) |
| `eta` | 1.0 | Sigmoid gate temperature (lower = sharper gating) |
| `baseline` | `"mean"` | Baseline type: `"mean"`, `"none"`, or `"expected"` |
| `entropy_bonus` | True | Whether to add entropy regularization |
| `entropy_coeff` | 0.01 | Entropy regularization coefficient |

```python
from torchtrade.losses import DGLoss

loss_module = DGLoss(actor_network=actor, eta=1.0, baseline="mean")

for batch in collector:
    loss_td = loss_module(batch)
    loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
    loss.backward()
    optimizer.step()

    # DG-specific diagnostics
    print(f"gate: {loss_td['gate'].item():.3f}, advantage: {loss_td['advantage'].item():.3f}")
```

**Baseline modes:**

- `"mean"` — batch mean of rewards (default, simple and effective)
- `"none"` — no baseline (raw rewards as advantage)
- `"expected"` — expected reward under current policy: $\sum_a \pi(a|s) \cdot R(s,a)$. Requires a `"counterfactual_rewards"` key of shape `(batch, n_actions)` in the input TensorDict.

**Papers:**

- [Delightful Policy Gradient (arXiv:2603.14608)](https://arxiv.org/abs/2603.14608) — Introduces DG and shows it enhances directional accuracy within single contexts and shifts expected gradient closer to supervised cross-entropy across multiple contexts.
- [Delightful Distributed Policy Gradient (arXiv:2603.20521)](https://arxiv.org/abs/2603.20521) — Extends DG to distributed RL with stale, buggy, or mismatched actors. DG without off-policy correction outperforms importance-weighted PG with exact behavior probabilities.

**Example:** See [`examples/losses/dg_mnist.py`](https://github.com/TorchTrade/torchtrade/blob/main/examples/losses/dg_mnist.py) for a comparison of CE, REINFORCE, and DG on MNIST framed as a bandit problem.

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
