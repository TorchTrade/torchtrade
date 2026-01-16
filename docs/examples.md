# Examples

TorchTrade provides a collection of example training scripts to help you get started. These examples are designed for **inspiration and learning** - use them as starting points to build your own custom training pipelines.

## Design Philosophy

TorchTrade examples closely follow the structure of [TorchRL's SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), enabling **near plug-and-play compatibility** with any TorchRL algorithm. This means:

- Familiar structure if you've used TorchRL before
- Easy adaptation of TorchRL algorithms to trading environments
- Minimal boilerplate - focus on what's unique to your strategy
- Hydra configuration for easy experimentation

---

## Available Examples

### Online Training (Offline Backtesting Environments)

Located in `examples/online/`:

| Example | Algorithm | Environment | Key Features |
|---------|-----------|-------------|--------------|
| **ppo/** | PPO | SeqLongOnlyEnv | Standard policy gradient with multi-timeframe observations |
| **ppo_futures/** | PPO | SeqFuturesEnv | Futures trading with leverage and margin management |
| **ppo_futures_sltp/** | PPO | SeqFuturesSLTPEnv | Futures with bracket orders (stop-loss/take-profit) |
| **ppo_futures_onestep/** | PPO | FuturesOneStepEnv | One-step futures for episodic optimization |
| **ppo_chronos/** | PPO | SeqLongOnlyEnv + ChronosEmbedding | Time series embedding with Chronos T5 models |
| **iql/** | IQL (Implicit Q-Learning) | SeqLongOnlyEnv | Offline RL algorithm for sequential trading |
| **dsac/** | DSAC (Distributional SAC) | SeqLongOnlyEnv | Soft actor-critic with distributional value functions |
| **grpo_futures_onestep/** | GRPO | FuturesOneStepEnv | Group relative policy optimization for one-step RL |
| **long_onestep_env/** | GRPO | LongOnlyOneStepEnv | One-step long-only with SLTP bracket orders |
| **rulebased/** | Rule-Based | SeqLongOnlyEnv | MeanReversionActor baseline (no RL training) |

### Offline Training

Located in `examples/offline/`:

| Example | Algorithm | Environment | Key Features |
|---------|-----------|-------------|--------------|
| **iql/** | IQL | SeqLongOnlyEnv | Offline RL from pre-collected trajectories |

### LLM Actors

Located in `examples/llm/`:

| Example | Type | Description |
|---------|------|-------------|
| **local/** | LocalLLMActor | Trading with local LLMs (vLLM/transformers backend) |

### Live Trading

Located in `examples/live/`:

| Example | Exchange | Description |
|---------|----------|-------------|
| **alpaca/** | Alpaca | Live paper trading with Alpaca API |

### Human Actors

Located in `examples/human/`:

| Example | Type | Description |
|---------|------|-------------|
| **human_exe_longseq/** | HumanActor | Interactive human trading for expert demonstrations |

### Transforms

Located in `examples/transforms/`:

| Example | Transform | Description |
|---------|-----------|-------------|
| **chronos_embedding_example.py** | ChronosEmbeddingTransform | Time series embedding with Chronos T5 models |

---

## Running Examples

All examples use Hydra for configuration management:

```bash
# Run with default configuration
uv run python examples/online/ppo/train.py

# Override config parameters
uv run python examples/online/ppo/train.py \
    env.symbol="BTC/USD" \
    optim.lr=1e-4 \
    collector.frames_per_batch=1000 \
    loss.gamma=0.95
```

### Common Hydra Overrides

| Parameter | Example | Description |
|-----------|---------|-------------|
| `env.symbol` | `"BTC/USD"` | Trading pair/symbol |
| `env.initial_cash` | `10000` | Starting capital |
| `env.time_frames` | `'["1min","5min"]'` | Multi-timeframe observations |
| `optim.lr` | `1e-4` | Learning rate |
| `loss.gamma` | `0.99` | Discount factor |
| `collector.frames_per_batch` | `2000` | Frames collected per iteration |
| `total_frames` | `100000` | Total training frames |

---

## Example Structure

Each training example follows this pattern:

```python
# 1. Configuration (via Hydra)
@hydra.main(config_path="config", config_name="config")
def main(cfg):

    # 2. Create environment
    env = make_env(cfg)

    # 3. Build policy network
    actor = make_actor(cfg, env)

    # 4. Create collector
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
    )

    # 5. Loss function
    loss_module = make_loss(cfg, actor)

    # 6. Optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr
    )

    # 7. Training loop
    for batch in collector:
        loss_values = loss_module(batch)
        loss = loss_values["loss_objective"] + loss_values["loss_critic"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

This structure mirrors [TorchRL's SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), making it easy to adapt other algorithms.

---

## Adapting TorchRL Algorithms

TorchTrade environments are fully compatible with TorchRL. To use a TorchRL algorithm:

1. **Use TorchTrade environment** instead of Gym environment
2. **Keep everything else the same** - collector, loss, optimizer
3. **Optionally customize** - reward functions, features, action spaces

**Example: Adapting TorchRL's A2C to TorchTrade**

```python
# From TorchRL A2C example
from torchrl.envs import GymEnv
env = GymEnv("CartPole-v1")

# Change to TorchTrade
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
env = SeqLongOnlyEnv(df, SeqLongOnlyEnvConfig(...))

# Rest of A2C code works unchanged!
collector = SyncDataCollector(env, policy, ...)
loss_module = A2CLoss(actor, critic, ...)
# ... training loop
```

---

## Configuration Files

Each example includes a `config/` directory with Hydra configs:

```
examples/online/ppo/
├── train.py              # Training script
├── config/
│   ├── config.yaml       # Main config
│   ├── env/
│   │   └── default.yaml  # Environment config
│   ├── collector/
│   │   └── default.yaml  # Data collection config
│   └── loss/
│       └── default.yaml  # Loss function config
```

This allows clean separation of concerns and easy experimentation.

---

## Creating Your Own Examples

To create a new training script:

1. **Copy an existing example** that's closest to your use case
2. **Modify the environment** - change to your preferred env and config
3. **Customize the policy** - adjust network architecture if needed
4. **Tune hyperparameters** - update config files
5. **Add custom logic** - rewards, features, transforms

**Tips:**
- Start with `ppo/` for standard RL
- Start with `grpo_futures_onestep/` for one-step RL
- Start with `ppo_chronos/` for time series embeddings
- Start with `rulebased/` for non-RL baselines

---

## Next Steps

- **[Offline Environments](environments/offline.md)** - Understand environment mechanics
- **[Reward Functions](guides/reward-functions.md)** - Design better reward signals
- **[Feature Engineering](guides/custom-features.md)** - Add technical indicators
- **[Transforms](components/transforms.md)** - Use ChronosEmbedding and other transforms
- **[TorchRL SOTA Implementations](https://github.com/pytorch/rl/tree/main/sota-implementations)** - Explore more TorchRL algorithms

---

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade_envs/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade_envs/issues)
- 📧 **Email**: torchtradecontact@gmail.com
