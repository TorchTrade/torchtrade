# Examples

TorchTrade provides a collection of example training scripts to help you get started. These examples are designed for **inspiration and learning** - use them as starting points to build your own custom training pipelines.

## Design Philosophy

TorchTrade examples closely follow the structure of [TorchRL's SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), enabling **near plug-and-play compatibility** with any TorchRL algorithm. This means:

- Familiar structure if you've used TorchRL before
- Easy adaptation of TorchRL algorithms to trading environments
- Minimal boilerplate - focus on what's unique to your strategy
- Hydra configuration for easy experimentation

---

## Direct Compatibility with TorchRL SOTA Implementations

To demonstrate the closeness to TorchRL's [SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), here is a direct comparison:

**TorchRL's A2C Example:**
```python
from torchrl.envs import GymEnv

# Environment setup
env = GymEnv("CartPole-v1")

# Everything else stays the same
collector = SyncDataCollector(env, policy, ...)
loss_module = A2CLoss(actor, critic, ...)
optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

for batch in collector:
    loss_values = loss_module(batch)
    loss_values["loss"].backward()
    optimizer.step()
```

**TorchTrade Adaptation (Only Environment Changes):**
```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Environment setup - ONLY CHANGE
env = SeqLongOnlyEnv(df, SeqLongOnlyEnvConfig(...))

# Everything else stays EXACTLY the same
collector = SyncDataCollector(env, policy, ...)
loss_module = A2CLoss(actor, critic, ...)
optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

for batch in collector:
    loss_values = loss_module(batch)
    loss_values["loss"].backward()
    optimizer.step()
```

**That's it!** The collector, loss function, optimizer, and training loop remain identical. This doesn't only allow the use of any TorchRL algorithm, but any of the other useful components of TorchRL - replay buffers, transforms, modules, data structures - and provides seamless integration into the entire TorchRL ecosystem.

---

## Example File Structure

Each example in TorchTrade follows a consistent file organization pattern that separates configuration, utilities, and training logic:

```
examples/online/<algorithm>/
├── config.yaml           # Main configuration file
├── utils.py             # Helper functions (env creation, network setup, etc.)
└── train.py            # Training script and main loop
```

### What Each Component Does

**`config.yaml` - Configuration Management**

The configuration file uses [Hydra](https://hydra.cc/) to manage all hyperparameters and settings. This includes:

- **Environment settings**: Symbol, timeframes, initial cash, transaction fees, window sizes
- **Network architecture**: Hidden dimensions, activation functions, layer configurations
- **Training hyperparameters**: Learning rate, batch size, discount factor (gamma), entropy coefficient
- **Collector settings**: Frames per batch, number of parallel environments
- **Logging**: Wandb project name, experiment tracking settings

By centralizing all parameters in YAML, you can easily experiment with different configurations without modifying code. Hydra also allows you to override any parameter from the command line:

```bash
# Override multiple parameters
python train.py env.symbol="ETH/USD" optim.lr=1e-4 loss.gamma=0.95
```

**Example config.yaml:**
```yaml
env:
  name: SeqLongOnlyEnv
  symbol: "BTC/USD"
  time_frames: ["5Min", "15Min"]
  window_sizes: [10, 10]
  execute_on: "15Min"
  initial_cash: [1000, 5000]
  transaction_fee: 0.0025
  train_envs: 10
  eval_envs: 2

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000

optim:
  lr: 2.5e-4
  max_grad_norm: 0.5
  anneal_lr: True

loss:
  gamma: 0.9
  mini_batch_size: 33333
  ppo_epochs: 3
  clip_epsilon: 0.1
  entropy_coef: 0.01

logger:
  backend: wandb
  project_name: TorchTrade-Online
  exp_name: ppo
```

**`utils.py` - Helper Functions**

This file contains modular helper functions that handle setup tasks:

- **`make_env()`**: Creates and configures the trading environment (offline or online)
- **`make_actor()`**: Builds the policy network architecture (deterministic or stochastic)
- **`make_critic()`**: Creates the value function network (if needed for the algorithm)
- **`make_loss()`**: Initializes the loss module (PPO, SAC, IQL, etc.)
- **`make_collector()`**: Sets up the data collection pipeline

These utility functions keep the main training script clean and make it easy to swap components (e.g., changing from SeqLongOnlyEnv to SeqFuturesEnv requires only modifying `make_env()`).

**`train.py` - Training Loop**

The main training script orchestrates everything:

1. **Hydra initialization**: Loads configuration from `config.yaml`
2. **Component creation**: Uses `utils.py` functions to create env, actor, loss, optimizer
3. **Training loop**: Collects data, computes losses, updates policy, logs metrics
4. **Evaluation**: Periodically evaluates the policy on test environments
5. **Checkpointing**: Saves model weights and training state

This structure mirrors [TorchRL's SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), making it familiar to TorchRL users and easy to adapt existing algorithms.

---

## Available Examples

The following examples demonstrate the flexibility of TorchTrade across different algorithms, environments, and use cases. These examples are meant to be starting points for further experimentation and adaptation - customize them according to your needs, ideas, and environments.

!!! warning "Hyperparameters Not Tuned"
    **All hyperparameters in our examples are NOT tuned.** The configurations provided are starting points for experimentation, not optimized settings. You should tune hyperparameters (learning rates, network architectures, reward functions, etc.) according to your specific trading environment, market conditions, and objectives.

### Online RL (Offline Backtesting Environments)

These examples use online RL algorithms (learning from interaction as it happens) with historical market data for backtesting. This allows you to train policies on past data before deploying them to live trading environments. We typically split the training data into training and test environments to evaluate the generalization performance of learned policies on unseen market conditions.

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

### Offline RL

These examples use offline RL algorithms that learn from pre-collected datasets without requiring live environment interaction during training. The data can be collected from interactions with offline backtesting environments or from real online live trading sessions. We provide simple example offline datasets at [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade).

Located in `examples/offline/`:

| Example | Algorithm | Environment | Key Features |
|---------|-----------|-------------|--------------|
| **iql/** | IQL | SeqLongOnlyEnv | Offline RL from pre-collected trajectories |

**Note**: Thanks to the compatibility with TorchRL, you can easily add other offline RL methods like CQL, TD3+BC, and Decision Transformers from TorchRL to do offline RL with TorchTrade environments.

### LLM Actors

TorchTrade implements LLMActor that allows the integration of local LLMs for trading decision-making. You can use models directly from [HuggingFace Models](https://huggingface.co/models) or quantized models from [Unsloth](https://unsloth.ai/) for memory-efficient and fast interactions. This approach leverages the reasoning capabilities of LLMs to make trading decisions and provides fertile soil for new ideas and adaptations for trading with LLMs thanks to TorchTrade.

Located in `examples/llm/`:

| Example | Type | Description |
|---------|------|-------------|
| **local/** | LocalLLMActor | Trading with local LLMs (vLLM/transformers backend) |

**Future Work**: We plan to provide example scripts for fine-tuning LLMs on reasoning traces from frontier models for trading. Additionally, we are working on integrating Vision-Language Models (VLMs) to process trading chart plots for decision-making.

### Rule-Based Actors

TorchTrade provides actor classes that allow easy creation of rule-based trading strategies using technical indicators and market signals, for example mean reversion, breakout, and more. These rule-based actors integrate seamlessly with TorchTrade environments for both backtesting and live trading, serving as baselines or components in hybrid approaches. This is especially interesting with our **[custom feature preprocessing](guides/custom-features.md)**, which allows you to add technical indicators and derived features to enhance rule-based strategies.

Located in `examples/online/rulebased/`:

| Example | Actor Type | Environment | Description |
|---------|------------|-------------|-------------|
| **rulebased/** | MeanReversionActor | SeqLongOnlyEnv | Mean reversion strategy using Bollinger Bands and Stochastic RSI |

**Future Work**: We plan to provide examples of hybrid approaches that combine rule-based policies with neural network policies as actors, leveraging the strengths of both deterministic strategies and learned behaviors.

### Live Trading

These examples demonstrate deploying trained policies to real exchange APIs for live trading. We strongly recommend starting with paper trading to validate your strategies risk-free before transitioning to live capital deployment.

Located in `examples/live/`:

| Example | Exchange | Description |
|---------|----------|-------------|
| **alpaca/** | Alpaca | Live paper trading with Alpaca API |

### Transforms

Inspired by work such as [R3M](https://arxiv.org/abs/2203.12601) and [VIP](https://arxiv.org/abs/2210.00030) that utilize large pretrained models for representation learning, we created the ChronosEmbeddingTransform using [Chronos forecasting models](https://github.com/amazon-science/chronos-forecasting) to embed historical trading data. This demonstrates the flexibility and adaptability of TorchTrade for integrating pretrained models as transforms for enhanced feature representations.

Located in `examples/transforms/`:

| Example | Transform | Description |
|---------|-----------|-------------|
| **chronos_embedding_example.py** | ChronosEmbeddingTransform | Time series embedding with Chronos T5 models |

**Note**: If you would like us to add additional transforms for other pretrained models (similar to ChronosEmbedding), we welcome [GitHub issues](https://github.com/TorchTrade/torchtrade/issues) with your requests. We're happy to implement these given the availability of model weights and resources.

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

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade/issues)
- 📧 **Email**: torchtradecontact@gmail.com
