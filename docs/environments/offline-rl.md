# Offline RL

TorchTrade supports **offline reinforcement learning**, enabling agents to learn from pre-collected datasets without requiring live environment interaction during training.

## Overview

TorchTrade provides **TensorDict-based datasets** that can be loaded and used directly with [TorchRL's replay buffer](https://docs.pytorch.org/rl/main/tutorials/rb_tutorial.html). These datasets are available for download from [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade) and contain pre-collected trading trajectories for offline RL research.

Offline RL can be performed using datasets collected from two sources:

1. **Offline Environment Interactions** - Collect trajectories by running policies in backtesting environments (SeqLongOnlyEnv, SeqFuturesEnv, etc.)
2. **Real Online Environment Interactions** - Record actual trading data from live exchanges (Alpaca, Binance, Bitget)

This approach is particularly valuable for:
- Learning from expert demonstrations or historical trading data
- Training without market risk or transaction costs
- Developing policies when live interaction is expensive or dangerous
- Bootstrapping learning before deploying to real markets

## Example: IQL (Implicit Q-Learning)

TorchTrade provides an example implementation of offline RL using **Implicit Q-Learning (IQL)** in `examples/offline/iql/`.

```python
# Example: Training IQL on pre-collected dataset
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

# 1. Create environment (for evaluation only)
env = SeqLongOnlyEnv(df, config)

# 2. Load pre-collected dataset
# Dataset should contain trajectories: (observation, action, reward, next_observation, done)
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=1_000_000),
)

# 3. Train IQL from offline data
for batch in replay_buffer:
    loss = iql_loss_module(batch)
    loss.backward()
    optimizer.step()
```

For a complete implementation, see [examples/offline/iql/](https://github.com/TorchTrade/torchtrade_envs/tree/main/examples/offline/iql).

## Dataset Collection

### From Offline Environments

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

# Collect trajectories with any policy (random, rule-based, pre-trained)
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=10000,
    total_frames=1_000_000,
)

# Store in replay buffer
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=1_000_000),
)

for batch in collector:
    replay_buffer.extend(batch)
```

### From Real Online Environments

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Collect real trading data (paper trading recommended)
env = AlpacaTorchTradingEnv(config)

# Record interactions
for episode in range(num_episodes):
    td = env.reset()
    while not td["done"].item():
        action = policy(td)
        td = env.step(td)
        replay_buffer.add(td)
```

## Provided Datasets

!!! todo "Coming Soon"
    We plan to provide pre-collected datasets on HuggingFace for offline RL research, including:

    - Expert demonstrations from rule-based strategies
    - Random policy trajectories for benchmarking
    - Real market interaction data (paper trading)

    Stay tuned at [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade)!

## Additional Offline RL Algorithms

TorchTrade's offline RL support is compatible with any offline RL algorithm from TorchRL, including:

- **[CQL (Conservative Q-Learning)](https://github.com/pytorch/rl/blob/main/sota-implementations/cql/cql_offline.py)** - Addresses overestimation in offline Q-learning
- **[TD3+BC](https://github.com/pytorch/rl/tree/main/sota-implementations/td3_bc)** - Combines TD3 with behavior cloning for offline learning
- **[Decision Transformer](https://github.com/pytorch/rl/tree/main/sota-implementations/decision_transformer)** - Sequence modeling approach to offline RL
- **Any TorchRL algorithm** - Use replay buffers with offline data

## Next Steps

- **[IQL Example](../examples.md#offline-training)** - Complete offline RL implementation
- **[Offline Environments](offline.md)** - Environments for dataset collection
- **[Online Environments](online.md)** - Live trading for data collection
- **[Examples](../examples.md)** - Browse all training examples

## References

- **[IQL Paper](https://arxiv.org/abs/2110.06169)** - Implicit Q-Learning algorithm
- **[TorchRL Replay Buffers](https://docs.pytorch.org/rl/main/tutorials/rb_tutorial.html)** - Data storage and sampling
- **[Offline RL Guide](https://arxiv.org/abs/2005.01643)** - Comprehensive offline RL guide
