# DQN Futures Trading Example

This example demonstrates Deep Q-Network (DQN) and Trading DQN (TDQN) training on the SeqFuturesEnv environment for cryptocurrency futures trading with leverage support.

## Network Types

### 1. **Standard DQN** (`network_type: dqn`)
Traditional DQN that takes market observations as input.

### 2. **TDQN - Trading DQN** (`network_type: tdqn`)
Position-aware Q-network specifically designed for trading. Takes both market observations AND current trading position as inputs, making it explicitly aware of whether you're long/short/flat when making decisions.

## Overview

DQN is an off-policy reinforcement learning algorithm that uses:
- **Experience Replay**: Stores past experiences in a replay buffer for stable learning
- **Target Networks**: Uses a separate target network with delayed updates to stabilize Q-value estimation
- **Epsilon-Greedy Exploration**: Gradually transitions from exploration to exploitation

## Key Features

- **Discrete Actions**: Buy, Sell, or Hold positions
- **Multi-timeframe Market Data**: Uses CNN encoders for different timeframes (1Min, 5Min)
- **Account State Tracking**: Monitors cash, positions, leverage, margin, and P&L
- **Coverage Tracking**: Monitors training data coverage and diversity

## Algorithm Details

### Off-Policy Learning
Unlike PPO (on-policy), DQN learns from a replay buffer of past experiences, allowing for better sample efficiency and stability.

### Epsilon-Greedy Exploration
- Starts with `eps_start=1.0` (100% random actions)
- Anneals to `eps_end=0.05` over `annealing_frames=250,000` frames
- Ensures proper exploration during early training

### Target Network Updates
- Uses hard updates every `hard_update_freq=50` gradient steps
- Provides stable Q-value targets during training

## Configuration

### Choosing Network Type

```yaml
model:
  network_type: tdqn        # "dqn" or "tdqn"
  activation: relu
  hidden_size: 128          # TDQN hidden layer size
  dropout: 0.2              # TDQN dropout rate
```

### Key Hyperparameters

Key hyperparameters in `config.yaml`:

```yaml
collector:
  init_random_frames: 10_000     # Random exploration before training
  eps_start: 1.0                 # Initial epsilon
  eps_end: 0.05                  # Final epsilon
  annealing_frames: 250_000      # Epsilon annealing duration

buffer:
  buffer_size: 100_000           # Replay buffer capacity
  batch_size: 128                # Training batch size

loss:
  gamma: 0.99                    # Discount factor
  hard_update_freq: 50           # Target network update frequency
  num_updates: 100               # Gradient updates per data collection
```

## Running the Example

```bash
# Basic run with TDQN (default)
python examples/online/dqn_futures/train.py

# Use standard DQN instead
python examples/online/dqn_futures/train.py model.network_type=dqn

# Override config parameters
python examples/online/dqn_futures/train.py \
    model.network_type=tdqn \
    env.leverage=10 \
    optim.lr=1e-3 \
    buffer.buffer_size=200_000

# Tune TDQN architecture
python examples/online/dqn_futures/train.py \
    model.network_type=tdqn \
    model.hidden_size=256 \
    model.dropout=0.3

# Disable logging
python examples/online/dqn_futures/train.py logger.backend=

# Enable compilation for faster training
python examples/online/dqn_futures/train.py compile.compile=True
```

## Model Architecture

```
Market Data (1Min, 5Min) --> CNN Encoders --> Encodings
Account State --> MLP Encoder --> Encoding
                                    |
                                    v
                        Concatenated Features
                                    |
                                    v
                            Q-Value MLP (128->128->3)
                                    |
                                    v
                        Q-Values for [Buy, Sell, Hold]
                                    |
                                    v
                            QValueActor (Greedy Action Selection)
                                    |
                                    v
                            EGreedyModule
                        (Epsilon-Greedy Exploration)
```

### Key Components

1. **Encoder**: CNN encoders for market data + MLP for account state
2. **Q-Value Network**: MLP that outputs Q-values for each action
3. **QValueActor**: Selects greedy action (argmax Q-value)
4. **EGreedyModule**: Applies epsilon-greedy exploration
5. **DQNLoss**: Computes temporal difference error with target network

## DQN vs TDQN Comparison

| Feature | Standard DQN | TDQN (Trading DQN) |
|---------|-------------|-------------------|
| Input | Observation only | Observation + Position |
| Position Awareness | Implicit | Explicit |
| Architecture | 2-layer MLP | 5-layer MLP with BatchNorm |
| Regularization | None | Dropout (0.2) |
| Training Stability | Moderate | Higher (BatchNorm helps) |
| Trading Suitability | Good | Better (position-aware) |

**When to use TDQN**: When current trading position significantly affects optimal actions (e.g., risk management, position sizing).

**When to use DQN**: For simpler problems or when position information is already encoded in observations.

## Comparison with PPO

| Feature | PPO (On-Policy) | DQN/TDQN (Off-Policy) |
|---------|----------------|----------------------|
| Replay Buffer | No | Yes |
| Sample Efficiency | Lower | Higher |
| Exploration | Stochastic Policy | Epsilon-Greedy |
| Target Network | No | Yes |
| Advantage Estimation | GAE | Q-Learning |
| Training Stability | Generally Stable | Requires Tuning |

## Expected Behavior

- **Early Training**: High epsilon leads to random exploration
- **Mid Training**: Epsilon decreases, agent starts exploiting learned Q-values
- **Late Training**: Low epsilon, mostly greedy action selection

Monitor these metrics:
- `train/epsilon`: Should decrease from 1.0 to 0.05
- `train/loss`: Should stabilize after initial fluctuations
- `eval/reward`: Should improve over time
- `train/reset_coverage`: Training data diversity (should be high for overfitting)

## Troubleshooting

### High Loss Values
- Increase `init_random_frames` for better initial buffer
- Decrease learning rate `optim.lr`
- Increase `hard_update_freq` for more stable targets

### Poor Exploration
- Increase `eps_start` or `eps_end`
- Extend `annealing_frames`
- Increase `init_random_frames`

### Slow Training
- Decrease `num_updates` per collection
- Enable compilation with `compile.compile=True`
- Reduce `buffer.buffer_size` if memory-constrained

## References

- [DQN Paper](https://arxiv.org/abs/1312.5602): "Playing Atari with Deep Reinforcement Learning"
- [TorchRL DQN Tutorial](https://pytorch.org/rl/tutorials/coding_dqn.html)
- [TorchRL Documentation](https://pytorch.org/rl/)
