# Offline Environments

Offline environments are designed for **training on historical data** (backtesting). These are not "offline RL" methods like CQL or IQL, but rather environments that use pre-collected market data instead of live exchange APIs.

## Overview

TorchTrade provides 6 offline environments:

| Environment | Asset Type | Futures | Leverage | Bracket Orders | One-Step |
|-------------|------------|---------|----------|----------------|----------|
| **SeqLongOnlyEnv** | Crypto/Stocks | ❌ | ❌ | ❌ | ❌ |
| **SeqLongOnlySLTPEnv** | Crypto/Stocks | ❌ | ❌ | ✅ | ❌ |
| **LongOnlyOneStepEnv** | Crypto/Stocks | ❌ | ❌ | ✅ | ✅ |
| **SeqFuturesEnv** | Crypto | ✅ | ✅ | ❌ | ❌ |
| **SeqFuturesSLTPEnv** | Crypto | ✅ | ✅ | ✅ | ❌ |
| **FuturesOneStepEnv** | Crypto | ✅ | ✅ | ✅ | ✅ |

---

## SeqLongOnlyEnv

Simple long-only spot trading environment for sequential RL algorithms.

### Features
- **Long-only trading**: Can only buy and hold (no short positions)
- **Sequential episodes**: Step-by-step trading simulation
- **3-action discrete space**: SELL (0), HOLD (1), BUY (2)
- **Multi-timeframe observations**: Observe multiple time scales simultaneously

### Configuration

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

config = SeqLongOnlyEnvConfig(
    # Multi-timeframe setup
    time_frames=["1min", "5min", "15min", "60min"],        # Values in minutes: "1min", "5min", "15min", "60min"
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute"),          # Execute every 5 minutes

    # Trading parameters
    initial_cash=1000,                 # Starting capital
    transaction_fee=0.0025,            # 0.25% per trade
    slippage=0.001,                    # 0.1% slippage

    # Episode configuration
    max_traj_length=None,              # Full dataset (or set limit)

    # Feature preprocessing (optional)
    feature_preprocessing_fn=None,      # Custom feature function

    # Reward function (optional)
    reward_function=None,               # Use default log return
)

# Create environment
env = SeqLongOnlyEnv(df, config)
```

### Observation Space

```python
observation = {
    "market_data_1Minute": Tensor([12, num_features]),    # 1m window
    "market_data_5Minute": Tensor([8, num_features]),     # 5m window
    "market_data_15Minute": Tensor([8, num_features]),    # 15m window
    "market_data_60Minute": Tensor([24, num_features]),   # 1h window
    "account_state": Tensor([7]),                         # Account state
}

# Account state (7 elements):
# [cash, position_size, position_value, entry_price,
#  current_price, unrealized_pnl_pct, holding_time]
```

### Action Space

Discrete(3):
- **Action 0**: SELL - Close current position
- **Action 1**: HOLD - Do nothing
- **Action 2**: BUY - Open position (100% of cash)

### Example Usage

```python
import pandas as pd
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Load data
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Configure environment
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlyEnv(df, config)

# Run episode
tensordict = env.reset()
done = False

while not done:
    # Your policy selects action
    action = policy(tensordict)  # Returns action in [0, 1, 2]
    tensordict["action"] = action

    # Step environment
    tensordict = env.step(tensordict)
    done = tensordict["done"].item()

    print(f"Reward: {tensordict['reward'].item():.4f}")
```

---

## SeqLongOnlySLTPEnv

Long-only environment with stop-loss and take-profit bracket orders.

### Features
- All features of `SeqLongOnlyEnv`
- **Bracket orders**: Each buy includes SL and TP levels
- **Combinatorial action space**: HOLD + (num_sl × num_tp) buy actions
- **Automatic position management**: Orders close on SL/TP trigger

### Configuration

```python
from torchtrade.envs.offline import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    # Stop-loss / Take-profit levels
    stoploss_levels=[-0.02, -0.05],     # -2%, -5%
    takeprofit_levels=[0.05, 0.10],     # +5%, +10%
    include_hold_action=True,           # Optional: set False to remove HOLD

    # Multi-timeframe setup (same as SeqLongOnlyEnv)
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),

    # Trading parameters
    initial_cash=1000,
    transaction_fee=0.0025,
    slippage=0.001,
)

env = SeqLongOnlySLTPEnv(df, config)
```

### Action Space

Discrete(1 + num_sl × num_tp) when `include_hold_action=True`, or Discrete(num_sl × num_tp) when `False`.

With `stoploss_levels=[-0.02, -0.05]` and `takeprofit_levels=[0.05, 0.10]` and `include_hold_action=True`:

- **Action 0**: HOLD / Close position
- **Action 1**: BUY with SL=-2%, TP=+5%
- **Action 2**: BUY with SL=-2%, TP=+10%
- **Action 3**: BUY with SL=-5%, TP=+5%
- **Action 4**: BUY with SL=-5%, TP=+10%

Total: 1 + (2 × 2) = **5 actions**

### Example Usage

```python
from torchtrade.envs.offline import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05, -0.10],   # 3 SL levels
    takeprofit_levels=[0.05, 0.10, 0.15],    # 3 TP levels
    include_hold_action=True,                # Optional: set False to remove HOLD
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlySLTPEnv(df, config)

# Action space: 1 + (3 × 3) = 10 actions (or 9 without HOLD)
print(f"Action space size: {env.action_spec.space.n}")
```

---

## LongOnlyOneStepEnv

One-step episodic environment optimized for GRPO and contextual bandit algorithms.

### Features
- **One-step episodes**: Each episode is a single decision point
- **Episodic rollout**: Action taken, then position held for fixed duration
- **Fast iteration**: Train policies quickly with episodic rollouts
- **Bracket orders**: Includes SL/TP like `SeqLongOnlySLTPEnv`

### Configuration

```python
from torchtrade.envs.offline import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig

config = LongOnlyOneStepEnvConfig(
    # Stop-loss / Take-profit
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_hold_action=True,           # Optional: set False to remove HOLD

    # Multi-timeframe setup
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),

    # Rollout configuration
    rollout_steps=24,                   # Hold position for 24 steps (2 hours)

    # Trading parameters
    initial_cash=1000,
    transaction_fee=0.0025,
    slippage=0.001,
)

env = LongOnlyOneStepEnv(df, config)
```

### How It Works

1. **Reset**: Environment starts at random market state
2. **Action**: Agent selects one action (HOLD or BUY with SL/TP)
3. **Rollout**: Position held for `rollout_steps`, tracking PnL
4. **Reward**: Terminal reward computed from rollout returns
5. **Done**: Episode ends after one action

### Example Usage

```python
from torchtrade.envs.offline import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig

config = LongOnlyOneStepEnvConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    rollout_steps=24,  # 2 hours (24 × 5min)
    initial_cash=1000
)

env = LongOnlyOneStepEnv(df, config)

# Run episode
tensordict = env.reset()
action = policy(tensordict)  # Select action once
tensordict["action"] = action
tensordict = env.step(tensordict)  # Episode immediately done

print(f"Terminal reward: {tensordict['reward'].item()}")
print(f"Done: {tensordict['done'].item()}")  # Always True
```

---

## SeqFuturesEnv

Futures trading environment with leverage, margin management, and liquidation mechanics.

### Features
- **Futures trading**: Long and short positions
- **Leverage support**: 1-125x leverage
- **Margin management**: Tracks margin ratio and liquidation price
- **Liquidation mechanics**: Positions liquidated if margin insufficient
- **3-action discrete**: SHORT (0), HOLD (1), LONG (2)

### Configuration

```python
from torchtrade.envs.offline import SeqFuturesEnv, SeqFuturesEnvConfig

config = SeqFuturesEnvConfig(
    # Multi-timeframe setup
    time_frames=["1min", "5min", "15min", "60min"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),

    # Futures parameters
    leverage=10,                        # 10x leverage
    margin_call_threshold=0.2,          # Liquidate at 20% margin ratio

    # Trading parameters
    initial_cash=10000,
    transaction_fee=0.0004,             # 0.04% (futures have lower fees)
    slippage=0.001,

    # Feature preprocessing (optional)
    feature_preprocessing_fn=None,

    # Reward function (optional)
    reward_function=None,
)

env = SeqFuturesEnv(df, config)
```

### Observation Space

```python
observation = {
    "market_data_1Minute": Tensor([12, num_features]),
    "market_data_5Minute": Tensor([8, num_features]),
    "market_data_15Minute": Tensor([8, num_features]),
    "market_data_60Minute": Tensor([24, num_features]),
    "account_state": Tensor([10]),  # 10 elements (futures-specific)
}

# Account state (10 elements):
# [cash, position_size, position_value, entry_price, current_price,
#  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
```

### Action Space

Discrete(3):
- **Action 0**: SHORT - Open short position with leverage
- **Action 1**: HOLD - Do nothing or maintain position
- **Action 2**: LONG - Open long position with leverage

### Liquidation Mechanics

Positions are liquidated when:
- **Margin ratio < margin_call_threshold**
- Margin ratio = (equity / position_value)
- Equity = cash + unrealized PnL

Example:
```
Initial cash: $10,000
Leverage: 10x
Position value: $100,000 (10x leverage)
Initial margin: $10,000

If position loses 20%:
Unrealized PnL: -$20,000
Equity: $10,000 - $20,000 = -$10,000 (liquidated!)
```

### Example Usage

```python
from torchtrade.envs.offline import SeqFuturesEnv, SeqFuturesEnvConfig

config = SeqFuturesEnvConfig(
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    leverage=5,                         # 5x leverage
    margin_call_threshold=0.2,          # 20% margin ratio
    initial_cash=10000
)

env = SeqFuturesEnv(df, config)

# Run episode
tensordict = env.reset()

while not tensordict["done"].item():
    action = policy(tensordict)
    tensordict["action"] = action
    tensordict = env.step(tensordict)

    # Check account state
    account = tensordict["account_state"]
    margin_ratio = account[7].item()

    if margin_ratio < 0.3:
        print(f"⚠️ Warning: Low margin ratio {margin_ratio:.2f}")
```

---

## SeqFuturesSLTPEnv

Futures environment with stop-loss/take-profit bracket orders.

### Features
- All features of `SeqFuturesEnv`
- **Bracket orders**: SL/TP for both long and short positions
- **Combinatorial action space**: HOLD + (long SL/TP) + (short SL/TP)

### Configuration

```python
from torchtrade.envs.offline import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig

config = SeqFuturesSLTPEnvConfig(
    # Stop-loss / Take-profit
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_hold_action=True,            # Optional: set False to remove HOLD
    include_short_positions=True,        # Enable short bracket orders

    # Futures parameters
    leverage=10,
    margin_call_threshold=0.2,

    # Multi-timeframe setup
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),

    # Trading parameters
    initial_cash=10000,
    transaction_fee=0.0004,
    slippage=0.001,
)

env = SeqFuturesSLTPEnv(df, config)
```

### Action Space

With `include_short_positions=True`:

Discrete(1 + 2 × (num_sl × num_tp)):

Example with 2 SL levels and 2 TP levels:
- **Action 0**: HOLD / Close position
- **Actions 1-4**: LONG with SL/TP combinations
- **Actions 5-8**: SHORT with SL/TP combinations

Total: 1 + 2 × (2 × 2) = **9 actions**

### Example Usage

```python
from torchtrade.envs.offline import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig

config = SeqFuturesSLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_short_positions=True,
    leverage=10,
    margin_call_threshold=0.2,
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=10000
)

env = SeqFuturesSLTPEnv(df, config)
print(f"Action space size: {env.action_spec.space.n}")  # 9 actions
```

---

## FuturesOneStepEnv

One-step episodic futures environment for GRPO training.

### Features
- Combines `SeqFuturesEnv` + `LongOnlyOneStepEnv`
- **One-step episodes**: Single decision with episodic rollout
- **Futures with leverage**: Up to 125x leverage
- **Bracket orders**: SL/TP for long and short

### Configuration

```python
from torchtrade.envs.offline import FuturesOneStepEnv, FuturesOneStepEnvConfig

config = FuturesOneStepEnvConfig(
    # Stop-loss / Take-profit
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_hold_action=True,            # Optional: set False to remove HOLD
    include_short_positions=True,

    # Futures parameters
    leverage=10,
    margin_call_threshold=0.2,

    # Rollout configuration
    rollout_steps=24,

    # Multi-timeframe setup
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),

    # Trading parameters
    initial_cash=10000,
    transaction_fee=0.0004,
)

env = FuturesOneStepEnv(df, config)
```

### Example Usage

```python
from torchtrade.envs.offline import FuturesOneStepEnv, FuturesOneStepEnvConfig

config = FuturesOneStepEnvConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_short_positions=True,
    leverage=5,
    rollout_steps=24,  # Hold for 2 hours
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=10000
)

env = FuturesOneStepEnv(df, config)

# Each episode is one decision
tensordict = env.reset()
action = policy(tensordict)
tensordict["action"] = action
tensordict = env.step(tensordict)  # Done immediately

print(f"Terminal reward: {tensordict['reward'].item()}")
```

---

## Choosing the Right Environment

### For Beginners
**→ SeqLongOnlyEnv**
- Simple long-only trading
- Easy to understand
- Good for learning RL basics

### For Risk Management Research
**→ SeqLongOnlySLTPEnv** or **SeqFuturesSLTPEnv**
- Stop-loss and take-profit orders
- Study risk-reward trade-offs
- Combinatorial action spaces

### For Fast Iteration / GRPO
**→ LongOnlyOneStepEnv** or **FuturesOneStepEnv**
- One-step episodes
- Fast training loops
- Episodic rollouts

### For Advanced Futures Trading
**→ SeqFuturesEnv** or **SeqFuturesSLTPEnv**
- Leverage up to 125x
- Margin management
- Liquidation mechanics
- Long and short positions

---

## Next Steps

- **[Online Environments](online.md)** - Live trading with exchange APIs
- **[Loss Functions](../components/losses.md)** - Training objectives (GRPOLoss for OneStepEnv, CTRL for representation learning)
- **[Transforms](../components/transforms.md)** - Data preprocessing (ChronosEmbeddingTransform, CoverageTracker)
- **[Actors](../components/actors.md)** - Alternative policies (RuleBasedActor for baselines, LLMActor for LLM trading)
- **[Feature Engineering](../guides/custom-features.md)** - Add technical indicators
- **[Reward Functions](../guides/reward-functions.md)** - Design better rewards
- **[Understanding the Sampler](../guides/sampler.md)** - How data sampling works
