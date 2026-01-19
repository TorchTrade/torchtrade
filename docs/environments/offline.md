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

### Action Space Control

All environments support the `include_hold_action` parameter (default: `True`):

- **`include_hold_action=True`** (default): Includes HOLD/no-op action in the action space
  - Standard environments: 3 actions (Sell/Short, Hold, Buy/Long)
  - SLTP environments: HOLD action + SL/TP combinations

- **`include_hold_action=False`**: Removes HOLD action, forcing the agent to always take a trading action
  - Standard environments: 2 actions (Sell/Short, Buy/Long)
  - SLTP environments: Only SL/TP combinations (no HOLD)

**Use Case**: Set to `False` when you want to ensure the agent is always actively trading rather than holding neutral positions. Useful for testing aggressive strategies or ensuring the agent explores trading actions.

!!! warning "Timeframe Format - Critical for Model Compatibility"
    When specifying `time_frames`, **always use canonical forms**:

    - ✅ **Correct**: `["1min", "5min", "15min", "1hour", "1day"]`
    - ❌ **Wrong**: `["1min", "5min", "15min", "60min"]`

    **Why this matters:**

    - `time_frames=["60min"]` creates observation key `"market_data_60Minute"`
    - `time_frames=["1hour"]` creates observation key `"market_data_1Hour"`
    - These are **DIFFERENT keys** - your model trained with `"60min"` won't work with config using `"1hour"`

    The framework will issue a warning if you use non-canonical forms. Use the suggested canonical forms to ensure model compatibility and cleaner observation keys.

    **Common conversions:**

    - `60min` → use `1hour`
    - `120min` → use `2hours`
    - `1440min` → use `1day`
    - `24hour` → use `1day`

---

## SeqLongOnlyEnv

Simple long-only spot trading environment for sequential RL algorithms. This is the offline backtesting equivalent of the [AlpacaTorchTradingEnv](online.md#alpacatorchtradingenv) live environment, designed for training on historical data before deploying to live stock and crypto markets.

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
    time_frames=["1min", "5min", "15min", "1hour"],        # Values: "1min", "5min", "15min", "1hour", "1day"
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute"),          # Execute every 5 minutes

    # Trading parameters
    initial_cash=1000,                 # Starting capital
    transaction_fee=0.0025,            # 0.25% per trade
    slippage=0.001,                    # 0.1% slippage

    # Episode configuration
    max_traj_length=None,              # Full dataset (or set limit)
    random_start=True,                 # True = random episode starts, False = sequential
    include_hold_action=True,          # Include HOLD action (default: True)

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
    "market_data_1Hour": Tensor([24, num_features]),      # 1h window
    "account_state": Tensor([7]),                         # Account state
}

# Account state (7 elements):
# [cash, position_size, position_value, entry_price,
#  current_price, unrealized_pnl_pct, holding_time]
```

### Action Space

**Default (include_hold_action=True)** - Discrete(3):
- **Action 0**: SELL - Close current position
- **Action 1**: HOLD - Do nothing
- **Action 2**: BUY - Open position (100% of cash)

**Without HOLD (include_hold_action=False)** - Discrete(2):
- **Action 0**: SELL - Close current position
- **Action 1**: BUY - Open position (100% of cash)

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

Long-only environment with stop-loss and take-profit bracket orders. This is the offline backtesting equivalent of the [AlpacaSLTPTorchTradingEnv](online.md#alpacasltptorchtradingenv) live environment, adding risk management through bracket orders for stock and crypto trading.

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

**Formula**:
- **With HOLD (include_hold_action=True)**: Discrete(1 + num_sl × num_tp)
- **Without HOLD (include_hold_action=False)**: Discrete(num_sl × num_tp)

**Example with include_hold_action=True**:

With `stoploss_levels=[-0.02, -0.05]` and `takeprofit_levels=[0.05, 0.10]`:

- **Action 0**: HOLD / Close position
- **Action 1**: BUY with SL=-2%, TP=+5%
- **Action 2**: BUY with SL=-2%, TP=+10%
- **Action 3**: BUY with SL=-5%, TP=+5%
- **Action 4**: BUY with SL=-5%, TP=+10%

Total: 1 + (2 × 2) = **5 actions**

**Example with include_hold_action=False**:

Same levels but no HOLD action:

- **Action 0**: BUY with SL=-2%, TP=+5%
- **Action 1**: BUY with SL=-2%, TP=+10%
- **Action 2**: BUY with SL=-5%, TP=+5%
- **Action 3**: BUY with SL=-5%, TP=+10%

Total: 2 × 2 = **4 actions**

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

One-step episodic environment optimized for [GRPO](https://arxiv.org/abs/2402.03300) and contextual bandit algorithms. The agent receives a randomly sampled observation from historical data, takes a single action, and then the environment simulates a future rollout until bracket orders are triggered (stop-loss or take-profit hit) or the maximum rollout length is reached, completing the episode.

### Features
- **One-step episodes**: Each episode is a single decision point
- **Episodic rollout**: Action taken, then position held for fixed duration
- **Fast iteration**: Train policies quickly with episodic rollouts
- **Bracket orders**: Includes SL/TP like `SeqLongOnlySLTPEnv`

!!! note "Deployment to Sequential Environments"
    Policies trained on `LongOnlyOneStepEnv` can be directly deployed to `SeqLongOnlySLTPEnv` for step-by-step trading, since both environments share the same observation and action spaces. This allows fast [GRPO-like](https://arxiv.org/abs/2402.03300) training followed by sequential backtesting or live trading.

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

Futures trading environment with leverage, margin management, and liquidation mechanics. This is the offline backtesting equivalent of the [BinanceFuturesTorchTradingEnv](online.md#binancefuturestorchtradingenv) and [BitgetFuturesTorchTradingEnv](online.md#bitgetfuturestorchtradingenv) live environments, designed for training futures trading strategies on historical data before deploying to live crypto markets.

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
    time_frames=["1min", "5min", "15min", "1hour"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),

    # Futures parameters
    leverage=10,                        # 10x leverage
    margin_call_threshold=0.2,          # Liquidate at 20% margin ratio

    # Trading parameters
    initial_cash=10000,
    transaction_fee=0.0004,             # 0.04% (futures have lower fees)
    slippage=0.001,
    include_hold_action=True,           # Include HOLD action (default: True)

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
    "market_data_1Hour": Tensor([24, num_features]),
    "account_state": Tensor([10]),  # 10 elements (futures-specific)
}

# Account state (10 elements):
# [cash, position_size, position_value, entry_price, current_price,
#  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
```

### Action Space

**Default (include_hold_action=True)** - Discrete(3):
- **Action 0**: SHORT - Open short position with leverage
- **Action 1**: HOLD - Do nothing or maintain position
- **Action 2**: LONG - Open long position with leverage

**Without HOLD (include_hold_action=False)** - Discrete(2):
- **Action 0**: SHORT - Open short position with leverage
- **Action 1**: LONG - Open long position with leverage

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

Futures environment with stop-loss/take-profit bracket orders. This is the offline backtesting equivalent of the [BinanceFuturesSLTPTorchTradingEnv](online.md#binancefuturessltptorchtradingenv) and [BitgetFuturesSLTPTorchTradingEnv](online.md#bitgetfuturessltptorchtradingenv) live environments, adding risk management through bracket orders for futures trading strategies.

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

**Formula** (with `include_short_positions=True`):
- **With HOLD (include_hold_action=True)**: Discrete(1 + 2 × (num_sl × num_tp))
- **Without HOLD (include_hold_action=False)**: Discrete(2 × (num_sl × num_tp))

**Example with include_hold_action=True**:

With 2 SL levels and 2 TP levels:
- **Action 0**: HOLD / Close position
- **Actions 1-4**: LONG with SL/TP combinations
- **Actions 5-8**: SHORT with SL/TP combinations

Total: 1 + 2 × (2 × 2) = **9 actions**

**Example with include_hold_action=False**:

Same levels but no HOLD action:
- **Actions 0-3**: LONG with SL/TP combinations
- **Actions 4-7**: SHORT with SL/TP combinations

Total: 2 × (2 × 2) = **8 actions**

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

One-step episodic futures environment optimized for [GRPO](https://arxiv.org/abs/2402.03300) training. The agent receives a randomly sampled observation from historical data, takes a single action (long/short with SL/TP or hold), and then the environment simulates a future rollout with leverage and margin mechanics until bracket orders are triggered or the maximum rollout length is reached, completing the episode.

### Features
- Combines `SeqFuturesEnv` + `LongOnlyOneStepEnv`
- **One-step episodes**: Single decision with episodic rollout
- **Futures with leverage**: Up to 125x leverage
- **Bracket orders**: SL/TP for long and short

!!! note "Deployment to Sequential Environments"
    Policies trained on `FuturesOneStepEnv` can be directly deployed to `SeqFuturesSLTPEnv` for step-by-step trading, since both environments share the same observation and action spaces. This allows fast [GRPO-like](https://arxiv.org/abs/2402.03300) training followed by sequential backtesting or live trading.

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

## State Management

All TorchTrade environments (both offline and online) use structured state management through dedicated dataclasses found in `torchtrade/envs/state.py`.

### PositionState

The `PositionState` dataclass encapsulates all position-related state in a single object:

```python
from torchtrade.envs.state import PositionState

# Used internally by all environments
position = PositionState()

# Available attributes:
position.current_position  # 0=no position, 1=long, -1=short
position.position_size     # Number of units held (negative for shorts)
position.position_value    # Current market value of the position
position.entry_price       # Price at which position was entered
position.unrealized_pnlpc  # Unrealized P&L as percentage
position.hold_counter      # Number of steps position has been held

# Reset all fields at once
position.reset()
```

**Benefits**:
- Groups related state variables together
- Provides single `.reset()` method for all position fields
- Makes position state explicit and easier to track
- Used consistently across all TorchTrade environments (offline and online)

See `torchtrade/envs/state.py:8-30` for implementation details.

### HistoryTracker

The `HistoryTracker` class records episode data for analysis and visualization:

```python
from torchtrade.envs.state import HistoryTracker, FuturesHistoryTracker

# For long-only environments
history = HistoryTracker()

# During episode
history.record_step(
    price=50000.0,
    action=1.0,
    reward=0.05,
    portfolio_value=10500.0
)

# Export for plotting/analysis
data = history.to_dict()
# Returns: {'base_prices': [...], 'actions': [...], 'rewards': [...], 'portfolio_values': [...]}

# Reset for new episode
history.reset()
```

**FuturesHistoryTracker** extends `HistoryTracker` with position tracking:

```python
# For futures environments (SeqFuturesEnv, etc.)
history = FuturesHistoryTracker()

history.record_step(
    price=50000.0,
    action=2.0,
    reward=0.03,
    portfolio_value=10300.0,
    position=0.5  # Positive=long, negative=short
)

# Export includes position history
data = history.to_dict()
# Returns: {..., 'positions': [0.5, ...]}
```

**Use Cases**:
- Plot portfolio value over time
- Analyze action distributions
- Track position holding patterns
- Debug environment behavior
- Generate performance metrics

**Note**: History tracking is available in both offline environments (SeqLongOnlyEnv, SeqFuturesEnv, etc.) and online environments (AlpacaTorchTradingEnv, BinanceFuturesTorchTradingEnv, etc.).

See `torchtrade/envs/state.py:33-148` for implementation details.

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
