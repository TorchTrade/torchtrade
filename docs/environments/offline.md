# Offline Environments

Offline environments are designed for **training on historical data** (backtesting). These are not "offline RL" methods like CQL or IQL, but rather environments that use pre-collected market data instead of live exchange APIs.

## Unified Architecture

TorchTrade provides **3 unified environment classes** that each support both spot and futures trading via configuration. Set `leverage=1` for spot (long-only) or `leverage>1` for futures (with margin management and liquidation mechanics). Use negative `action_levels` to enable short positions.

| Environment | Bracket Orders | One-Step | Best For |
|-------------|----------------|----------|----------|
| **SequentialTradingEnv** | ❌ | ❌ | Standard sequential trading |
| **SequentialTradingEnvSLTP** | ✅ | ❌ | Risk management with SL/TP |
| **OneStepTradingEnv** | ✅ | ✅ | GRPO, contextual bandits |

### Environment Categories

**Sequential** (`SequentialTradingEnv`) - Step-by-step trading with **fractional position sizing**. The agent can continuously adapt its position at every timestep. Action values represent the fraction of capital to deploy (e.g., 0.5 = 50% allocation).

**SL/TP** (`SequentialTradingEnvSLTP`) - Extends sequential with **bracket order risk management**. Each trade includes configurable stop-loss and take-profit trigger levels with a combinatorial action space.

**OneStep** (`OneStepTradingEnv`) - Optimized for **fast episodic training** with algorithms like [GRPO](https://arxiv.org/abs/2402.03300). The agent takes one action, and the environment internally simulates a rollout until SL/TP triggers or max rollout length. Policies can be deployed to sequential environments for step-by-step execution.

---

!!! note "Extensible Framework"
    The framework is designed to be extensible. Users can create **custom environments** by inheriting from existing base classes or implementing new ones from scratch. See [Building Custom Environments](../guides/custom-environment.md) for guidance.

### Action Space Control

All environments support the `include_hold_action` parameter (default: `True`):

- **`include_hold_action=True`** (default): Includes HOLD/no-op action in the action space
  - Standard environments: 3 actions (Sell/Short, Hold, Buy/Long)
  - SLTP environments: HOLD action + SL/TP combinations

- **`include_hold_action=False`**: Removes HOLD action, forcing the agent to always take a trading action
  - Standard environments: 2 actions (Sell/Short, Buy/Long)
  - SLTP environments: Only SL/TP combinations (no HOLD)

**Use Case**: Set to `False` when you want to ensure the agent is always actively trading rather than holding neutral positions. Useful for testing aggressive strategies or ensuring the agent explores trading actions.

### Fractional Position Sizing

**Non-SLTP environments** (`SequentialTradingEnv`) support **fractional position sizing** where action values directly represent the fraction of balance to allocate to positions.

#### How It Works

**Action Interpretation:**
- Action values range from **-1.0 to 1.0**
- **Magnitude** = fraction of balance to allocate (0.5 = 50%, 1.0 = 100%)
- **Sign** = direction (positive = long, negative = short)
- **Zero** = market neutral (close all positions, stay in cash)

**Position Sizing Formula:**
```python
# For futures environments with leverage:
position_size = (balance × |action| × leverage) / price

# For long-only environments (no leverage):
position_size = (balance × action) / price
```

**Examples:**

=== "Futures"
    ```python
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    config = SequentialTradingEnvConfig(
        leverage=5,  # Fixed 5x leverage
        action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],  # 5 discrete actions
        initial_cash=10000
    )
    env = SequentialTradingEnv(df, config)

    # Action interpretation with $10k balance, 5x leverage, $50k BTC price:
    # action = -1.0  → 100% short: (10k × 1.0 × 5) / 50k = -1.0 BTC short
    # action = -0.5  → 50% short:  (10k × 0.5 × 5) / 50k = -0.5 BTC short
    # action =  0.0  → Market neutral: 0 BTC (flat, all cash)
    # action =  0.5  → 50% long:   (10k × 0.5 × 5) / 50k = 0.5 BTC long
    # action =  1.0  → 100% long:  (10k × 1.0 × 5) / 50k = 1.0 BTC long
    ```

=== "Spot"
    ```python
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    config = SequentialTradingEnvConfig(
        action_levels=[0.0, 0.5, 1.0],  # Default: close, 50%, 100%
        initial_cash=10000
    )
    env = SequentialTradingEnv(df, config)

    # Action interpretation with $10k balance, $50k BTC price:
    # action =  0.0  → Close position (go to 100% cash)
    # action =  0.5  → 50% invested: 10k × 0.5 / 50k = 0.1 BTC
    # action =  1.0  → 100% invested: 10k × 1.0 / 50k = 0.2 BTC

    # Note: Negative actions are technically supported for backwards compatibility
    # but are not recommended as they add redundancy (behave same as action=0)
    ```

#### Customizing Action Levels

Action levels are **fully customizable**. You can specify any list of values in [-1.0, 1.0]:

```python
# Fine-grained control (9 actions)
action_levels = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

# More precision near neutral (7 actions) - asymmetric
action_levels = [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]

# Conservative (no full positions, 5 actions)
action_levels = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Long-only (buy-focused, 5 actions)
action_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

# Very coarse (3 actions)
action_levels = [-1.0, 0.0, 1.0]
```

**Default Values:**
- Futures: `[-1.0, -0.5, 0.0, 0.5, 1.0]` (5 actions: short/neutral/long)
- Long-only: `[0.0, 0.5, 1.0]` (3 actions: close/half/full invested)

Note: For long-only, action=0.0 closes the position. Negative actions are technically supported for backwards compatibility but not recommended (they behave identically to action=0.0, adding redundancy to the action space).

### Leverage Design

When `leverage > 1`, **leverage is a fixed global parameter**, not part of the action space.

**Design Philosophy:**
- **Leverage** = "How much risk am I willing to take?" (configuration/risk management)
- **Action** = "How much of my allocation should I deploy?" (learned policy)
- These are fundamentally different questions and should be separated

**Benefits:**
- Smaller action space → faster convergence
- Easy to enforce global leverage constraints
- `action=0.5` always means "use 50% of capital" regardless of leverage
- Matches real trader workflows (choose leverage once, then size positions)

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

## SequentialTradingEnv

The core sequential trading environment supporting both spot and futures trading. The trading mode is determined by configuration: set `leverage=1` (default) for spot trading, or `leverage>1` for futures with margin management and liquidation mechanics. Use negative `action_levels` to enable short positions.

### Features
- **Spot or futures**: Configured via `leverage` parameter
- **Fractional position sizing**: Action values represent fraction of capital to deploy
- **Sequential episodes**: Step-by-step trading simulation
- **Multi-timeframe observations**: Observe multiple time scales simultaneously
- **Futures mechanics** (when `leverage > 1`): Margin management, liquidation, short positions

### Configuration

=== "Spot Trading"
    ```python
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    config = SequentialTradingEnvConfig(
        # Multi-timeframe setup
        time_frames=["1min", "5min", "15min", "1hour"],
        window_sizes=[12, 8, 8, 24],
        execute_on=(5, "Minute"),

        # Spot trading (default)
        action_levels=[0.0, 0.5, 1.0],    # Close / 50% / 100% long
        initial_cash=1000,
        transaction_fee=0.0025,
        slippage=0.001,
    )

    env = SequentialTradingEnv(df, config)
    ```

=== "Futures Trading"
    ```python
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    config = SequentialTradingEnvConfig(
        # Multi-timeframe setup
        time_frames=["1min", "5min", "15min", "1hour"],
        window_sizes=[12, 8, 8, 24],
        execute_on=(5, "Minute"),

        # Futures trading
        leverage=10,                          # 10x leverage
        action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],  # Short/neutral/long
        initial_cash=10000,
        transaction_fee=0.0004,
        slippage=0.001,
    )

    env = SequentialTradingEnv(df, config)
    ```

### Observation Space

```python
observation = {
    "market_data_1Minute": Tensor([12, num_features]),    # 1m window
    "market_data_5Minute": Tensor([8, num_features]),     # 5m window
    "market_data_15Minute": Tensor([8, num_features]),    # 15m window
    "market_data_1Hour": Tensor([24, num_features]),      # 1h window
    "account_state": Tensor([6]),                         # Universal 6-element state
}

# Account state (6 elements):
# [exposure_pct, position_direction, unrealized_pnl_pct,
#  holding_time, leverage, distance_to_liquidation]
#
# Spot mode: position_direction in {0, +1}, leverage=1.0, distance_to_liquidation=1.0
# Futures mode: position_direction in {-1, 0, +1}, leverage=1-125, calculated distance
```

### Action Space

Actions are determined by `action_levels`. The number of discrete actions equals `len(action_levels)`:

- **Spot default** `[0.0, 0.5, 1.0]` → 3 actions: close / 50% long / 100% long
- **Futures default** `[-1.0, -0.5, 0.0, 0.5, 1.0]` → 5 actions: full short / half short / flat / half long / full long

### Liquidation (Futures)

When `leverage > 1`, positions are liquidated if margin is insufficient. E.g., with $10k cash at 10x leverage, a 20% loss exceeds equity and triggers liquidation.

---

## SequentialTradingEnvSLTP

Extends `SequentialTradingEnv` with **bracket order risk management**. Each trade includes configurable stop-loss and take-profit trigger levels. Supports both spot and futures modes via the same `leverage` parameter.

### Configuration

```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig

config = SequentialTradingEnvSLTPConfig(
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

env = SequentialTradingEnvSLTP(df, config)
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

---

## OneStepTradingEnv

One-step episodic environment optimized for [GRPO](https://arxiv.org/abs/2402.03300) and contextual bandit algorithms. The agent receives a randomly sampled observation from historical data, takes a single action (long/short with SL/TP or hold), and then the environment simulates a future rollout until bracket orders are triggered or the maximum rollout length is reached, completing the episode. Supports both spot and futures modes via `leverage`.

### Features
- **One-step episodes**: Single decision with episodic rollout
- **Bracket orders**: SL/TP for long and short
- **Spot or futures**: Configured via `leverage` parameter (same as `SequentialTradingEnv`)

!!! note "Deployment to Sequential Environments"
    Policies trained on `OneStepTradingEnv` can be directly deployed to `SequentialTradingEnvSLTP` for step-by-step trading, since both environments share the same observation and action spaces. This allows fast [GRPO-like](https://arxiv.org/abs/2402.03300) training followed by sequential backtesting or live trading.

### Configuration

```python
from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig

config = OneStepTradingEnvConfig(
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

env = OneStepTradingEnv(df, config)
```


---

## Visualization

All offline environments support `render_history()` to visualize episode performance:

```python
env.render_history()  # Display after running an episode
fig = env.render_history(return_fig=True)  # Or get the figure
```

- **Spot mode**: 2 subplots (price + actions, portfolio vs buy-and-hold)
- **Futures mode**: 3 subplots (adds position history)

See [Visualization Guide](visualization.md) for details.

---

## Choosing the Right Environment

### For Beginners
**→ SequentialTradingEnv** (spot mode, default)
- Simple long-only trading with fractional position sizing
- Easy to understand, good for learning RL basics

### For Risk Management Research
**→ SequentialTradingEnvSLTP**
- Stop-loss and take-profit bracket orders
- Study risk-reward trade-offs with combinatorial action spaces

### For Fast Iteration / GRPO
**→ OneStepTradingEnv**
- One-step episodes with episodic rollouts
- Fast training loops for contextual bandit-style learning

### For Advanced Futures Trading
**→ Any environment with `leverage > 1`**
- Leverage up to 125x, margin management, liquidation mechanics
- Long and short positions via negative `action_levels`

---

## Next Steps

- **[Online Environments](online.md)** - Deploy to live exchanges
- **[Feature Engineering](../guides/custom-features.md)** - Add technical indicators and custom rewards
