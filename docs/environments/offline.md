# Offline Environments

Offline environments are designed for **training on historical data** (backtesting). These are not "offline RL" methods like CQL or IQL, but rather environments that use pre-collected market data instead of live exchange APIs. For deploying trained policies to real exchanges, see [Online Environments](online.md).

## Unified Architecture

TorchTrade provides **3 unified environment classes** that each support both spot and futures trading via configuration. Set `leverage=1` for spot (long-only) or `leverage>1` for futures (with margin management and liquidation mechanics). Use negative `action_levels` to enable short positions.

| Environment | Bracket Orders | One-Step | Best For |
|-------------|----------------|----------|----------|
| **SequentialTradingEnv** | - | - | Standard sequential trading |
| **VectorizedSequentialTradingEnv** | - | - | High-throughput training (experimental) |
| **SequentialTradingEnvSLTP** | Yes | - | Risk management with SL/TP |
| **OneStepTradingEnv** | Yes | Yes | GRPO, contextual bandits |

**Sequential** (`SequentialTradingEnv`) — Step-by-step trading with **fractional position sizing**. Action values represent the fraction of capital to deploy (e.g., 0.5 = 50% allocation).

**SL/TP** (`SequentialTradingEnvSLTP`) — Extends sequential with **bracket order risk management**. Each trade includes configurable stop-loss and take-profit levels with a combinatorial action space.

**OneStep** (`OneStepTradingEnv`) — Optimized for **fast episodic training** with [GRPO](https://arxiv.org/abs/2402.03300). The agent takes one action, and the environment simulates a rollout until SL/TP triggers or max rollout length. Policies can be deployed to `SequentialTradingEnvSLTP` for step-by-step execution.

!!! note "Extensible Framework"
    Users can create **custom environments** by inheriting from existing base classes. See [Building Custom Environments](../guides/custom-environment.md).

---

### Account State

All environments expose a universal 6-element `account_state` tensor as part of the observation:

| Index | Element | Description | Spot | Futures |
|-------|---------|-------------|------|---------|
| 0 | `exposure_pct` | Position value / portfolio value | 0.0–1.0 | 0.0–N (with leverage) |
| 1 | `position_direction` | Sign of position size | 0 or +1 | -1, 0, or +1 |
| 2 | `unrealized_pnl_pct` | Unrealized P&L as % of entry price | ≥0 | Any |
| 3 | `holding_time` | Steps since position opened | ≥0 | ≥0 |
| 4 | `leverage` | Current leverage | 1.0 | 1–125 |
| 5 | `distance_to_liquidation` | Normalized distance to liquidation price | 1.0 (no risk) | Calculated |

This structure is shared across offline and online environments, ensuring policies transfer seamlessly between training and live deployment.

### Fractional Position Sizing

`SequentialTradingEnv` uses `action_levels` to define discrete fractional position sizes in [-1.0, 1.0]:

- **Magnitude** = fraction of balance to allocate (0.5 = 50%, 1.0 = 100%)
- **Sign** = direction (positive = long, negative = short, zero = flat/close)
- With leverage: position size = `balance × |action| × leverage / price`

```python
action_levels = [-1.0, 0.0, 1.0]              # Coarse: full short/flat/full long
action_levels = [0.0, 0.25, 0.5, 0.75, 1.0]   # Long-only with granularity
action_levels = [-0.5, -0.25, 0.0, 0.25, 0.5]  # Conservative, no full positions
```

**SLTP environments** use `include_hold_action` (default `True`) to optionally include a HOLD/no-op action alongside the SL/TP bracket combinations.

### Leverage

Leverage is a **fixed global parameter**, not part of the action space. `action=0.5` always means "deploy 50% of capital" regardless of leverage setting. This keeps the action space small and separates risk management (leverage) from the learned policy (position sizing).

!!! warning "Timeframe Format"
    Always use canonical forms: `["1min", "5min", "15min", "1hour", "1day"]`.
    Non-canonical forms like `"60min"` create different observation keys (`market_data_60Minute` vs `market_data_1Hour`), breaking model compatibility.

---

## SequentialTradingEnv

The core sequential trading environment. Trading mode is determined by configuration: `leverage=1` for spot, `leverage>1` for futures.

### Configuration

=== "Spot Trading"
    ```python
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    config = SequentialTradingEnvConfig(
        time_frames=["1min", "5min", "15min", "1hour"],
        window_sizes=[12, 8, 8, 24],
        execute_on=(5, "Minute"),
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
        time_frames=["1min", "5min", "15min", "1hour"],
        window_sizes=[12, 8, 8, 24],
        execute_on=(5, "Minute"),
        leverage=5,
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
    "account_state": Tensor([6]),                         # See Account State above
}
```

### Liquidation (Futures)

When `leverage > 1`, positions are liquidated if margin is insufficient. E.g., with $10k cash at 10x leverage, a 20% loss exceeds equity and triggers liquidation.

### Vectorized Version (Experimental)

`VectorizedSequentialTradingEnv` is a batched tensor implementation of `SequentialTradingEnv` that processes N environments in a single `_step()` call using pure tensor operations. It achieves **20-400x higher throughput** compared to `ParallelEnv` by eliminating inter-process communication overhead.

!!! warning "Experimental"
    This environment is still experimental. While it passes extensive scalar-vectorized equivalence tests, it has not been battle-tested in production training runs. Use with caution and verify results against the standard `SequentialTradingEnv`.

```python
from torchtrade.envs.offline import VectorizedSequentialTradingEnv, VectorizedSequentialTradingEnvConfig

config = VectorizedSequentialTradingEnvConfig(
    num_envs=64,
    time_frames=["1min", "5min", "15min", "1hour"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),
    action_levels=[0.0, 0.5, 1.0],
    initial_cash=1000,
    transaction_fee=0.0025,
    leverage=1,  # or >1 for futures
)

env = VectorizedSequentialTradingEnv(df, config)
```

See the [PPO Vectorized example](https://github.com/TorchTrade/torchtrade/tree/main/examples/online_rl/ppo_vectorized) for a complete training setup.

---

## SequentialTradingEnvSLTP

Extends `SequentialTradingEnv` with **bracket order risk management**. Supports both spot and futures modes via `leverage`.

### Configuration

```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig

config = SequentialTradingEnvSLTPConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_hold_action=True,

    # Futures parameters (leverage > 1 enables short bracket orders)
    leverage=5,
    margin_call_threshold=0.2,

    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=10000,
    transaction_fee=0.0004,
    slippage=0.001,
)

env = SequentialTradingEnvSLTP(df, config)
```

### Action Space

With 2 SL levels, 2 TP levels, and `leverage > 1` (futures):

- **Action 0**: HOLD / Close position
- **Actions 1-4**: LONG with SL/TP combinations
- **Actions 5-8**: SHORT with SL/TP combinations

Formula: `1 + 2 × (num_sl × num_tp)` = **9 actions**. Without HOLD (`include_hold_action=False`): `2 × (num_sl × num_tp)` = **8 actions**.

---

## OneStepTradingEnv

One-step episodic environment for [GRPO](https://arxiv.org/abs/2402.03300) and contextual bandits. The agent takes a single action, and the environment simulates a rollout until SL/TP triggers or max rollout length. Supports spot and futures via `leverage`.

!!! note "Deployment"
    Policies trained on `OneStepTradingEnv` can be deployed directly to `SequentialTradingEnvSLTP` — both share the same observation and action spaces.

### Configuration

```python
from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig

config = OneStepTradingEnvConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    include_hold_action=True,
    rollout_steps=24,

    # Futures parameters (leverage > 1 enables short bracket orders)
    leverage=5,
    margin_call_threshold=0.2,

    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
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

All environments render 3 subplots: price + actions, portfolio vs buy-and-hold, and exposure history.

See [Visualization Guide](visualization.md) for details.

---

## Next Steps

- **[Online Environments](online.md)** - Deploy to live exchanges
- **[Feature Engineering](../guides/custom-features.md)** - Add technical indicators and custom rewards
