# TorchTrade Documentation

Welcome to the TorchTrade documentation! TorchTrade is a machine learning framework for algorithmic trading built on TorchRL.

TorchTrade's goal is to provide accessible deployment of RL methods to trading. The framework supports various RL methodologies including **online RL**, **offline RL**, **model-based RL**, **contrastive learning**, and many more areas of reinforcement learning research. Beyond RL, TorchTrade integrates traditional trading methods such as **rule-based strategies**, as well as modern approaches including **LLMs** (both local models and frontier model integrations) as trading actors.

## What is TorchTrade?

TorchTrade provides modular environments for both live trading with major exchanges and offline backtesting. The framework supports:

- 🎯 **Multi-Timeframe Observations** - Train on 1m, 5m, 15m, 1h bars simultaneously
- 🤖 **Multiple RL Algorithms** - PPO, DQN, IQL, GRPO, DSAC implementations
- 📊 **Feature Engineering** - Add technical indicators and custom features
- 🔴 **Live Trading** - Direct API integration with major exchanges
- 📉 **Risk Management** - Stop-loss/take-profit, margin, leverage, liquidation mechanics
- 🔮 **Futures Trading** - Up to 125x leverage with proper margin management
- 📦 **Ready-to-Use Datasets** - Pre-processed OHLCV data available at [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade)

## Quick Navigation

### Getting Started
- **[Installation & Setup](getting-started.md)** - Get up and running in minutes
- **[First Environment](getting-started.md#your-first-environment)** - Create and run your first trading environment
- **[First Training Run](getting-started.md#training-your-first-policy)** - Train a PPO policy

### Environments
- **[Offline Environments](environments/offline.md)** - Backtesting with historical data
  - SequentialTradingEnv, SequentialTradingEnvSLTP, OneStepTradingEnv
- **[Online Environments](environments/online.md)** - Live trading with exchange APIs
  - Alpaca, Binance, Bitget integrations

### Components
- **[Loss Functions](components/losses.md)** - Training objectives (GRPOLoss, CTRLLoss, CTRLPPOLoss)
- **[Transforms](components/transforms.md)** - Data preprocessing (CoverageTracker, ChronosEmbeddingTransform)
- **[Actors](components/actors.md)** - Trading policies (RuleBasedActor, LLMActor, LocalLLMActor)

### Advanced Customization
- **[Feature Engineering](guides/custom-features.md)** - Add technical indicators and features
- **[Reward Functions](guides/reward-functions.md)** - Design reward functions for your strategy
- **[Performance Metrics](guides/metrics.md)** - Evaluate and customize trading performance metrics

## Key Features

### Multi-Timeframe Support
Observe market data at multiple time scales simultaneously:

```python
config = SequentialTradingEnvConfig(
    time_frames=["1min", "5min", "15min", "60min"],
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute")           # Execute every 5 minutes
)
```

### Futures Trading with Leverage
Trade with leverage and manage margin:

```python
config = SequentialTradingEnvConfig(
    leverage=10,                       # 10x leverage
    initial_cash=10000,
    margin_call_threshold=0.2,         # 20% margin ratio triggers liquidation
)
```

### Stop-Loss / Take-Profit Bracket Orders
Risk management with combinatorial action spaces:

```python
config = SequentialTradingEnvSLTPConfig(
    stoploss_levels=[-0.02, -0.05],    # -2%, -5%
    takeprofit_levels=[0.05, 0.10],    # +5%, +10%
    include_hold_action=True,          # Optional: set False to remove HOLD
)
# Action space: HOLD + (2 SL × 2 TP) = 5 actions (or 4 without HOLD)
```

## Environment Comparison

### Offline Environments (Backtesting)

All environments support both spot and futures trading via config (`leverage=1` for spot, `leverage>1` for futures with margin/liquidation mechanics).

| Environment | Bracket Orders | One-Step | Best For |
|-------------|----------------|----------|----------|
| **SequentialTradingEnv** | ❌ | ❌ | Standard sequential trading |
| **SequentialTradingEnvSLTP** | ✅ | ❌ | Risk management with SL/TP |
| **OneStepTradingEnv** | ✅ | ✅ | GRPO, contextual bandits |

### Live Environments (Exchange APIs)

| Environment | Exchange | Futures | Leverage | Bracket Orders |
|-------------|----------|---------|----------|----------------|
| **AlpacaTorchTradingEnv** | Alpaca | ❌ | ❌ | ❌ |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | ❌ | ❌ | ✅ |
| **BinanceFuturesTorchTradingEnv** | Binance | ✅ | ✅ | ❌ |
| **BinanceFuturesSLTPTorchTradingEnv** | Binance | ✅ | ✅ | ✅ |
| **BitgetFuturesTorchTradingEnv** | Bitget | ✅ | ✅ | ❌ |
| **BitgetFuturesSLTPTorchTradingEnv** | Bitget | ✅ | ✅ | ✅ |

## Next Steps

**[Getting Started Guide](getting-started.md)** - Install TorchTrade and run your first environment.
