# TorchTrade Documentation

Welcome to the TorchTrade documentation! TorchTrade is a machine learning framework for algorithmic trading built on TorchRL.

TorchTrade's goal is to provide accessible deployment of RL methods to trading. The framework supports various RL methodologies including **online RL**, **offline RL**, **model-based RL**, **contrastive learning**, and many more areas of reinforcement learning research. Beyond RL, TorchTrade integrates traditional trading methods such as **rule-based strategies**, as well as modern approaches including **LLMs** (both local models and frontier model integrations) as trading actors.

## What is TorchTrade?

TorchTrade provides modular environments for both live trading with major exchanges and offline backtesting. The framework supports:

- 🎯 **Multi-Timeframe Observations** - Train on 1m, 5m, 15m, 1h bars simultaneously
- 🤖 **Multiple RL Algorithms** - PPO, IQL, GRPO, DSAC implementations
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
  - SeqLongOnlyEnv, SeqFuturesEnv, OneStepEnv variants
- **[Online Environments](environments/online.md)** - Live trading with exchange APIs
  - Alpaca, Binance, Bitget integrations

### Components
- **[Loss Functions](components/losses.md)** - Training objectives (GRPOLoss, CTRLLoss, CTRLPPOLoss)
- **[Transforms](components/transforms.md)** - Data preprocessing (CoverageTracker, ChronosEmbeddingTransform)
- **[Actors](components/actors.md)** - Trading policies (RuleBasedActor, LLMActor, LocalLLMActor, HumanActor)

### Advanced Customization
- **[Feature Engineering](guides/custom-features.md)** - Add technical indicators and features
- **[Reward Functions](guides/reward-functions.md)** - Design reward functions for your strategy
- **[Performance Metrics](guides/metrics.md)** - Evaluate and customize trading performance metrics

## Key Features

### Multi-Timeframe Support
Observe market data at multiple time scales simultaneously:

```python
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min", "60min"],
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute")           # Execute every 5 minutes
)
```

### Futures Trading with Leverage
Trade with leverage and manage margin:

```python
config = SeqFuturesEnvConfig(
    leverage=10,                       # 10x leverage
    initial_cash=10000,
    margin_call_threshold=0.2,         # 20% margin ratio triggers liquidation
)
```

### Stop-Loss / Take-Profit Bracket Orders
Risk management with combinatorial action spaces:

```python
config = SeqLongOnlySLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05],    # -2%, -5%
    takeprofit_levels=[0.05, 0.10],    # +5%, +10%
    include_hold_action=True,          # Optional: set False to remove HOLD
)
# Action space: HOLD + (2 SL × 2 TP) = 5 actions (or 4 without HOLD)
```

## Architecture Overview

```
Raw OHLCV Data (1-minute bars)
    ↓
MarketDataObservationSampler
    ├── Resample to multiple timeframes
    ├── Apply feature preprocessing
    └── Create sliding windows
    ↓
TensorDict Observations
    ├── market_data_* (per timeframe)
    └── account_state (cash, position, PnL)
    ↓
TorchRL Environment (EnvBase)
    ├── _reset() - Initialize episode
    ├── _step(action) - Execute trade
    ├── _calculate_reward() - Compute reward
    └── _check_termination() - Check end
    ↓
Loss Function (PPO/IQL/GRPO/DSAC)
    └── Optimizer → Policy Update
```

## Environment Comparison

### Offline Environments (Backtesting)

| Environment | Futures | Leverage | Bracket Orders | One-Step | Best For |
|-------------|---------|----------|----------------|----------|----------|
| **SeqLongOnlyEnv** | ❌ | ❌ | ❌ | ❌ | Beginners, simple strategies |
| **SeqLongOnlySLTPEnv** | ❌ | ❌ | ✅ | ❌ | Risk management research |
| **LongOnlyOneStepEnv** | ❌ | ❌ | ✅ | ✅ | GRPO, contextual bandits |
| **SeqFuturesEnv** | ✅ | ✅ | ❌ | ❌ | Advanced futures backtesting |
| **SeqFuturesSLTPEnv** | ✅ | ✅ | ✅ | ❌ | Risk-managed futures |
| **FuturesOneStepEnv** | ✅ | ✅ | ✅ | ✅ | Fast futures iteration |

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

Ready to get started? Head to the **[Getting Started Guide](getting-started.md)** to install TorchTrade and run your first environment!

Already familiar with the basics? Check out:

- **[Offline Environments](environments/offline.md)** - Deep dive into backtesting environments
- **[Reward Functions](guides/reward-functions.md)** - Design better reward signals
- **[Building Custom Environments](guides/custom-environment.md)** - Extend the framework

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade/issues)
- 📧 **Email**: torchtradecontact@gmail.com

---

**Built with TorchRL • Designed for Algorithmic Trading • Open Source**
