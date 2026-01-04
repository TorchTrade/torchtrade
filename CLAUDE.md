# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchTrade is a reinforcement learning framework for algorithmic trading built on TorchRL. It provides modular environments for both live trading (via Alpaca API) and offline backtesting, with support for multiple timeframes, custom feature engineering, and various RL algorithms.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=torchtrade/envs --cov-report=term-missing

# Run a single test file
pytest tests/envs/offline/test_seqlongonly.py -v

# Run a specific test
pytest tests/envs/offline/test_seqlongonly.py::test_step_buy_action -v

# Start documentation server
mkdocs serve

# Run training example (IQL)
python examples/online/iql/train.py

# Run with Hydra config overrides
python examples/online/iql/train.py env.symbol="ETH/USD" optim.lr=1e-4
```

## Architecture

### Environment Hierarchy

The codebase provides two categories of TorchRL-compatible environments:

**Live Trading Environments** (`torchtrade.envs.alpaca`):
- `AlpacaTorchTradingEnv` - 3-action discrete (sell/hold/buy) connected to Alpaca API
- `AlpacaSLTPTorchTradingEnv` - Combinatorial action space with bracket orders (stop-loss/take-profit levels)

**Offline Backtesting Environments** (`torchtrade.envs.offline`):
- `SeqLongOnlyEnv` - Sequential long-only trading from OHLCV DataFrames
- `SeqLongOnlySLTPEnv` - Sequential with SL/TP bracket orders
- `LongOnlyOneStepEnv` - One-step setting for GRPO/contextual bandit training
- `SeqFuturesEnv` - Futures trading with leverage and margin

All environments inherit from TorchRL's `EnvBase` and follow the TensorDict pattern.

### Data Flow

1. **Observation**: Market data is provided as multi-timeframe OHLCV windows. The `MarketDataObservationSampler` handles resampling and feature preprocessing.

2. **Feature Engineering**: Custom preprocessing functions transform raw OHLCV into features. Features must be columns starting with `features_` prefix.

3. **Account State**: 7-element vector: `[cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]`

4. **Actions**: Discrete categorical. Standard envs use 3 actions (sell/hold/buy). SLTP envs have 1 + (num_sl_levels × num_tp_levels) actions.

### Key Components

- `torchtrade/envs/alpaca/obs_class.py` - Alpaca data fetching and observation construction
- `torchtrade/envs/alpaca/order_executor.py` - Order execution via Alpaca API
- `torchtrade/envs/offline/sampler.py` - Multi-timeframe data sampling for offline envs
- `torchtrade/actor/llm_actor.py` - LLM-based trading agent using OpenAI API
- `torchtrade/losses/grpo_loss.py` - GRPO loss implementation for one-step RL

### Configuration Pattern

Environments use dataclass configs (e.g., `SeqLongOnlyEnvConfig`). Training uses Hydra with configs in `examples/online/*/config.yaml`.

Key config parameters:
- `time_frames` + `window_sizes`: Multi-timeframe observation windows
- `execute_on`: Timeframe for trade execution
- `transaction_fee`, `slippage`: Cost modeling
- `initial_cash`: Starting capital (can be tuple for domain randomization)

## Testing

Tests use pytest fixtures from `tests/conftest.py` that generate synthetic OHLCV data. Mock classes for Alpaca API are in `tests/envs/alpaca/mocks.py`.

## External Dependencies

- Uses `trading-nets` package from GitHub for neural network architectures
- Training logs to Weights & Biases
- Training data from HuggingFace datasets (e.g., `Sebasdi/TorchTrade_btcusd_spot_1m_12_2024_to_09_2025`)
- Alpaca API for live trading (requires `API_KEY` and `SECRET_KEY` in `.env`)
