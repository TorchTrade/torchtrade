# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchTrade is a reinforcement learning framework for algorithmic trading built on TorchRL. It provides 11 modular environments spanning 3 exchange integrations (Alpaca, Binance, Bitget) for live trading and 6 offline backtesting environments. The framework supports multiple timeframes, custom feature engineering, futures trading with leverage, bracket orders with stop-loss/take-profit, and various RL algorithms.

**Live Exchange Integrations:**
- Alpaca: US stocks and crypto with paper trading mode
- Binance: Cryptocurrency futures with high leverage and testnet
- Bitget: Cryptocurrency futures with competitive fees and testnet

**Key Features:**
- Multi-timeframe observations
- Futures environments with leverage and margin management
- Bracket orders (SLTP) with combinatorial action spaces
- Shared abstractions (BaseFuturesObservationClass, SLTPMixin)
- History tracking with FuturesHistoryTracker

## Agent Usage

**IMPORTANT**: When working with TorchRL-specific implementations, debugging, or architectural decisions, use the `torchrl-engineer` agent. This includes:
- Implementing RL algorithms (PPO, SAC, DQN, IQL, GRPO, etc.)
- Creating or modifying replay buffers
- Designing environment transforms
- Working with TensorDict operations
- Troubleshooting TorchRL environments
- Optimizing training loops
- Understanding TorchRL's internal mechanisms

To invoke the agent, use the Task tool with `subagent_type="torchrl-engineer"`.

**IMPORTANT: Pull Request Review Agents**

Before pushing any PR, **ALWAYS** use these specialized review agents:

1. **Code Simplifier Agent** - Use `pr-review-toolkit:code-simplifier` to review code quality:
   - Identifies overly complex code patterns
   - Suggests simpler alternatives
   - Finds unnecessary abstractions
   - Recommends ways to reduce cognitive load
   - Invoke with: `Task` tool with `subagent_type="pr-review-toolkit:code-simplifier"`

2. **Test Analyzer Agent** - Use `pr-review-toolkit:pr-test-analyzer` to review test coverage:
   - Analyzes test completeness and quality
   - Identifies missing edge cases
   - Ensures adequate coverage of new functionality
   - Verifies critical paths are tested
   - Invoke with: `Task` tool with `subagent_type="pr-review-toolkit:pr-test-analyzer"`

**Workflow**: Create PR → Run code-simplifier agent → Run pr-test-analyzer agent → Address findings → Merge

## Environment Management

**IMPORTANT: Use UV for Worktrees**

When working with git worktrees, **always use `uv` instead of a shared conda environment**:

```bash
# In each worktree, create an isolated environment
uv sync
```

**Why UV for worktrees?**
- Creates isolated local Python environment per worktree
- Prevents dependency conflicts between different branches
- Eliminates need for repeated `pip install -e .` after switching branches
- Avoids "module not found" errors from shared conda environments

**When to use pip install:**
- Single working directory (not using worktrees)
- After making changes to setup.py or dependencies

## Development Requirements

**CRITICAL: Always follow these steps when adding new features:**

### 1. Update Tests
- Create or modify tests for new functionality
- Run full test suite: `pytest tests/ -v`
- Ensure 100% pass rate before committing
- Mock external APIs appropriately (see `tests/envs/*/mocks.py`)

### 2. Update Documentation
- Update `README.md` with usage examples for new features
- Update relevant `docs/` files
- Add comprehensive docstrings to new functions/classes
- Document configuration parameters in dataclass configs

### 3. **CRITICAL**: Apply to ALL Environments
**If a feature affects environments, it MUST be applied to ALL relevant environments:**

- **Live environments**: Alpaca, Binance, Bitget (both basic and SLTP variants)
- **Offline environments**: SeqLongOnly, SeqFutures, OneStep variants (both basic and SLTP)

**Examples of features that must be applied universally:**
- History tracking (FuturesHistoryTracker added to all futures environments)
- Observation specs changes (must propagate to all environments)
- Reward function modifications (must be consistent across environments)
- Account state changes (all environments must use same structure)

**Why this is critical:**
- Ensures consistency across all trading platforms
- Prevents feature disparity between online and offline training
- Maintains architectural integrity
- Allows seamless policy transfer between environments

### 4. **CRITICAL**: Run PR Review Agents Before Merging

**Before pushing any PR, ALWAYS run both review agents:**

1. **Code Quality Review**:
   ```
   Use @pr-review-toolkit:code-simplifier agent to review the PR
   ```
   - Identifies opportunities for simplification
   - Catches overly complex patterns
   - Ensures code maintainability

2. **Test Coverage Review**:
   ```
   Use @pr-review-toolkit:pr-test-analyzer agent to review the PR
   ```
   - Verifies test completeness
   - Identifies missing edge cases
   - Ensures critical paths are tested

**Required Workflow**:
1. Create PR with your changes
2. Run `@pr-review-toolkit:code-simplifier` agent
3. Run `@pr-review-toolkit:pr-test-analyzer` agent
4. Address all findings from both agents
5. Merge PR

**Do NOT merge without running both agents!**

## Build & Development Commands

```bash
# Use UV for isolated environments (PREFERRED for worktrees)
uv sync

# Or use pip for single working directory
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

The codebase provides 11 TorchRL-compatible environments across two categories:

**Live Trading Environments** (5 environments across 3 exchanges):

`torchtrade.envs.alpaca`:
- `AlpacaTorchTradingEnv` - 3-action discrete (sell/hold/buy) for US stocks/crypto
- `AlpacaSLTPTorchTradingEnv` - Bracket orders with SL/TP levels

`torchtrade.envs.binance`:
- `BinanceFuturesTorchTradingEnv` - 3-action discrete (short/close/long) for crypto futures
- `BinanceFuturesSLTPTorchTradingEnv` - Bracket orders with SL/TP levels for futures

`torchtrade.envs.bitget`:
- `BitgetFuturesTorchTradingEnv` - 3-action discrete (short/close/long) for crypto futures
- `BitgetFuturesSLTPTorchTradingEnv` - Bracket orders with SL/TP levels for futures

**Offline Backtesting Environments** (6 environments):

`torchtrade.envs.offline`:
- `SeqLongOnlyEnv` - Sequential long-only trading from OHLCV DataFrames
- `SeqLongOnlySLTPEnv` - Sequential with SL/TP bracket orders
- `LongOnlyOneStepEnv` - One-step setting for GRPO/contextual bandit training
- `SeqFuturesEnv` - Futures trading with leverage and margin
- `SeqFuturesSLTPEnv` - Futures with SL/TP bracket orders
- `FuturesOneStepEnv` - One-step futures for GRPO training

**Shared Abstractions:**
- `BaseFuturesObservationClass` - Abstract base for exchange-specific observation classes
- `SLTPMixin` - Shared logic for stop-loss/take-profit environments
- `FuturesHistoryTracker` - Episode history tracking for futures environments

All environments inherit from TorchRL's `EnvBase` and follow the TensorDict pattern.

### Data Flow

1. **Observation**: Market data is provided as multi-timeframe OHLCV windows. The `MarketDataObservationSampler` handles resampling and feature preprocessing.

2. **Feature Engineering**: Custom preprocessing functions transform raw OHLCV into features. Features must be columns starting with `features_` prefix.

3. **Account State**:
   - **Spot/Long-Only**: 7 elements: `[cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]`
   - **Futures**: 10 elements: `[cash, position_size, position_value, entry_price, current_price, unrealized_pnlpct, leverage, margin_ratio, liquidation_price, holding_time]`

4. **Actions**: Discrete categorical. Standard envs use 3 actions (sell/hold/buy or short/close/long). SLTP envs have 1 + (num_sl_levels × num_tp_levels) actions.

### Lookahead Bias Prevention (Issue #10 - Fixed)

**CRITICAL**: TorchTrade prevents lookahead bias in multi-timeframe observations by indexing higher timeframe bars by their END time.

**The Problem**: Pandas `resample()` indexes bars by START time but aggregates through END time. A 5-minute bar at 00:25:00 contains data from 00:25:00-00:29:59. Without correction, an agent executing at 00:27:00 would see data from minute 29 (future data).

**The Fix** (`torchtrade/envs/offline/sampler.py` lines 71-76):
```python
# After resampling, shift higher timeframe bars by their period
if tf != execute_on:
    offset = pd.Timedelta(tf.to_pandas_freq())
    resampled.index = resampled.index + offset
```

This ensures:
- Higher timeframe bars are indexed by END time (when they're complete)
- Only completed bars are visible to the agent at any execution time
- Backtest results accurately reflect real-world constraints
- Policies transfer reliably to live trading

**Impact**: This is a BREAKING CHANGE - higher timeframe data is now correct but different from before. Existing models trained with the old (incorrect) data may need retraining.

### Key Components

**Live Trading Infrastructure:**
- `torchtrade/envs/alpaca/obs_class.py` - Alpaca data fetching and observation construction
- `torchtrade/envs/alpaca/order_executor.py` - Order execution via Alpaca API
- `torchtrade/envs/binance/obs_class.py` - Binance futures data fetching (inherits from BaseFuturesObservationClass)
- `torchtrade/envs/binance/futures_order_executor.py` - Binance futures order execution with bracket orders
- `torchtrade/envs/bitget/obs_class.py` - Bitget futures data fetching (inherits from BaseFuturesObservationClass)
- `torchtrade/envs/bitget/futures_order_executor.py` - Bitget futures order execution with bracket orders

**Shared Abstractions:**
- `torchtrade/envs/futures/obs_class.py` - BaseFuturesObservationClass for common observation logic
- `torchtrade/envs/sltp_mixin.py` - SLTPMixin for bracket order logic
- `torchtrade/envs/sltp_helpers.py` - Helper functions for SL/TP bracket pricing
- `torchtrade/envs/state.py` - FuturesHistoryTracker for episode history

**Offline Infrastructure:**
- `torchtrade/envs/offline/sampler.py` - Multi-timeframe data sampling for offline envs

**Training:**
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

Tests use pytest fixtures from `tests/conftest.py` that generate synthetic OHLCV data. Mock classes for exchange APIs are in:
- `tests/envs/alpaca/mocks.py` - Alpaca API mocks
- `tests/envs/binance/mocks.py` - Binance API mocks
- `tests/envs/bitget/mocks.py` - Bitget API mocks

All tests must pass before merging (target: 100% pass rate).

## External Dependencies

- Uses `trading-nets` package from GitHub for neural network architectures
- Training logs to Weights & Biases
- Training data from HuggingFace datasets (e.g., `Sebasdi/TorchTrade_btcusd_spot_1m_12_2024_to_09_2025`)

**Exchange APIs for Live Trading:**
- Alpaca API: Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`
- Binance API: Requires `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` in `.env` (supports testnet)
- Bitget API: Requires `BITGET_API_KEY`, `BITGET_SECRET`, and `BITGET_PASSPHRASE` in `.env` (supports testnet)
