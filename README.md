# TorchTrade

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![TorchRL](https://img.shields.io/badge/TorchRL-Compatible-orange.svg)

**Reinforcement learning environments for algorithmic trading. Train policies in backtesting, deploy to live markets.**

TorchTrade is a modular RL framework built on TorchRL that provides:
- 🎯 **9 Trading Environments** - Spot, futures, bracket orders, one-step variants
- 🤖 **Multiple RL Algorithms** - PPO, IQL, GRPO, DSAC, CTRL implementations
- 📊 **Multi-Timeframe Support** - Train on multiple time scales simultaneously
- 🔴 **Live Trading** - Direct Alpaca and Binance API integration
- 🧠 **LLM Integration** - Use GPT-4o-mini as trading agent
- 📐 **Rule-Based Actors** - Hard-coded strategies for imitation learning and baselines
- 📈 **Research to Production** - Same code for backtesting and live deployment

---

## Quick Start

### 1. Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/TorchTrade/torchtrade_envs.git
cd torchtrade_envs

# Install TorchTrade and all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### 2. Your First Environment

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import pandas as pd

# Load your OHLCV data (columns: timestamp, open, high, low, close, volume)
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create environment with multi-timeframe support
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15],        # 1-minute, 5-minute, 15-minute bars
    window_sizes=[12, 8, 8],       # Lookback windows per timeframe
    execute_on=(5, "Minute"),      # Execute trades every 5 minutes
    initial_cash=1000
)
env = SeqLongOnlyEnv(df, config)

# Run your first episode
tensordict = env.reset()
tensordict = env.step(tensordict)  # Take action (held in tensordict["action"])
print(f"Reward: {tensordict['reward'].item()}")
```

### 3. Train Your First Policy

```bash
# Train PPO on long-only environment
uv run python examples/online/ppo/train.py

# Customize with Hydra overrides
uv run python examples/online/ppo/train.py env.symbol="ETH/USD" optim.lr=1e-4
```

---

## Environment Overview

### Offline Environments

Offline environments are designed for **training on historical data** (backtesting). These are not "offline RL" methods, but rather environments that use pre-collected market data instead of live APIs.

| Environment | Asset Type | Futures | Leverage | Bracket Orders | One-Step | Best For |
|-------------|------------|---------|----------|----------------|----------|----------|
| **SeqLongOnlyEnv** | Crypto/Stocks | ❌ | ❌ | ❌ | ❌ | Beginners, simple strategies |
| **SeqLongOnlySLTPEnv** | Crypto/Stocks | ❌ | ❌ | ✅ | ❌ | Risk management research |
| **LongOnlyOneStepEnv** | Crypto/Stocks | ❌ | ❌ | ✅ | ✅ | Contextual bandits, GRPO |
| **SeqFuturesEnv** | Crypto | ✅ | ✅ (1-125x) | ❌ | ❌ | Advanced futures backtesting |
| **SeqFuturesSLTPEnv** | Crypto | ✅ | ✅ (1-125x) | ✅ | ❌ | Risk-managed futures |
| **FuturesOneStepEnv** | Crypto | ✅ | ✅ (1-125x) | ✅ | ✅ | Fast futures iteration, GRPO |

**Key Differences:**
- **Futures vs Spot**: Futures environments support leverage (1-125x), margin tracking, and liquidation mechanics. Spot environments are long-only.
- **Bracket Orders (SL/TP)**: SLTP variants support stop-loss and take-profit levels with combinatorial action spaces.
- **One-Step**: Optimized for GRPO training with episodic rollouts instead of sequential step-by-step trading.

### Live Environments

Live environments connect to real trading APIs for paper trading or live execution.

| Environment | API | Asset Type | Futures | Leverage | Bracket Orders | Best For |
|-------------|-----|------------|---------|----------|----------------|----------|
| **AlpacaTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ❌ | Paper/live spot trading |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ✅ | Live risk management |
| **BinanceFuturesTorchTradingEnv** | Binance | Crypto | ✅ | ✅ (1-125x) | ❌ | Binance futures trading |

**Key Differences:**
- **Alpaca**: Commission-free stocks and crypto trading with paper trading mode. Best for US markets.
- **Binance**: Cryptocurrency futures with high leverage. Supports testnet for safe testing.

---

## Common Use Cases

### Training PPO on Backtesting Data

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import datasets

# Load historical data from HuggingFace
df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025")
df = df["train"].to_pandas()
df['0'] = pd.to_datetime(df['0'])

# Configure multi-timeframe environment
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],        # 1m, 5m, 15m, 1h
    window_sizes=[12, 8, 8, 24],       # Lookback windows
    execute_on=(5, "Minute"),          # Execute every 5 minutes
    initial_cash=[1000, 5000],         # Domain randomization
    transaction_fee=0.0025,            # 0.25% transaction fee
    slippage=0.001                     # 0.1% slippage
)

# Create environment
env = SeqLongOnlyEnv(df, config)

# Train with PPO - see full example: examples/online/ppo/train.py
```

### Live Trading with Alpaca

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Configure live environment
config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 8],
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),
    paper=True  # Start with paper trading!
)

# Create environment (requires .env with API_KEY and SECRET_KEY)
env = AlpacaTorchTradingEnv(config)

# See full example: examples/live/alpaca/collect_live.py
```

### Using Stop-Loss/Take-Profit Bracket Orders

```python
from torchtrade.envs.offline import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05],     # -2%, -5%
    takeprofit_levels=[0.05, 0.10],     # +5%, +10%
    time_frames=[1, 5, 15],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlySLTPEnv(df, config)

# Action space: 1 (HOLD) + 2×2 (SL/TP combinations) = 5 actions
# Action 0: HOLD
# Action 1: BUY with SL=-2%, TP=+5%
# Action 2: BUY with SL=-2%, TP=+10%
# Action 3: BUY with SL=-5%, TP=+5%
# Action 4: BUY with SL=-5%, TP=+10%
```

### LLM-Based Trading with GPT-4

```python
from torchtrade.actor.llm_actor import LLMActor

# Use GPT-4o-mini as trading policy
policy = LLMActor(model="gpt-4o-mini", debug=True)

# LLM sees market data + account state, returns action
tensordict = env.reset()
action = policy(tensordict)

# See full example: examples/live/alpaca/collect_live_llm.py
```

### Rule-Based Trading Strategies (Expert Actors)

Use hard-coded trading strategies for imitation learning pre-training or as baselines:

```python
from torchtrade.actor import create_expert_ensemble

# Create ensemble of expert actors
experts = create_expert_ensemble(
    market_data_keys=["market_data_5Minute_24"],
    env_type="spot"  # or "sltp", "futures"
)

# Available strategies:
# - MomentumActor: Follow trends (Sharpe: 0.5-1.0)
# - MeanReversionActor: Fade extremes (Sharpe: 0.3-0.8)
# - BreakoutActor: Volatility expansion (Sharpe: 0.2-1.5)

# Use for demonstration collection
obs = env.reset()
for expert in experts:
    obs_with_action = expert(obs.clone())
    print(f"{expert.__class__.__name__}: action={obs_with_action['action'].item()}")

# Collect 1000 episodes of demonstrations from all experts
# python examples/online/rulebased/collect_demonstrations.py \
#     --expert all --num_episodes 100 --save_path demos.pt

# Then use for behavioral cloning pre-training before RL fine-tuning
# See: examples/online/rulebased/README.md and Issue #54
```

**Why use rule-based actors?**
- 🚀 **Bootstrap RL training** - Start from reasonable baseline instead of random initialization
- 📊 **Imitation learning** - Pre-train with behavioral cloning on expert demonstrations
- 🎯 **Baselines** - Compare learned policies against simple heuristics
- 🔍 **Interpretable** - Understand what strategies work in different market conditions

### Custom Feature Engineering

```python
import pandas as pd
import ta

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features"""
    # All feature columns must start with 'features_'
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Add RSI
    df["features_rsi_14"] = ta.momentum.RSIIndicator(
        df["close"], window=14
    ).rsi()

    # Add MACD
    macd = ta.trend.MACD(df["close"])
    df["features_macd_histogram"] = macd.macd_diff()

    # Fill NaN values
    df.fillna(0, inplace=True)
    return df

# Use in environment config
config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=custom_preprocessing,
    time_frames=[1, 5, 15],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)
```

### Loading and Using Trained Policies

```python
import torch
from torchrl.collectors import SyncDataCollector

# Load trained policy
policy.load_state_dict(torch.load("ppo_policy.pth"))
policy.eval()

# Collect rollouts with deterministic policy
collector = SyncDataCollector(
    eval_env,
    policy,
    frames_per_batch=10000,
    total_frames=-1,
    device="cuda"
)

for batch in collector:
    # Evaluate performance
    rewards = batch["reward"]
    print(f"Mean reward: {rewards.mean().item()}")

# See examples/online/ppo/train.py for full evaluation loop
```

---

## Training Algorithms

TorchTrade includes implementations of multiple RL algorithms:

| Algorithm | Type | Environment Type | Example Location | Key Features |
|-----------|------|------------------|------------------|--------------|
| **PPO** | On-policy | Sequential | `examples/online/ppo/` | Stable, general-purpose |
| **IQL** | Offline | Sequential | `examples/online/iql/` | Learn from historical data |
| **GRPO** | Policy gradient | One-step | `examples/online/grpo_futures_onestep/` | Fast futures training |
| **DSAC** | Off-policy | Sequential | `examples/online/dsac/` | Discrete soft actor-critic |
| **CTRL** | Self-supervised | Sequential | Research | Contrastive learning |

### Run Training Examples

```bash
# PPO on long-only environment
uv run python examples/online/ppo/train.py

# GRPO on futures one-step
uv run python examples/online/grpo_futures_onestep/train.py

# IQL offline training
uv run python examples/online/iql/train.py

# Customize with Hydra overrides
uv run python examples/online/ppo/train.py \
    env.symbol="ETH/USD" \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer

### Full Installation

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone repository
git clone https://github.com/TorchTrade/torchrl_alpaca_env.git
cd torchrl_alpaca_env

# 3. Install TorchTrade and all dependencies
uv sync

# 4. Install development dependencies (optional)
uv sync --extra dev

# 5. Install documentation dependencies (optional)
uv sync --extra docs

# 6. Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# 7. For live trading, create .env file
cat > .env << EOF
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
EOF

# 8. Run tests to verify installation
uv run pytest tests/ -v --cov=torchtrade
```

---

## Key Concepts

### Multi-Timeframe Observations

TorchTrade environments support simultaneous observation of multiple timeframes, allowing policies to understand market dynamics at different time scales:

```python
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],        # Minutes: 1m, 5m, 15m, 1h
    window_sizes=[12, 8, 8, 24],       # Bars per timeframe
    execute_on=(5, "Minute")           # Trade execution frequency
)

# Results in observations:
# - market_data_1m: [12, num_features] - Last 12 one-minute bars
# - market_data_5m: [8, num_features] - Last 40 minutes (8 × 5m)
# - market_data_15m: [8, num_features] - Last 120 minutes (8 × 15m)
# - market_data_60m: [24, num_features] - Last 24 hours (24 × 1h)
```

### Observation Structure

Environments return observations as TensorDict:

```python
# Standard environments (SeqLongOnlyEnv, SeqLongOnlySLTPEnv, etc.)
observation = {
    "market_data_1m": tensor([12, num_features]),   # 1-minute window
    "market_data_5m": tensor([8, num_features]),    # 5-minute window
    "market_data_15m": tensor([8, num_features]),   # 15-minute window
    "account_state": tensor([7]),                   # Account state vector
}

# Account state (7 elements):
# [cash, position_size, position_value, entry_price,
#  current_price, unrealized_pnl_pct, holding_time]

# Futures environments use 10-element account state:
# [cash, position_size, position_value, entry_price, current_price,
#  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
```

### Action Spaces

TorchTrade environments support three action space types:

**1. Standard 3-action discrete:**
```python
# Spot trading
Action 0: SELL (close position)
Action 1: HOLD (do nothing)
Action 2: BUY (open position)

# Futures trading
Action 0: SHORT (open short position)
Action 1: HOLD (do nothing)
Action 2: LONG (open long position)
```

**2. Combinatorial (Stop-Loss/Take-Profit):**
```python
# Long-only with brackets
Action 0: HOLD
Actions 1..N: BUY with (SL, TP) combinations

# Example: stoploss_levels=[-0.02, -0.05], takeprofit_levels=[0.05, 0.10]
# Action 0: HOLD
# Action 1: BUY with SL=-2%, TP=+5%
# Action 2: BUY with SL=-2%, TP=+10%
# Action 3: BUY with SL=-5%, TP=+5%
# Action 4: BUY with SL=-5%, TP=+10%
```

**3. Futures combinatorial (Long/Short brackets):**
```python
Action 0: HOLD/Close
Actions 1..N: LONG with (SL, TP) combinations
Actions N+1..2N: SHORT with (SL, TP) combinations
```

### Custom Reward Functions

TorchTrade provides a flexible reward function system that allows you to customize how your agent is rewarded during training.

**Default Behavior:**
All environments use simple log returns by default: `log(portfolio_value_t / portfolio_value_t-1)`

**Pre-Built Reward Functions:**

```python
from torchtrade.envs.reward import (
    sharpe_ratio_reward,           # Risk-adjusted returns
    drawdown_penalty_reward,        # Log return with drawdown penalty
    terminal_comparison_reward,     # Sparse terminal vs buy & hold
    hybrid_dense_sparse_reward,     # Combination of step-wise + terminal
    realized_pnl_reward            # Only reward realized profits
)
```

**Example 1: Using Pre-Built Reward Functions**

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.reward import drawdown_penalty_reward

# Configure environment with custom reward
config = SeqLongOnlyEnvConfig(
    symbol="BTC/USD",
    time_frames=[1, 5, 15],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000,
    reward_function=drawdown_penalty_reward  # Penalize large drawdowns
)

env = SeqLongOnlyEnv(df, config)
```

**Example 2: Creating Custom Reward Functions**

```python
from torchtrade.envs.reward import RewardContext
import numpy as np

def risk_adjusted_profit(ctx: RewardContext) -> float:
    """
    Custom reward that penalizes frequent trading and rewards consistent returns.
    """
    # Base reward: log return
    if ctx.old_portfolio_value <= 0:
        return 0.0
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Penalty for transaction costs
    fee_penalty = -ctx.fee_paid / ctx.old_portfolio_value

    # Penalty for excessive trading
    trade_penalty = -0.001 if ctx.trade_executed else 0.0

    # Terminal bonus for beating buy & hold
    terminal_bonus = 0.0
    if ctx.current_step >= ctx.max_steps - 1:
        buy_hold = ctx.metadata.get('buy_and_hold_value', ctx.old_portfolio_value)
        if ctx.new_portfolio_value > buy_hold:
            terminal_bonus = 0.5  # Bonus for outperformance

    return log_return + fee_penalty + trade_penalty + terminal_bonus

# Use custom reward
config = SeqLongOnlyEnvConfig(
    symbol="BTC/USD",
    reward_function=risk_adjusted_profit
)
```

**Example 3: Accessing Environment History**

```python
def sharpe_based_reward(ctx: RewardContext) -> float:
    """
    Reward based on running Sharpe ratio of portfolio returns.
    """
    # Access portfolio history from metadata
    history = ctx.metadata.get('portfolio_value_history', [])

    if len(history) < 2:
        return 0.0

    # Compute returns
    returns = [
        np.log(history[i] / history[i-1])
        for i in range(1, len(history))
    ]

    # Compute Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-9
    sharpe = mean_return / std_return

    return np.clip(sharpe, -10.0, 10.0)
```

**Available Context Fields:**

The `RewardContext` provides access to:
- `old_portfolio_value`, `new_portfolio_value` - Portfolio values before/after action
- `action` - Action taken (0, 1, 2, etc.)
- `current_step`, `max_steps` - Episode progress
- `trade_executed` - Whether a trade occurred
- `fee_paid`, `slippage_amount` - Transaction costs
- `metadata` - Environment-specific data:
  - `portfolio_value_history` - Historical portfolio values (offline envs)
  - `action_history`, `reward_history` - Historical actions/rewards (offline envs)
  - `buy_and_hold_value` - Buy & hold benchmark (terminal step only)
  - `leverage`, `margin_ratio`, `liquidation_price` - Futures-specific (futures envs)
  - `rollout_returns` - Returns during rollout (one-step envs)

**Reward Function Best Practices:**

1. **Keep rewards bounded** - Use `np.clip()` to prevent extreme values
2. **Consider scale** - Rewards in range [-10, 10] work well with most algorithms
3. **Balance dense vs sparse** - Dense rewards (every step) help learning, sparse rewards (terminal only) reduce noise
4. **Penalize bad behavior** - Add penalties for excessive trading, large drawdowns, liquidations
5. **Test extensively** - Verify your reward function produces expected values across different market conditions

---

## Architecture Overview

TorchTrade follows a modular architecture:

```
Raw OHLCV Data (1-minute bars)
    ↓
MarketDataObservationSampler
    ├── Resample to multiple timeframes (1m, 5m, 15m, 1h)
    ├── Apply feature preprocessing (technical indicators)
    └── Create sliding windows for each timeframe
    ↓
TensorDict Observations
    ├── market_data_* (one per timeframe)
    └── account_state (cash, position, PnL, etc.)
    ↓
TorchRL Environment (EnvBase)
    ├── _reset() - Initialize episode
    ├── _step(action) - Execute trade simulation
    ├── _calculate_reward() - Compute step reward
    └── _check_termination() - Check episode end
    ↓
Data Collector (SyncDataCollector)
    ├── Parallel environment execution
    └── Batch collection
    ↓
Loss Function (PPO/IQL/GRPO/DSAC)
    ├── Policy loss
    ├── Value loss
    └── Entropy regularization
    ↓
Optimizer (Adam)
    └── Policy network update
    ↓
Evaluation Loop
    ├── Test on held-out data
    ├── Compute metrics (Sharpe, drawdown, etc.)
    └── Render trading history visualization
```

---

## Configuration with Hydra

TorchTrade uses Hydra for configuration management:

```yaml
# examples/online/ppo/config.yaml
env:
  name: SeqLongOnlyEnv
  symbol: "BTC/USD"
  time_frames: [1, 5, 15, 60]
  window_sizes: [12, 8, 8, 24]
  execute_on: [5, "Min"]
  initial_cash: [1000, 5000]  # Domain randomization
  transaction_fee: 0.0025
  slippage: 0.001

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000

optim:
  lr: 2.5e-4
  anneal_lr: True
  max_grad_norm: 0.5

loss:
  gamma: 0.9
  gae_lambda: 0.95
  clip_epsilon: 0.1
  entropy_coef: 0.01
```

Override parameters from command line:

```bash
uv run python examples/online/ppo/train.py \
    env.symbol="ETH/USD" \
    env.initial_cash=[5000,10000] \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

---

## Features Highlights

- ✨ **Multi-Timeframe** - Train on 1m, 5m, 15m, 1h bars simultaneously for multi-scale market understanding
- 🤖 **LLM Integration** - Use GPT-4o-mini or other LLMs as trading policies via OpenAI API
- 📐 **Rule-Based Actors** - Hard-coded strategies (momentum, mean reversion, breakout) for imitation learning and baselines
- 📊 **Coverage Tracking** - Measure state space exploration diversity with entropy-based metrics
- 🎯 **Custom Loss Functions** - GRPO, CTRL (contrastive learning), standard PPO/IQL implementations
- 📈 **Weights & Biases** - Built-in W&B logging for experiments, metrics, and visualizations
- 🔄 **Domain Randomization** - Randomize initial cash, episode start positions for robustness
- 🛡️ **Paper Trading** - Test strategies safely with Alpaca paper trading before live deployment
- ⚡ **Performance Optimized** - Supports torch.compile() and CUDA graphs for fast training
- 🔧 **Modular Design** - Easy to extend environments, add custom features, modify rewards
- 📉 **Risk Management** - Built-in stop-loss/take-profit, margin/leverage, liquidation mechanics

---

## Full Documentation

- 📘 **[MkDocs Documentation](docs/)** - Run `mkdocs serve` to start documentation server
- 📂 **[Environment Guides](docs/Environments/)** - Detailed guides for each environment
  - [SeqLongOnlyEnv](docs/Environments/SeqLongOnlyEnv.md) - Sequential long-only trading
  - [SeqLongOnlySLTPEnv](docs/Environments/SeqLongOnlySLTPEnv.md) - Long-only with brackets
  - [LongOnlyOneStepEnv](docs/Environments/LongOnlyOneStepEnv.md) - One-step variant
- 💼 **[Training Examples](examples/)**
  - [Live Trading with Alpaca](examples/live/alpaca/README.md) - Complete live trading guide
  - [Rule-Based Actors](examples/online/rulebased/README.md) - Hard-coded strategies for imitation learning
  - [PPO Training](examples/online/ppo/) - Proximal Policy Optimization
  - [IQL Training](examples/online/iql/) - Implicit Q-Learning
  - [GRPO Training](examples/online/grpo_futures_onestep/) - Group Relative Policy Optimization
- 🔬 **[Research Iterations](RewardFunctionResearch/)** - Reward function experiments and findings

---

## Trading Platforms

Start live trading with these supported platforms:

### 🪙 Cryptocurrency Trading

**[Binance](https://www.binance.com/en/activity/referral-entry/CPA)** - Leading cryptocurrency exchange
- **Supported by:** `BinanceFuturesTorchTradingEnv`
- **Features:** Spot & futures trading, up to 125x leverage, testnet available
- **Commission:** Maker 0.02% / Taker 0.04% (with BNB discount)
- **Get Started:** [Sign up for Binance](https://www.binance.com/en/activity/referral-entry/CPA) <!-- Replace with your affiliate link -->

### 📈 Stock & Crypto API

**[Alpaca](https://alpaca.markets/)** - Commission-free trading API
- **Supported by:** `AlpacaTorchTradingEnv`, `AlpacaSLTPTorchTradingEnv`
- **Features:** Commission-free stocks & crypto, paper trading, real-time data
- **Best for:** US markets, algorithmic trading
- **Get Started:** [Sign up for Alpaca](https://alpaca.markets/signup)

### 💱 Forex & CFD Trading

**[OANDA](https://www.oanda.com/)** - Forex and CFD trading platform
- **Status:** Future integration planned
- **Features:** 68+ currency pairs, CFDs, competitive spreads
- **Regulation:** FCA, ASIC, CFTC, NFA regulated
- **Get Started:** [Sign up for OANDA](https://www.oanda.com/) <!-- Replace with your affiliate link -->

---

**Note:** Binance and OANDA links are affiliate links. Using them helps support TorchTrade development at no extra cost to you. Alpaca does not offer an affiliate program.

---

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=torchtrade --cov-report=term-missing

# Run specific test file
uv run pytest tests/envs/offline/test_seqlongonly.py -v

# Run specific test
uv run pytest tests/envs/offline/test_seqlongonly.py::test_step_buy_action -v
```

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install with development dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ -v --cov=torchtrade --cov-report=html
```

### Reporting Issues

Found a bug or have a feature request? Please open an issue:

[GitHub Issues](https://github.com/TorchTrade/torchrl_alpaca_env/issues)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Citation

If you use TorchTrade in your research, please cite:

```bibtex
@software{torchtrade2025,
  title={TorchTrade: Reinforcement Learning Environments for Algorithmic Trading},
  author={Dittert, Sebastian},
  year={2025},
  url={https://github.com/TorchTrade/torchtrade_envs}
}
```

---

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade_envs/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade_envs/issues)
- 📧 **Email**: torchtradecontact@gmail.com
- 💰 **Donate**: [PayPal](https://www.paypal.me/yourname)

---

**Built with TorchRL • Designed for Algorithmic Trading • Open Source**
