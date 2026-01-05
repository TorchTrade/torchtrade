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
- 📈 **Research to Production** - Same code for backtesting and live deployment

---

## Quick Start

### 1. Installation

```bash
# Create conda environment
conda create --name torchtrade python=3.9
conda activate torchtrade

# Clone and install
git clone https://github.com/TorchTrade/torchtrade_envs.git
cd torchtrade_envs
pip install -e .
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
python examples/online/ppo/train.py

# Customize with Hydra overrides
python examples/online/ppo/train.py env.symbol="ETH/USD" optim.lr=1e-4
```

---

## Environment Overview

Choose the right environment for your trading strategy:

| Environment | Category | Trading Type | Action Space | Key Features | Best For |
|-------------|----------|--------------|--------------|--------------|----------|
| **SeqLongOnlyEnv** | Offline | Long-only | 3-action discrete | Sequential trading, simple | Beginners, simple strategies |
| **SeqLongOnlySLTPEnv** | Offline | Long-only + Brackets | Combinatorial (1+N×M) | Stop-loss/take-profit | Risk management research |
| **LongOnlyOneStepEnv** | Offline | One-step long | Discrete SL/TP | GRPO-optimized rollouts | Contextual bandits |
| **SeqFuturesEnv** | Offline | Futures | 3-action (short/hold/long) | Leverage (1-125x), margin | Advanced backtesting |
| **SeqFuturesSLTPEnv** | Offline | Futures + Brackets | Combinatorial | Full bracket orders | Risk-managed futures |
| **FuturesOneStepEnv** | Offline | One-step futures | Discrete | GRPO, leverage, liquidation | Fast iteration |
| **AlpacaTorchTradingEnv** | Live | Live spot | 3-action | Real-time Alpaca API | Paper/live trading |
| **AlpacaSLTPTorchTradingEnv** | Live | Live + Brackets | Combinatorial | Alpaca bracket orders | Live risk management |
| **BinanceFuturesTorchTradingEnv** | Live | Live futures | 3-action | Binance futures, leverage | Binance live trading |

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
python examples/online/ppo/train.py

# GRPO on futures one-step
python examples/online/grpo_futures_onestep/train.py

# IQL offline training
python examples/online/iql/train.py

# Customize with Hydra overrides
python examples/online/ppo/train.py \
    env.symbol="ETH/USD" \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Full Installation

```bash
# 1. Create conda environment
conda create --name torchtrade python=3.9
conda activate torchtrade

# 2. Clone repository
git clone https://github.com/TorchTrade/torchrl_alpaca_env.git
cd torchrl_alpaca_env

# 3. Install TorchTrade
pip install -e .

# 4. Install development dependencies (optional)
pip install -e .[dev]

# 5. For live trading, create .env file
cat > .env << EOF
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
EOF

# 6. Run tests to verify installation
pytest tests/ -v --cov=torchtrade
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

### Reward Functions

**Sequential Environments (SeqLongOnlyEnv, SeqFuturesEnv):**
- Dense per-step rewards based on portfolio returns
- Sparse terminal reward comparing to buy-and-hold baseline
- Differential Sharpe Ratio bonus for risk-adjusted returns

**One-Step Environments (LongOnlyOneStepEnv, FuturesOneStepEnv):**
- Sharpe ratio of log-returns during the rollout period
- Annualized based on execution timeframe
- Liquidation penalty (-2.0) for futures environments

**Customization:**
All reward functions can be customized by overriding the `_calculate_reward()` method.

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
python examples/online/ppo/train.py \
    env.symbol="ETH/USD" \
    env.initial_cash=[5000,10000] \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

---

## Features Highlights

- ✨ **Multi-Timeframe** - Train on 1m, 5m, 15m, 1h bars simultaneously for multi-scale market understanding
- 🤖 **LLM Integration** - Use GPT-4o-mini or other LLMs as trading policies via OpenAI API
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
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=torchtrade --cov-report=term-missing

# Run specific test file
pytest tests/envs/offline/test_seqlongonly.py -v

# Run specific test
pytest tests/envs/offline/test_seqlongonly.py::test_step_buy_action -v
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
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=torchtrade --cov-report=html
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
  url={https://github.com/TorchTrade/torchrl_alpaca_env}
}
```

---

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchrl_alpaca_env/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchrl_alpaca_env/issues)
- 📧 **Email**: torchtradecontact@gmail.com
- 💰 **Donate**: [PayPal](https://www.paypal.me/yourname)

---

**Built with TorchRL • Designed for Algorithmic Trading • Open Source**
