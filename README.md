<p align="center">
  <img src="docs/images/torchtrade_white.png" alt="TorchTrade" width="250">
</p>

<p align="center">
  <a href="https://torchtrade.github.io/torchtrade.io/index.html">
    <img src="https://img.shields.io/badge/Website-torchtrade.io-blueviolet.svg" alt="Website">
  </a>

  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/TorchRL-Compatible-green.svg" alt="TorchRL">

<a href="https://github.com/sponsors/TorchTrade">
  <img src="https://img.shields.io/badge/Sponsor-TorchTrade-ea4aaa.svg" alt="Sponsor">
</a>

<a href="https://discord.gg/ptqnufYmaj">
  <img src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2.svg?logo=discord&logoColor=white" alt="Discord">
</a>
</p>

# TorchTrade

**A machine learning framework for algorithmic trading built on [TorchRL](https://github.com/pytorch/rl).**

TorchTrade's goal is to provide accessible deployment of RL methods to trading. The framework supports various RL methodologies including **online RL**, **offline RL**, **model-based RL**, **contrastive learning**, and many more areas of reinforcement learning research. Beyond RL, TorchTrade integrates traditional trading methods such as **rule-based strategies**, as well as modern approaches including **LLMs** (both local models and frontier model integrations) as trading actors.

TorchTrade provides modular environments for both live trading with major exchanges and offline backtesting. The framework supports:
- 🎯 **Multi-Timeframe Observations** - Train on 1m, 5m, 15m, 1h bars simultaneously
- 🤖 **Multiple RL Algorithms** - PPO, DQN, IQL, GRPO, DSAC, CTRL implementations
- 📊 **Feature Engineering** - Add technical indicators and custom features
- 🔴 **Live Trading** - Direct Alpaca, Binance, Bitget, Bybit, OKX, and Polymarket integrations
- 🧠 **LLM Integration** - Use GPT-4o-mini or local LLMs as trading agents
- 📐 **Rule-Based Actors** - Hard-coded strategies for imitation learning and baselines
- 🔮 **Pretrained Encoder Transforms** - Foundation model embeddings for time series
- 📦 **Ready-to-Use Datasets** - Pre-processed OHLCV data at [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade)
- 📈 **Research to Production** - Same code for backtesting and live deployment
- 🛠️ **[Claude Code Agents](https://code.claude.com/docs/en/sub-agents)** - Pre-built AI coding agents for developing and extending TorchTrade ([available on our website](https://torchtrade.github.io/torchtrade.io/index.html))
- 📝 **Research Articles** - In-depth articles on RL trading strategies and framework design ([get here](https://torchtrade.github.io/torchtrade.io/index.html))

> **⚠️ Work in Progress:** TorchTrade is under active development. We continuously add new features, improvements, and optimizations. Expect API changes, new environments, and enhanced functionality in future releases.
>
> **Current Scope:** The framework currently focuses on single-asset trading environments (one symbol per environment). Multi-asset portfolio optimization and cross-asset trading environments are planned for future releases.

---

## 📚 Website & Documentation

🌐 **[TorchTrade Website](https://torchtrade.github.io/torchtrade.io/index.html)** — Landing page with overview, features, Claude Code agents, and research articles

📖 **[TorchTrade Documentation](https://torchtrade.github.io/torchtrade/)** — Comprehensive guides, tutorials, and API reference

- **[Getting Started](https://torchtrade.github.io/torchtrade/getting-started/)** - Installation and first environment
- **[Environments](https://torchtrade.github.io/torchtrade/environments/offline/)** - Offline and online trading environments
- **[Examples](https://torchtrade.github.io/torchtrade/examples/)** - Training scripts for PPO, IQL, GRPO, and more
- **[Components](https://torchtrade.github.io/torchtrade/components/losses/)** - Loss functions, transforms, and actors
- **[Advanced Customization](https://torchtrade.github.io/torchtrade/guides/custom-features/)** - Custom features, rewards, and environments

---

## Quick Start

### 1. Installation

```bash
# Install UV (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/TorchTrade/torchtrade.git
cd torchtrade
uv sync
source .venv/bin/activate  # On Unix/macOS

# Optional: Install with extra features
uv sync --extra llm              # LLM actors (OpenAI API + local vLLM/transformers)
uv sync --extra chronos          # Chronos forecasting transforms
uv sync --all-extras             # Install all optional dependencies
```

### 2. Your First Environment

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
import pandas as pd

# Load OHLCV data
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create environment (spot trading = long-only)
config = SequentialTradingEnvConfig(
    trading_mode="spot",  # or "futures" for leveraged trading
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)
env = SequentialTradingEnv(df, config)

# Run
tensordict = env.reset()
tensordict = env.step(tensordict)
print(f"Reward: {tensordict['reward'].item()}")
```

### 3. Train Your First Policy

```bash
# Train PPO with default settings
uv run python examples/online_rl/ppo/train.py

# Customize with Hydra overrides
uv run python examples/online_rl/ppo/train.py \
    env.symbol="BTC/USD" \
    optim.lr=1e-4
```

For detailed tutorials, see **[Getting Started Guide](https://torchtrade.github.io/torchtrade/getting-started/)**.

---

## Live Environments

TorchTrade supports live trading with major exchanges:

| Environment | Exchange | Asset Type | Futures | Leverage | Bracket Orders |
|-------------|----------|------------|---------|----------|----------------|
| **AlpacaTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ❌ |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ✅ |
| **BinanceFuturesTorchTradingEnv** | Binance | Crypto | ✅ | ✅ (1-125x) | ❌ |
| **BinanceFuturesSLTPTorchTradingEnv** | Binance | Crypto | ✅ | ✅ (1-125x) | ✅ |
| **BitgetFuturesTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ (1-125x) | ❌ |
| **BitgetFuturesSLTPTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ (1-125x) | ✅ |
| **BybitFuturesTorchTradingEnv** | Bybit | Crypto | ✅ | ✅ (1-100x) | ❌ |
| **BybitFuturesSLTPTorchTradingEnv** | Bybit | Crypto | ✅ | ✅ (1-100x) | ✅ |
| **OKXFuturesTorchTradingEnv** | OKX | Crypto | ✅ | ✅ (1-125x) | ❌ |
| **OKXFuturesSLTPTorchTradingEnv** | OKX | Crypto | ✅ | ✅ (1-125x) | ✅ |
| **PolymarketBetEnv** | Polymarket | Prediction markets | ❌ | ❌ | ❌ |

**Need another broker?** Request support for additional platforms (Interactive Brokers, Kraken, etc.) by [creating an issue](https://github.com/TorchTrade/torchtrade/issues/new) or emailing torchtradecontact@gmail.com.

See **[Online Environments Documentation](https://torchtrade.github.io/torchtrade/environments/online/)** for setup guides and examples.

---

## Trading Platforms

Start live trading with these supported platforms:

### 🪙 Cryptocurrency Trading

**[Binance](https://accounts.binance.com/register?ref=25015935)** - Leading cryptocurrency exchange
- **Supported by:** `BinanceFuturesTorchTradingEnv`, `BinanceFuturesSLTPTorchTradingEnv`
- **Features:** Spot & futures trading, up to 125x leverage, testnet available
- **Commission:** Maker 0.02% / Taker 0.04% (with BNB discount)
- **Get Started:** [Sign up for Binance](https://accounts.binance.com/register?ref=25015935)

**[Bitget](https://www.bitget.com/)** - Fast-growing cryptocurrency exchange
- **Supported by:** `BitgetFuturesTorchTradingEnv`, `BitgetFuturesSLTPTorchTradingEnv`
- **Features:** Futures trading with up to 125x leverage, testnet for safe testing
- **Commission:** Maker 0.02% / Taker 0.06%
- **Get Started:** [Sign up for Bitget](https://www.bitget.com/)

**[Bybit](https://www.bybit.com/)** - Top cryptocurrency derivatives exchange
- **Supported by:** `BybitFuturesTorchTradingEnv`, `BybitFuturesSLTPTorchTradingEnv`
- **Features:** Futures trading with up to 100x leverage, native bracket orders (SL/TP), testnet for safe testing
- **Commission:** Maker 0.02% / Taker 0.055%
- **Get Started:** [Sign up for Bybit](https://www.bybit.com/)

**[OKX](https://www.okx.com/)** - Leading global cryptocurrency exchange
- **Supported by:** `OKXFuturesTorchTradingEnv`, `OKXFuturesSLTPTorchTradingEnv`
- **Features:** Futures trading with up to 125x leverage, bracket orders via attachAlgoOrds, demo trading
- **Commission:** Maker 0.02% / Taker 0.05%
- **Get Started:** [Sign up for OKX](https://www.okx.com/)

### 🔮 Prediction Markets

**[Polymarket](https://polymarket.com/)** - Decentralized prediction market on Polygon
- **Supported by:** `PolymarketBetEnv`
- **Features:** Rolling one-shot bets on short-cadence binary markets (BTC/ETH/SOL up-or-down at 5m / 15m / 1h / 4h / daily cadences), Gamma API market scanner, dry-run paper trading without `py-clob-client` installed
- **Get Started:** [Browse markets at Polymarket](https://polymarket.com/)

### 📈 Stock & Crypto API

**[Alpaca](https://alpaca.markets/)** - Commission-free trading API
- **Supported by:** `AlpacaTorchTradingEnv`, `AlpacaSLTPTorchTradingEnv`
- **Features:** Commission-free stocks & crypto, paper trading, real-time data
- **Best for:** US markets, algorithmic trading
- **Get Started:** [Sign up for Alpaca](https://alpaca.markets/signup)

---

## Support TorchTrade Development

- **Buy Me a Coffee**: [buymeacoffee.com/torchtrade](https://buymeacoffee.com/torchtrade)
- ⭐ **Star the repo**: Help others discover TorchTrade on [GitHub](https://github.com/TorchTrade/torchtrade)

Your support helps maintain the project, add new features, and keep documentation up-to-date!

---

<details>
<summary><h2>📦 Offline Environments</h2></summary>

All environments support both spot (`leverage=1`) and futures (`leverage>1`) trading via config.

| Environment | Bracket Orders | One-Step | Best For |
|-------------|----------------|----------|----------|
| **SequentialTradingEnv** | ❌ | ❌ | Standard sequential trading |
| **SequentialTradingEnvSLTP** | ✅ | ❌ | Risk management with SL/TP |
| **OneStepTradingEnv** | ✅ | ✅ | GRPO, contextual bandits |

See **[Offline Environments Documentation](https://torchtrade.github.io/torchtrade/environments/offline/)** for detailed guides.

</details>

<details>
<summary><h2>🚀 Training Algorithms & Examples</h2></summary>

TorchTrade includes implementations of multiple RL algorithms, all usable across any environment via Hydra config switching:

- **PPO** - `examples/online_rl/ppo/`
- **PPO + Chronos** (time series embeddings) - `examples/online_rl/ppo_chronos/`
- **DQN** - `examples/online_rl/dqn/`
- **IQL** - `examples/online_rl/iql/`
- **DSAC** - `examples/online_rl/dsac/`
- **GRPO** - `examples/online_rl/grpo/`
- **CTRL** - Research

### Run Training Examples

```bash
# PPO with default environment (sequential SLTP)
uv run python examples/online_rl/ppo/train.py

# PPO with different environments (switch via command-line)
uv run python examples/online_rl/ppo/train.py env=sequential_futures
uv run python examples/online_rl/ppo/train.py env=onestep_futures
uv run python examples/online_rl/ppo/train.py env=sequential_spot

# GRPO with default (one-step futures)
uv run python examples/online_rl/grpo/train.py

# GRPO with spot trading
uv run python examples/online_rl/grpo/train.py env=onestep_spot

# Customize with Hydra overrides
uv run python examples/online_rl/ppo/train.py \
    env=sequential_futures \
    env.symbol="ETH/USD" \
    env.leverage=10 \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

**Available environment configs** (`env=<name>`):
- `sequential_spot` - Basic spot trading
- `sequential_futures` - Basic futures trading
- `sequential_sltp` - Spot with bracket orders
- `sequential_futures_sltp` - Futures with bracket orders
- `onestep_spot` - Contextual bandit (spot)
- `onestep_futures` - Contextual bandit (futures)

See **[Examples Documentation](https://torchtrade.github.io/torchtrade/examples/)** for all available examples.

</details>

<details>
<summary><h2>🔧 Installation & Setup</h2></summary>

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer

### Full Installation

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone repository
git clone https://github.com/TorchTrade/torchtrade.git
cd torchtrade

# 3. Install dependencies
uv sync

# Optional: Install with extra features
# uv sync --extra llm              # LLM actors (OpenAI API + local vLLM/transformers)
# uv sync --extra chronos          # Chronos forecasting transforms
# uv sync --extra dev              # Development/testing tools
# uv sync --extra docs             # Documentation building
# uv sync --all-extras             # Install all optional dependencies

# 4. Activate virtual environment
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate  # Windows

# 5. For live trading, create .env file
cat > .env << EOF
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_api_secret
OKX_PASSPHRASE=your_okx_passphrase
EOF

# 6. Verify installation
uv run pytest tests/ -v
```

</details>

<details>
<summary><h2>💡 Common Use Cases</h2></summary>

### Training PPO on Backtesting Data

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
import datasets

# Load historical data from HuggingFace
df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025")
df = df["train"].to_pandas()
df['0'] = pd.to_datetime(df['0'])

# Configure multi-timeframe environment
config = SequentialTradingEnvConfig(
    trading_mode="spot",  # Long-only trading
    time_frames=["1min", "5min", "15min", "60min"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),
    initial_cash=[1000, 5000],  # Domain randomization
    transaction_fee=0.0025,
    slippage=0.001
)

env = SequentialTradingEnv(df, config)
# Train with PPO - see examples/online_rl/ppo/train.py
```

### Live Trading with Alpaca

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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

env = AlpacaTorchTradingEnv(config)
# See examples/live/alpaca/collect_live.py
```

### LLM-Based Trading

```python
from torchtrade.actor import FrontierLLMActor

# Use GPT as trading policy
policy = FrontierLLMActor(
    model="gpt-4o-mini",
    market_data_keys=env.market_data_keys,
    account_state_labels=env.account_state,  # list of label strings, e.g. ["exposure_pct", ...]
    action_levels=env.action_levels,
    debug=True,
)

tensordict = env.reset()
action = policy(tensordict)
# See examples/llm/frontier/offline.py
```

### Rule-Based Trading Strategies

```python
from torchtrade.actor import MeanReversionActor

# Use as baseline or for imitation learning
actor = MeanReversionActor(
    market_data_keys=["market_data_5Minute_24"],
)
```

### Feature Engineering

```python
import ta

def custom_preprocessing(df):
    """Add technical indicators as features"""
    df["features_open"] = df["open"]
    df["features_close"] = df["close"]
    df["features_rsi_14"] = ta.momentum.RSIIndicator(
        df["close"], window=14
    ).rsi()
    df.fillna(0, inplace=True)
    return df

config = SequentialTradingEnvConfig(
    trading_mode="spot",
    feature_preprocessing_fn=custom_preprocessing,
    time_frames=["1min", "5min"],
    window_sizes=[12, 8],
)
```

See **[Advanced Customization](https://torchtrade.github.io/torchtrade/guides/custom-features/)** for more examples.

</details>


<details>
<summary><h2>🎯 Key Concepts</h2></summary>

### Multi-Timeframe Observations

```python
config = SequentialTradingEnvConfig(
    trading_mode="spot",
    time_frames=["1min", "5min", "15min", "60min"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute")
)

# Results in observations:
# - market_data_1min: [12, num_features] - Last 12 one-minute bars
# - market_data_5min: [8, num_features] - Last 40 minutes
# - market_data_15min: [8, num_features] - Last 120 minutes
# - market_data_60min: [24, num_features] - Last 24 hours
```

### Observation Structure

```python
observation = {
    "market_data_1min": tensor([12, num_features]),
    "market_data_5min": tensor([8, num_features]),
    "account_state": tensor([6]),  # Universal 6-element state
}

# Account state (universal): [exposure_pct, position_direction, unrealized_pnl_pct,
#                             holding_time, leverage, distance_to_liquidation]
# Element definitions:
#   - exposure_pct: position_value / portfolio_value (0-1+ with leverage)
#   - position_direction: sign(position_size) (-1=short, 0=flat, +1=long)
#   - unrealized_pnl_pct: (current_price - entry_price) / entry_price * direction
#   - holding_time: steps since position opened
#   - leverage: 1.0 for spot, 1-125 for futures
#   - distance_to_liquidation: normalized distance (1.0 for spot/no position)
#
# Spot mode: position_direction in {0, +1}, leverage=1.0, distance_to_liquidation=1.0
# Futures mode: position_direction in {-1, 0, +1}, leverage=1-125, calculated distance
```

### Action Spaces

**Standard (3 actions):**
- Action 0: SELL/SHORT
- Action 1: HOLD
- Action 2: BUY/LONG

**SLTP Combinatorial:**
- Action 0: HOLD
- Actions 1..N: BUY/LONG with (SL, TP) combinations
- Actions N+1..2N: SHORT with (SL, TP) combinations (futures only)

See **[Advanced Customization](https://torchtrade.github.io/torchtrade/guides/custom-features/)** for detailed explanations.

</details>

<details>
<summary><h2>⚙️ Configuration with Hydra</h2></summary>

TorchTrade uses Hydra for configuration management with a defaults list pattern:

```yaml
# examples/online_rl/ppo/config.yaml
defaults:
  - env: sequential_sltp  # Load environment config
  - _self_

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000

optim:
  lr: 2.5e-4
  anneal_lr: true
  max_grad_norm: 0.5

loss:
  gamma: 0.9
  clip_epsilon: 0.1
  entropy_coef: 0.01
```

```yaml
# examples/online_rl/env/sequential_sltp.yaml
env:
  name: SequentialTradingEnvSLTP
  trading_mode: "spot"
  symbol: "BTC/USD"
  time_frames: ["5Min", "15Min"]
  window_sizes: [10, 10]
  execute_on: "15Min"
  initial_cash: [1000, 5000]
  transaction_fee: 0.0025
  # ... more env config
```

Override from command line:

```bash
# Switch environment entirely
uv run python examples/online_rl/ppo/train.py env=sequential_futures

# Override specific parameters
uv run python examples/online_rl/ppo/train.py \
    env.symbol="ETH/USD" \
    env.leverage=10 \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

</details>


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

# Build documentation
mkdocs serve
```

### Reporting Issues

Found a bug or have a feature request?

[GitHub Issues](https://github.com/TorchTrade/torchtrade/issues)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Support

- 📧 **Email**: torchtradecontact@gmail.com

---

**Built with TorchRL • Designed for Algorithmic Trading • Open Source**
