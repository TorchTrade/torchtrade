# Getting Started with TorchTrade

This guide will help you install TorchTrade and run your first trading environment.

## Prerequisites

- Python 3.10 or higher
- CUDA (optional, for GPU acceleration)
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer

## Installation

### 1. Install UV

UV is a fast Python package installer and environment manager:

```bash
# On Unix/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```bash
git clone https://github.com/TorchTrade/torchtrade.git
cd torchtrade
```

### 3. Install Dependencies

```bash
# Install TorchTrade and all dependencies
uv sync

# Optional: Install with extra features
uv sync --extra llm              # LLM actors (OpenAI API + local vLLM/transformers)
uv sync --extra chronos          # Chronos forecasting transforms
uv sync --all-extras             # Install all optional dependencies

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### 4. Verify Installation

```bash
# Run tests to verify everything works
uv run pytest tests/ -v
```

## Your First Environment

Let's create a simple trading environment using historical OHLCV data.

### Step 1: Prepare Your Data

TorchTrade expects OHLCV data with the following columns:

```python
import pandas as pd

# Your data should have these columns
df = pd.DataFrame({
    'timestamp': [...],  # datetime or parseable strings
    'open': [...],       # opening prices
    'high': [...],       # high prices
    'low': [...],        # low prices
    'close': [...],      # closing prices
    'volume': [...]      # trading volume
})
```

**Note**: You can also use our pre-processed datasets from [HuggingFace/Torch-Trade](https://huggingface.co/Torch-Trade) which include various cryptocurrency pairs with 1-minute OHLCV data.

### Step 2: Create an Environment

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
import pandas as pd

# Load your data
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Configure environment
config = SequentialTradingEnvConfig(
    time_frames=["1min", "5min", "15min"],        # 1m, 5m, 15m bars
    window_sizes=[12, 8, 8],       # Lookback windows
    execute_on=(5, "Minute"),      # Execute every 5 minutes
    initial_cash=1000,             # Starting capital
    transaction_fee=0.0025,        # 0.25% fee
    slippage=0.001                 # 0.1% slippage
)

# Create environment
env = SequentialTradingEnv(df, config)
```

## Training Your First Policy

### Quick Training Run

```bash
# Train PPO with default config
uv run python examples/online_rl/ppo/train.py

# Customize with Hydra overrides
uv run python examples/online_rl/ppo/train.py \
    env.symbol="BTC/USD" \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

See the **[Examples](examples.md)** page for all available algorithms (PPO, DQN, IQL, DSAC, GRPO) and usage guides.

## Common Use Cases

### Loading Historical Data from HuggingFace

```python
import datasets

# Load pre-processed Bitcoin data
ds = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025")
df = ds["train"].to_pandas()
df['0'] = pd.to_datetime(df['0'])  # First column is timestamp
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Create environment
env = SequentialTradingEnv(df, config)
```

### Multi-Timeframe Configuration

```python
config = SequentialTradingEnvConfig(
    time_frames=["1min", "5min", "15min", "60min"],        # 1m, 5m, 15m, 1h
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute"),          # Execute every 5 minutes
    initial_cash=[1000, 5000],         # Domain randomization
)
```

The environment will provide observations:
- `market_data_1Minute`: [12, num_features] - Last 12 one-minute bars
- `market_data_5Minute`: [8, num_features] - Last 40 minutes
- `market_data_15Minute`: [8, num_features] - Last 120 minutes
- `market_data_60Minute`: [24, num_features] - Last 24 hours

### Using Stop-Loss / Take-Profit

```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig

config = SequentialTradingEnvSLTPConfig(
    stoploss_levels=[-0.02, -0.05],     # -2%, -5%
    takeprofit_levels=[0.05, 0.10],     # +5%, +10%
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SequentialTradingEnvSLTP(df, config)

# Action space: 1 (HOLD) + 2×2 (SL/TP combinations) = 5 actions
# Action 0: HOLD
# Action 1: BUY with SL=-2%, TP=+5%
# Action 2: BUY with SL=-2%, TP=+10%
# Action 3: BUY with SL=-5%, TP=+5%
# Action 4: BUY with SL=-5%, TP=+10%
```

## Live Trading Setup

For live trading with real exchanges, you'll need API credentials.

### Alpaca (US Stocks & Crypto)

Alpaca offers commission-free paper trading for testing strategies without risk. See [Alpaca Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading) for API credentials setup.

```bash
# Create .env file
cat > .env << EOF
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
EOF
```

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min", "5Min"],
    window_sizes=[12, 8],
    execute_on="5Min",
    paper=True  # Start with paper trading!
)

env = AlpacaTorchTradingEnv(config)
```

### Binance (Crypto Futures)

If you want to trade on Binance, register [here](https://accounts.binance.com/register?ref=25015935) in case you have no account. Binance also allows for demo trading, see [here](https://www.binance.com/en/square/post/14316321292186).

```bash
# Add to .env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
```

```python
from torchtrade.envs.binance import (
    BinanceFuturesTorchTradingEnv,
    BinanceFuturesTradingEnvConfig
)

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",
    leverage=5,                        # 5x leverage
    quantity_per_trade=0.01,
    demo=True,                         # Use testnet
)

env = BinanceFuturesTorchTradingEnv(config)
```

**Note**: Alpaca and Binance are just two examples of live environments/brokers that TorchTrade supports. For more details on all available exchanges and configurations, see **[Online Environments](environments/online.md)**. We're always open to including additional brokers - if you'd like to request support for a new exchange, please [create an issue](https://github.com/TorchTrade/torchtrade/issues) or contact us directly at torchtradecontact@gmail.com.

## Next Steps

- **[Offline Environments](environments/offline.md)** - Deep dive into backtesting environments
- **[Online Environments](environments/online.md)** - Live trading with exchange APIs
- **[Examples](examples.md)** - Training scripts for all algorithms

??? tip "Troubleshooting"
    - **"No module named 'torchtrade'"** → Activate the venv: `source .venv/bin/activate`
    - **"CUDA out of memory"** → Reduce `frames_per_batch` or set `device: "cpu"`
    - **"Columns do not match required format"** → Ensure columns are `['timestamp', 'open', 'high', 'low', 'close', 'volume']`
    - **"Environment returns NaN rewards"** → Check for NaN/zero prices: `df = df.dropna()`
