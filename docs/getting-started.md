# Getting Started with TorchTrade

This guide will help you install TorchTrade and run your first trading environment.

## Prerequisites

- Python 3.8 or higher
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
uv sync --extra llm              # OpenAI API support for LLMActor
uv sync --extra llm_local        # Local LLM inference (vLLM, transformers)
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
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import pandas as pd

# Load your data
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Configure environment
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min"],        # 1m, 5m, 15m bars
    window_sizes=[12, 8, 8],       # Lookback windows
    execute_on=(5, "Minute"),      # Execute every 5 minutes
    initial_cash=1000,             # Starting capital
    transaction_fee=0.0025,        # 0.25% fee
    slippage=0.001                 # 0.1% slippage
)

# Create environment
env = SeqLongOnlyEnv(df, config)
```

### Step 3: Run the Environment

```python
# Reset environment
tensordict = env.reset()

print("Initial observation keys:", tensordict.keys())
print("Market data shape:", tensordict["market_data_5Minute"].shape)
print("Account state:", tensordict["account_state"])

# Take an action
tensordict["action"] = torch.tensor([2])  # BUY action
tensordict = env.step(tensordict)

print(f"Reward: {tensordict['reward'].item()}")
print(f"Done: {tensordict['done'].item()}")
```

**Note**: TorchTrade uses a default log-return reward function, but you can customize it to shape agent behavior. See **[Reward Functions](guides/reward-functions.md)** for examples including transaction cost penalties, Sharpe ratio rewards, and more.

## Training Your First Policy

Let's train a PPO policy on the long-only environment.

### Quick Training Run

```bash
# Train PPO with default config
uv run python examples/online/ppo/train.py

# Customize with Hydra overrides
uv run python examples/online/ppo/train.py \
    env.symbol="BTC/USD" \
    optim.lr=1e-4 \
    loss.gamma=0.95
```

**Note**: TorchTrade provides several example training scripts (PPO, IQL, DSAC, GRPO, etc.) designed for inspiration and learning. These examples follow the structure of [TorchRL's SOTA implementations](https://github.com/pytorch/rl/tree/main/sota-implementations), enabling near plug-and-play compatibility with any TorchRL algorithm. See the **[Examples](examples.md)** page for a complete list and usage guide.

### Understanding the Training Script

The training script structure:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.collectors import SyncDataCollector
from torchrl.modules import TanhNormal, ProbabilisticActor

# 1. Create environment
config = SeqLongOnlyEnvConfig(...)
env = SeqLongOnlyEnv(df, config)

# 2. Create your custom policy architecture
# TorchTrade provides simple default networks in the examples
# See torchtrade/models/simple_encoders.py for reference implementations
# Check the examples/ directory for more details on network architectures
actor_net = YourCustomNetwork(...)
policy = ProbabilisticActor(...)

# 3. Create data collector
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=10000,
    total_frames=1_000_000
)

# 4. Training loop
for batch in collector:
    loss = loss_module(batch)
    loss.backward()
    optimizer.step()
```

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
env = SeqLongOnlyEnv(df, config)
```

### Multi-Timeframe Configuration

```python
config = SeqLongOnlyEnvConfig(
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
from torchtrade.envs.offline import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05],     # -2%, -5%
    takeprofit_levels=[0.05, 0.10],     # +5%, +10%
    time_frames=["1min", "5min", "15min"],
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

Now that you have the basics, explore these topics:

- **[Offline Environments](environments/offline.md)** - Deep dive into backtesting environments
- **[Online Environments](environments/online.md)** - Live trading with exchange APIs
- **[Feature Engineering](guides/custom-features.md)** - Add technical indicators
- **[Reward Functions](guides/reward-functions.md)** - Design better reward signals
- **[Performance Metrics](guides/metrics.md)** - Evaluate agent performance with trading metrics
- **[Understanding the Sampler](guides/sampler.md)** - How multi-timeframe sampling works
- **[Building Custom Environments](guides/custom-environment.md)** - Extend TorchTrade

## Troubleshooting

### Common Issues

**Issue: "No module named 'torchtrade'"**

Solution: Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate  # Windows
```

**Issue: "CUDA out of memory"**

Solution: Reduce batch size or use CPU:
```python
# In training config
collector:
  frames_per_batch: 50000  # Reduce from 100000
  device: "cpu"  # Use CPU instead of CUDA
```

**Issue: "Columns do not match required format"**

Solution: Ensure your DataFrame has the exact column names:
```python
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
```

**Issue: "Environment returns NaN rewards"**

Solution: Check for invalid price data:
```python
# Remove NaN values
df = df.dropna()

# Check for zero/negative prices
assert (df[['open', 'high', 'low', 'close']] > 0).all().all()
```

## Getting Help

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade/issues)
- 📧 **Email**: torchtradecontact@gmail.com
