# Alpaca Live Trading Examples

This directory contains example scripts for live trading with Alpaca using TorchRL environments.

## Overview

| Script | Environment | Action Space | Description |
|--------|-------------|--------------|-------------|
| `collect_live.py` | `AlpacaTorchTradingEnv` | 3 actions (sell/hold/buy) | Standard live data collection |
| `collect_live_sltp.py` | `AlpacaSLTPTorchTradingEnv` | N actions (hold + SL/TP combos) | Live collection with bracket orders |
| `collect_live_llm.py` | `AlpacaTorchTradingEnv` | 3 actions (sell/hold/buy) | LLM-based trading using OpenAI |

## Environment Comparison

### Standard Environment (`AlpacaTorchTradingEnv`)

Simple 3-action discrete environment:
- **Action 0**: Sell all
- **Action 1**: Hold (do nothing)
- **Action 2**: Buy all

Account state (7 elements):
```
[cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
```

### SL/TP Environment (`AlpacaSLTPTorchTradingEnv`)

Combinatorial action space using Alpaca bracket orders:
- **Action 0**: Hold (do nothing)
- **Actions 1..N**: Buy with specific (stop_loss%, take_profit%) combination

Account state (7 elements):
```
[cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
```

Example with `stoploss_levels=(-0.02, -0.05)` and `takeprofit_levels=(0.05, 0.10)`:
```
Action 0: HOLD
Action 1: BUY with SL=-2%, TP=+5%
Action 2: BUY with SL=-2%, TP=+10%
Action 3: BUY with SL=-5%, TP=+5%
Action 4: BUY with SL=-5%, TP=+10%
```

## Requirements

1. Create a `.env` file in the project root:
```bash
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
```

2. Install dependencies:
```bash
pip install alpaca-py python-dotenv ta torchrl
```

## Configuration Options

### Timeframes and Window Sizes

```python
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),   # 1-minute bars
        TimeFrame(5, TimeFrameUnit.Minute),   # 5-minute bars
        TimeFrame(15, TimeFrameUnit.Minute),  # 15-minute bars
        TimeFrame(1, TimeFrameUnit.Hour),     # 1-hour bars
    ],
    window_sizes=[12, 8, 8, 24],  # Number of bars per timeframe
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),  # Trade execution frequency
)
```

### SL/TP Levels (for SL/TP environment)

```python
config = AlpacaSLTPTradingEnvConfig(
    # Stop loss levels (negative percentages)
    stoploss_levels=(-0.01, -0.02, -0.05, -0.10),  # -1%, -2%, -5%, -10%

    # Take profit levels (positive percentages)
    takeprofit_levels=(0.02, 0.05, 0.10, 0.20),    # +2%, +5%, +10%, +20%

    # Results in: 1 + (4 * 4) = 17 actions
)
```

### Custom Feature Preprocessing

You can provide a custom preprocessing function to engineer features from OHLCV data:

```python
import pandas as pd
import numpy as np
import ta

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom feature engineering for OHLCV data.

    Input columns: ["open", "high", "low", "close", "volume"]
    Output: DataFrame with columns starting with "features_"
    """
    df = df.copy().reset_index(drop=True)

    # Log returns
    df["features_return_log"] = np.log(df["close"]).diff()

    # RSI
    df["features_rsi_14"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    # MACD
    macd = ta.trend.MACD(close=df["close"])
    df["features_macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20)
    df["features_bb_pct"] = bb.bollinger_pband()

    # Drop NaN rows
    df.dropna(inplace=True)

    return df

# Use in environment
env = AlpacaTorchTradingEnv(
    config,
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("SECRET_KEY"),
    feature_preprocessing_fn=custom_preprocessing,
)
```

## Using a Trained Policy

Instead of random actions, you can use a trained policy for data collection or live trading.

### Loading a Pre-trained IQL Policy

```python
import torch
from examples.live.policies.iql_policy import make_discrete_iql_model

# Create the policy model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = make_discrete_iql_model(device)

# Load trained weights
policy.load_state_dict(torch.load("path/to/iql_policy.pth", map_location=device))
policy.eval()

# Use with collector
from torchrl.collectors import SyncDataCollector

collector = SyncDataCollector(
    env,
    policy,  # Use trained policy instead of None (random)
    frames_per_batch=1,
    total_frames=10000,
    device=device,
)
```

### Creating a Custom Policy for SL/TP Environment

For the SL/TP environment, you need a policy that outputs actions matching the combinatorial action space:

```python
from torch.distributions import Categorical
from torchrl.data import Categorical as CategoricalSpec
from torchrl.modules import MLP, ProbabilisticActor, SafeModule
from tensordict.nn import InteractionType

def make_sltp_policy(num_actions: int, device="cpu"):
    """
    Create a policy for the SL/TP environment.

    Args:
        num_actions: Size of action space (1 + len(sl_levels) * len(tp_levels))
    """
    # Simple MLP policy (customize architecture as needed)
    actor_net = MLP(
        in_features=7,  # account_state size (same for both envs)
        num_cells=[128, 128],
        out_features=num_actions,
        activation_class=torch.nn.ReLU,
        device=device,
    )

    actor_module = SafeModule(
        module=actor_net,
        in_keys=["account_state"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        spec=Composite(action=CategoricalSpec(num_actions)).to(device),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    return actor

# Usage
env = AlpacaSLTPTorchTradingEnv(config, ...)
policy = make_sltp_policy(num_actions=env.action_spec.n, device=device)
```

### Using Market Data Encoders

For more sophisticated policies that use market data, see `examples/live/policies/iql_policy.py` which demonstrates:

- Multi-timeframe encoders (BiNMTABLModel)
- Account state encoding
- Sequential encoding pipeline

```python
from examples.live.policies.iql_policy import make_discrete_iql_model

# This creates:
# - Separate encoders for each timeframe (1min, 5min, 15min, 1h)
# - Account state encoder
# - Actor network that combines all encodings
policy = make_discrete_iql_model(device)
```

## Using an LLM for Trading

You can use a Large Language Model (LLM) as the trading policy instead of a trained neural network. The `LLMActor` class wraps OpenAI's API to make trading decisions based on market data.

### Requirements

Add your OpenAI API key to the `.env` file:
```bash
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
OPENAI_API_KEY=your_openai_api_key
```

### How LLMActor Works

The `LLMActor` converts the environment's TensorDict into a structured prompt containing:
- **Account state**: cash, position size, entry price, unrealized P&L, etc.
- **Market data**: OHLCV data from multiple timeframes formatted as tables

The LLM receives this data with a system prompt instructing it to think step-by-step and output a trading decision (`buy`, `sell`, or `hold`).

### Usage Example

```python
from torchtrade.actor.llm_actor import LLMActor

# Create LLM actor with OpenAI model
policy = LLMActor(model="gpt-4o-mini", debug=True)

# Use with collector
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=1,
    total_frames=1000,
    device=device,
    trust_policy=True,
)
```

### Custom LLM Configuration

You can customize the `LLMActor` by modifying:
- `model`: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o")
- `system_prompt`: Trading instructions for the LLM
- `market_data_keys`: Which timeframe data to include in prompts
- `features_keys`: Which OHLCV features to show

```python
from torchtrade.actor.llm_actor import LLMActor

class CustomLLMActor(LLMActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
        You are a conservative trading agent. Only buy when RSI < 30.
        Only sell when RSI > 70. Otherwise hold.
        Output: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>
        """
```

### LLM Response Format

The LLM is prompted to use a structured response format:
```
<think>
[Step-by-step reasoning about the market conditions]
</think>
<answer>buy</answer>
```

The `LLMActor` extracts the action from the `<answer>` tags and optionally stores the reasoning in the TensorDict.

## Running the Examples

### Random Data Collection (Standard Environment)

```bash
python examples/live/alpaca/collect_live.py
```

### Random Data Collection (SL/TP Environment)

```bash
python examples/live/alpaca/collect_live_sltp.py
```

### LLM-Based Trading

```bash
python examples/live/alpaca/collect_live_llm.py
```

### With Trained Policy

Modify the script to load your trained policy:

```python
# In collect_live.py or collect_live_sltp.py

from examples.live.policies.iql_policy import make_discrete_iql_model

# Create and load policy
policy = make_discrete_iql_model(device)
policy.load_state_dict(torch.load("your_trained_policy.pth", map_location=device))
policy.eval()

# Pass to collector
collector = make_collector(
    env,
    policy=policy,  # Changed from None
    frames_per_batch=1,
    total_frames=total_farming_steps,
)
```

## Output

All scripts save collected data to a replay buffer:
- `replay_buffer_random.pt` - Standard environment
- `replay_buffer_random_sltp.pt` - SL/TP environment
- `replay_buffer_gpt5mini.pt` - LLM-based trading (includes thinking traces)

These can be loaded for offline training:

```python
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(10_000),
    batch_size=32,
)
replay_buffer.loads("replay_buffer_random.pt")

# Sample a batch
batch = replay_buffer.sample()
```

## Paper vs Live Trading

Both environments support paper trading (default) and live trading:

```python
config = AlpacaTradingEnvConfig(
    paper=True,   # Paper trading (safe for testing)
    # paper=False,  # Live trading (real money!)
)
```

**Warning**: Always test thoroughly with paper trading before enabling live trading.
