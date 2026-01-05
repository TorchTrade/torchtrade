# Rule-Based Actors for Imitation Learning

This directory contains examples demonstrating how to use rule-based trading actors for imitation learning pre-training.

## Overview

Rule-based actors implement deterministic trading strategies using technical indicators. They serve as "expert" policies that can:

1. **Generate demonstrations** for behavioral cloning (imitation learning)
2. **Bootstrap RL training** with prior knowledge instead of random initialization
3. **Provide baselines** for evaluating learned policies

See [Issue #54](https://github.com/TorchTrade/torchtrade_envs/issues/54) for the full motivation and implementation plan.

## Available Actors

### Spot Trading Actors (`SeqLongOnlyEnv`)

- **`MomentumActor`**: Momentum trading strategy (long in uptrends, flat otherwise)
- **`MeanReversionActor`**: Mean reversion strategy (buy oversold, sell overbought)
- **`BreakoutActor`**: Breakout strategy using Bollinger Bands

### SL/TP Actors (`SeqLongOnlySLTPEnv`)

- **`MomentumSLTPActor`**: Momentum with dynamic SL/TP selection
- **`MeanReversionSLTPActor`**: Mean reversion with dynamic SL/TP selection
- **`BreakoutSLTPActor`**: Breakout with dynamic SL/TP selection

### Futures Actors (`SeqFuturesEnv`)

- **`MomentumFuturesActor`**: Momentum strategy (long/short/flat)
- **`MeanReversionFuturesActor`**: Mean reversion (fade extremes)
- **`BreakoutFuturesActor`**: Breakout strategy with shorting capability

## Quick Start

### Tune Expert Hyperparameters

Find optimal hyperparameters for an expert on your data:

```bash
# Tune MomentumActor parameters
python tune_expert_hyperparameters.py \
    --expert momentum \
    --train_episodes 20 \
    --test_episodes 50

# Save best configuration and trading plot
python tune_expert_hyperparameters.py \
    --expert mean_reversion \
    --save_config best_config.json \
    --save_plot trading_history.png

# Use custom data and test split
python tune_expert_hyperparameters.py \
    --expert breakout \
    --data_path Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025 \
    --test_split_start "2025-01-01"
```

### Collect Demonstrations from All Experts

```bash
python collect_demonstrations.py \
    --expert all \
    --num_episodes 100 \
    --save_path demos.pt
```

### Collect from Specific Expert

```bash
python collect_demonstrations.py \
    --expert momentum \
    --num_episodes 200 \
    --save_path momentum_demos.pt
```

### Evaluate Expert Performance

```bash
python collect_demonstrations.py \
    --expert all \
    --num_episodes 0 \
    --eval_episodes 50
```

## Usage Examples

### Example 1: Basic Usage

```python
from torchtrade.actor import MomentumActor
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import pandas as pd

# Create environment
df = pd.read_csv("btc_ohlcv.csv")
config = SeqLongOnlyEnvConfig()
env = SeqLongOnlyEnv(df, config)

# Create actor
actor = MomentumActor(
    market_data_keys=["market_data_1Minute_12"],
    momentum_window=10,
    volatility_threshold=0.02,
)

# Run episode
obs = env.reset()
done = False
while not done:
    obs = actor(obs)  # Actor sets action in obs
    obs = env.step(obs)
    done = obs["done"].item()
```

### Example 2: Create Expert Ensemble

```python
from torchtrade.actor import create_expert_ensemble

# Create ensemble of all experts
experts = create_expert_ensemble(
    market_data_keys=["market_data_5Minute_24"],
    env_type="spot"  # or "sltp", "futures"
)

# Collect diverse demonstrations
for expert in experts:
    print(f"Running {expert.__class__.__name__}")
    # ... collect demonstrations
```

### Example 3: SLTP Environment

```python
from torchtrade.actor import MomentumSLTPActor
from torchtrade.envs import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
)
env = SeqLongOnlySLTPEnv(df, config)

actor = MomentumSLTPActor(
    market_data_keys=["market_data_5Minute_24"],
    stoploss_levels=config.stoploss_levels,
    takeprofit_levels=config.takeprofit_levels,
)

obs = actor(env.reset())
# Action will be 0 (HOLD) or 1-4 (BUY with different SL/TP combinations)
print(f"Selected action: {obs['action'].item()}")
```

### Example 4: Futures Environment

```python
from torchtrade.actor import MomentumFuturesActor
from torchtrade.envs import SeqFuturesEnv, SeqFuturesEnvConfig

config = SeqFuturesEnvConfig(leverage=10)
env = SeqFuturesEnv(df, config)

actor = MomentumFuturesActor(
    market_data_keys=["market_data_15Minute_24"],
)

obs = actor(env.reset())
# Action: 0=SHORT, 1=HOLD/CLOSE, 2=LONG
print(f"Selected action: {obs['action'].item()}")
```

## Strategy Details

### Momentum Strategy

**Logic:**
- Calculate short-term momentum (mean return over last N bars)
- Calculate volatility (std dev of returns)
- Go long when momentum > threshold and volatility < threshold
- Go short when momentum < -threshold and volatility < threshold
- Hold otherwise

**Parameters:**
- `momentum_window`: Default 10 bars
- `volatility_window`: Default 20 bars
- `momentum_threshold`: Default 0.01 (1%)
- `volatility_threshold`: Default 0.02 (2%)

**Expected Performance:**
- Sharpe: 0.5 to 1.0
- Action distribution: 40% long, 40% short/sell, 20% hold
- Works in: Trending markets
- Fails in: Ranging markets

### Mean Reversion Strategy

**Logic:**
- Calculate moving average (MA)
- Calculate deviation from MA
- Buy when price < MA - threshold (oversold)
- Sell when price > MA + threshold (overbought)
- Hold when near MA

**Parameters:**
- `ma_window`: Default 20 bars
- `deviation_threshold`: Default 0.02 (2%)

**Expected Performance:**
- Sharpe: 0.3 to 0.8
- Action distribution: 30% long, 30% short/sell, 40% hold
- Works in: Ranging/sideways markets
- Fails in: Strong trends

### Breakout Strategy

**Logic:**
- Calculate Bollinger Bands (MA ± N * std)
- Buy when price breaks above upper band
- Sell when price breaks below lower band
- Hold when within bands

**Parameters:**
- `bb_window`: Default 20 bars
- `bb_std`: Default 2.0 (number of standard deviations)

**Expected Performance:**
- Sharpe: 0.2 to 1.5 (high variance)
- Action distribution: 25% long, 25% short/sell, 50% hold
- Works in: Volatile markets with breakouts
- Fails in: Many false breakouts

## Hyperparameter Tuning

The `tune_expert_hyperparameters.py` script helps you find optimal parameters for each expert strategy on your specific market data.

### How It Works

1. **Load Data**: Loads historical OHLCV data from HuggingFace
2. **Train/Test Split**: Splits data by date (e.g., 2025-01-01)
3. **Grid Search**: Tests all hyperparameter combinations on train data
4. **Evaluation**: Evaluates best config on test data with visualization

### Hyperparameter Search Spaces

**MomentumActor:**
```python
{
    "momentum_window": [5, 10, 15, 20],
    "volatility_window": [10, 20, 30],
    "momentum_threshold": [0.005, 0.01, 0.015, 0.02],
    "volatility_threshold": [0.015, 0.02, 0.025, 0.03],
}
```

**MeanReversionActor:**
```python
{
    "ma_window": [10, 15, 20, 25, 30],
    "deviation_threshold": [0.01, 0.015, 0.02, 0.025, 0.03],
}
```

**BreakoutActor:**
```python
{
    "bb_window": [15, 20, 25, 30],
    "bb_std": [1.5, 2.0, 2.5, 3.0],
}
```

### Example Output

```
============================================================
TUNING RESULTS: MOMENTUM EXPERT
============================================================

Best Hyperparameters (Train Set):
  momentum_window: 10
  volatility_window: 20
  momentum_threshold: 0.01
  volatility_threshold: 0.02

Train Set Performance:
  Mean Return:        45.23
  Std Return:         12.34
  Sharpe Ratio:        3.667
  Win Rate:           65.0%
  Max Return:         78.90
  Min Return:         12.34
  Avg Length:        234.5

Test Set Performance:
  Mean Return:        42.10
  Std Return:         14.20
  Sharpe Ratio:        2.965
  Win Rate:           60.0%
  Max Return:         71.23
  Min Return:          8.91
  Avg Length:        245.2

Generalization (Test vs Train):
  Sharpe Difference: -0.702
  ✓ Similar performance on test set
```

The script also generates a **trading history plot** showing portfolio value over time on the test set.

### Custom Hyperparameter Grid

You can provide a custom grid via JSON file:

```bash
# Create custom_grid.json
cat > custom_grid.json << EOF
{
  "momentum_window": [8, 12, 16],
  "volatility_window": [15, 25],
  "momentum_threshold": [0.008, 0.012],
  "volatility_threshold": [0.018, 0.022]
}
EOF

# Use custom grid
python tune_expert_hyperparameters.py \
    --expert momentum \
    --custom_grid custom_grid.json
```

### Loading Best Configuration

After tuning, you can load and use the best configuration:

```python
import json
from torchtrade.actor import MomentumActor

# Load best config
with open("best_config.json", 'r') as f:
    config = json.load(f)

# Create actor with best hyperparameters
actor = MomentumActor(
    market_data_keys=["market_data_5Minute_24"],
    **config["best_hyperparams"]
)

# Now use actor for demonstration collection or evaluation
```

## Next Steps: Behavioral Cloning

Once you've collected demonstrations, you can use them for behavioral cloning:

```python
import torch
from torch.utils.data import DataLoader

# Load demonstrations
demos = torch.load("demos.pt")

# Create DataLoader
loader = DataLoader(demos, batch_size=256, shuffle=True)

# Train policy to imitate expert actions
for epoch in range(50):
    for batch in loader:
        obs = batch["observation"]
        expert_actions = batch["action"]

        # Forward pass
        logits = policy(obs)
        loss = criterion(logits, expert_actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Then fine-tune with PPO for better performance!

## Command Line Arguments

### `tune_expert_hyperparameters.py`

```
--expert               Which expert to tune (momentum|mean_reversion|breakout)
--data_path           HuggingFace dataset path (default: Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025)
--test_split_start    Test split start date (default: 2025-01-01)
--train_episodes      Number of episodes for training evaluation (default: 20)
--test_episodes       Number of episodes for test evaluation (default: 50)
--save_config         Path to save best configuration JSON (optional)
--save_plot          Path to save trading history plot (optional)
--custom_grid        Path to custom hyperparameter grid JSON (optional)
--seed               Random seed (default: 42)
```

### `collect_demonstrations.py`

```
--expert               Which expert to use (momentum|mean_reversion|breakout|all)
--num_episodes         Number of episodes to collect per expert (default: 100)
--eval_episodes        Number of episodes for evaluation (default: 10)
--save_path           Path to save demonstrations (optional)
--data_path           HuggingFace dataset path (default: Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025)
--test_split_start    Test split start date (default: 2025-01-01)
--debug               Enable debug mode for actors
```

## References

- **Issue #54**: [Implement Imitation Learning Pre-training + RL Fine-tuning](https://github.com/TorchTrade/torchtrade_envs/issues/54)
- **Behavioral Cloning**: Pomerleau (1988) - "ALVINN: An Autonomous Land Vehicle in a Neural Network"
- **DAgger**: Ross et al. (2011) - "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
- **GAIL**: Ho & Ermon (2016) - "Generative Adversarial Imitation Learning"
