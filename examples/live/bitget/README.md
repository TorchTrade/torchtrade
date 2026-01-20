# Bitget Live Trading Environment

This directory contains examples for live trading with Bitget Futures using the TorchTrade framework. The implementation uses [CCXT](https://github.com/ccxt/ccxt) library to interface with Bitget's V2 API.

## Overview

TorchTrade provides two Bitget futures trading environments:

### 1. BitgetFuturesTorchTradingEnv
A simple discrete action space environment for futures trading:
- **Action 0 (-1.0)**: Go SHORT (or close if in position)
- **Action 1 (0.0)**: HOLD current position (do nothing)
- **Action 2 (1.0)**: Go LONG (or close if in position)

### 2. BitgetFuturesSLTPTorchTradingEnv
Advanced environment with Stop-Loss/Take-Profit combinatorial action space:
- **Action 0**: HOLD (do nothing)
- **Actions 1..N**: LONG with specific (stop_loss_pct, take_profit_pct) combination
- **Actions N+1..M**: SHORT with specific (stop_loss_pct, take_profit_pct) combination (if enabled)

The environment automatically closes positions when either SL or TP is triggered by Bitget's order system.

## Account State

Both environments provide a 10-element account state tensor:

```python
[
    cash,                    # Available balance
    position_size,           # Position quantity (positive=long, negative=short)
    position_value,          # Absolute notional value of position
    entry_price,             # Entry price of current position
    current_price,           # Current mark price
    unrealized_pnl_pct,      # Unrealized PnL as percentage
    leverage,                # Current leverage
    margin_ratio,            # Position value / total balance
    liquidation_price,       # Liquidation price
    holding_time,            # Steps holding current position
]
```

## Requirements

### 1. Dependencies
```bash
uv add ccxt>=4.0.0
```

### 2. Bitget API Credentials

You need a Bitget account with API access. For testing, use the [Bitget Demo Trading](https://www.bitget.com/demo-trading) environment.

Create a `.env` file in your project root:
```env
BITGETACCESSAPIKEY=your_api_key_here
BITGETSECRETKEY=your_secret_key_here
BITGETPASSPHRASE=your_passphrase_here
```

**Important**: Bitget requires all three credentials (key, secret, AND passphrase).

### 3. Symbol Format

Bitget environments use CCXT's perpetual swap format:
```python
symbol = "BTC/USDT:USDT"  # Correct format
# NOT "BTCUSDT" (old V1 format)
```

Format breakdown: `{BASE}/{QUOTE}:{SETTLE}`
- BTC: Base currency
- USDT: Quote currency
- USDT: Settlement currency (after colon)

## Configuration Options

### Trading Parameters

```python
config = BitgetFuturesTradingEnvConfig(
    # Symbol (CCXT perpetual swap format)
    symbol="BTC/USDT:USDT",

    # Timeframes for market data
    time_frames=["5min", "15min"],  # Can be list or single timeframe
    window_sizes=[6, 32],            # History window for each timeframe
    execute_on="1min",               # Timeframe for trade execution timing

    # Trading parameters
    product_type="USDT-FUTURES",     # V2 API: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES
    leverage=5,                       # Leverage (1-125)
    quantity_per_trade=0.002,        # Base quantity per trade
    trade_mode=TradeMode.QUANTITY,   # QUANTITY or CASH mode

    # Margin mode
    margin_mode=MarginMode.ISOLATED,  # ISOLATED or CROSSED

    # Position mode
    position_mode=PositionMode.ONE_WAY,  # ONE_WAY (recommended) or HEDGE

    # Environment settings
    demo=True,                        # Use testnet (True) or production (False)
    include_base_features=False,      # Include raw OHLCV data in observations
    close_position_on_reset=False,    # Close positions on env.reset()

    # Termination
    done_on_bankruptcy=True,          # End episode if balance too low
    bankrupt_threshold=0.1,           # 10% of initial balance
)
```

### Margin Modes

**ISOLATED (Recommended)**:
- Each position has its own separate margin allocation
- If liquidated, only that position's margin is lost
- Lower risk, better for beginners
- Example: Open BTC position with 100 USDT margin - only that 100 USDT is at risk

**CROSSED**:
- All positions share the entire account balance as margin
- If one position liquidates, it can affect other positions
- Higher risk but more capital efficient
- Example: Multiple positions share your 1000 USDT balance

```python
from torchtrade.envs.bitget.futures_order_executor import MarginMode

config = BitgetFuturesTradingEnvConfig(
    margin_mode=MarginMode.ISOLATED,  # Safer default
    # margin_mode=MarginMode.CROSSED,  # Advanced users only
)
```

### Position Modes

**ONE_WAY (Recommended)**:
- Single net position per symbol
- Going LONG when SHORT automatically closes the short first
- Simpler to manage, recommended for most strategies
- Example: Can only be either long OR short BTC at any time

**HEDGE**:
- Can hold separate long and short positions simultaneously
- Requires explicit position management
- More complex, typically for advanced hedging strategies
- Example: Can be long 0.1 BTC and short 0.2 BTC at the same time

```python
from torchtrade.envs.bitget.futures_order_executor import PositionMode

config = BitgetFuturesTradingEnvConfig(
    position_mode=PositionMode.ONE_WAY,  # Recommended for most use cases
    # position_mode=PositionMode.HEDGE,   # Advanced hedging strategies
)
```

### SLTP Environment Configuration

Additional parameters for `BitgetFuturesSLTPTradingEnvConfig`:

```python
config = BitgetFuturesSLTPTradingEnvConfig(
    # ... all standard parameters above, plus:

    # Stop loss levels as percentages (negative values)
    stoploss_levels=(-0.025, -0.05, -0.1),  # -2.5%, -5%, -10%

    # Take profit levels as percentages (positive values)
    takeprofit_levels=(0.05, 0.1, 0.2),     # 5%, 10%, 20%

    # Action space options
    include_short_positions=True,   # Include short actions
    include_hold_action=True,       # Include HOLD action (index 0)
)
```

With 3 SL levels, 3 TP levels, and both long/short enabled:
- 1 HOLD action
- 9 LONG actions (3 SL × 3 TP)
- 9 SHORT actions (3 SL × 3 TP)
- **Total: 19 actions**

## Usage Examples

### Basic Example with Random Actions

```python
import os
from dotenv import load_dotenv
from torchtrade.envs.bitget import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig,
)
from torchtrade.envs.bitget.futures_order_executor import (
    MarginMode,
    PositionMode,
)

load_dotenv()

# Create environment configuration
config = BitgetFuturesTradingEnvConfig(
    symbol="BTC/USDT:USDT",
    demo=True,
    time_frames=["5min", "15min"],
    window_sizes=[6, 32],
    execute_on="1min",
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,
    position_mode=PositionMode.ONE_WAY,
)

# Create environment
env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)

# Reset environment
tensordict = env.reset()
print(f"Observation keys: {list(tensordict.keys())}")
print(f"Account state: {tensordict['account_state']}")

# Random action step
action = env.action_spec.rand()
next_tensordict = env.step(tensordict.set("action", action))
print(f"Reward: {next_tensordict['reward'].item()}")
```

### Custom Feature Preprocessing

```python
import pandas as pd

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom technical indicators."""
    df = df.copy().reset_index(drop=True)

    # Basic features
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Custom indicators
    df["features_rsi"] = compute_rsi(df["close"], 14)
    df["features_macd"] = compute_macd(df["close"])

    # Drop NaN rows from indicators
    df.dropna(inplace=True)

    return df

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
    feature_preprocessing_fn=custom_preprocessing,
)
```

### Data Collection with TorchRL

See `run_live.py` for a complete example of collecting live trading data into a replay buffer:

```python
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

# Create data collector
collector = SyncDataCollector(
    env,
    policy=None,  # None = random actions
    frames_per_batch=1,
    total_frames=10000,
    device="cpu",
)

# Create replay buffer
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(10_000),
    batch_size=1,
)

# Collect data
for tensordict in collector:
    replay_buffer.extend(tensordict.reshape(-1))

# Save buffer
replay_buffer.dumps("./replay_buffer.pt")
```

### Using Trained Policies

```python
from torchrl.modules import ActorValueOperator

# Load your trained policy
policy = ActorValueOperator(
    actor=your_actor_network,
    value=your_value_network,
).eval()

# Use with collector
collector = SyncDataCollector(
    env,
    policy=policy,
    frames_per_batch=1,
    total_frames=10000,
)

for tensordict in collector:
    # Policy actions are used instead of random
    pass
```

### SLTP Environment Example

```python
from torchtrade.envs.bitget import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig,
)

config = BitgetFuturesSLTPTradingEnvConfig(
    symbol="BTC/USDT:USDT",
    demo=True,
    time_frames=["5min"],
    window_sizes=[32],
    execute_on="1min",
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,
    position_mode=PositionMode.ONE_WAY,
    # SL/TP configuration
    stoploss_levels=(-0.02, -0.05),      # -2%, -5%
    takeprofit_levels=(0.03, 0.06, 0.10),  # 3%, 6%, 10%
    include_short_positions=True,
)

env = BitgetFuturesSLTPTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)

print(f"Action space size: {env.action_spec.n}")
print(f"Action map (first 5):")
for idx, action in list(env.action_map.items())[:5]:
    print(f"  Action {idx}: {action}")
```

## Environment Transforms

Apply standard TorchRL transforms for better training:

```python
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.transforms import (
    InitTracker,
    StepCounter,
    DoubleToFloat,
    RewardSum,
)

def apply_env_transforms(env, max_episode_steps=1000):
    """Apply standard transforms to the environment."""
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),                    # Track episode initialization
            StepCounter(max_episode_steps),   # Limit episode length
            DoubleToFloat(),                  # Convert double to float tensors
            RewardSum(),                      # Track cumulative reward
        ),
    )
    return transformed_env

env = BitgetFuturesTorchTradingEnv(config, ...)
env = apply_env_transforms(env, max_rollout_steps=72)
```

## Demo vs Production Trading

### Demo Trading (Testnet)
```python
config = BitgetFuturesTradingEnvConfig(
    demo=True,  # Use Bitget testnet
    symbol="BTC/USDT:USDT",
)
```

- Uses Bitget's demo trading environment
- Fake money (typically 1000 USDT demo balance)
- Same API as production
- Perfect for testing strategies
- Create demo account at: https://www.bitget.com/demo-trading

### Production Trading
```python
config = BitgetFuturesTradingEnvConfig(
    demo=False,  # Use real Bitget account
    symbol="BTC/USDT:USDT",
)
```

**⚠️ Warning**: Production trading uses real money. Start with small position sizes and thoroughly test strategies in demo mode first.

## Important Notes

### API Migration (V1 → V2)
- TorchTrade uses **CCXT library** with Bitget V2 API
- Old V1 API was decommissioned on November 28, 2025
- Symbol format changed from "BTCUSDT" to "BTC/USDT:USDT"
- Product types changed from "SUMCBL" to "USDT-FUTURES"

### Position Management
- Environment does NOT automatically close positions on cleanup
- Use `close_position_on_reset=True` if needed
- Manual cleanup: `env.trader.close_position()` before `env.close()`
- This prevents accidental liquidation of intended positions

### Leverage and Risk
- Higher leverage = higher risk of liquidation
- Start with low leverage (2-5x) when testing
- ISOLATED margin mode recommended for risk management
- Monitor margin ratio to avoid liquidation

### Time Synchronization
- Bitget uses UTC timezone
- Environment handles timestamp alignment automatically
- `execute_on` parameter determines trade execution frequency

### Order Execution
- Market orders are used by default
- Orders execute at next `execute_on` timestamp
- SLTP environment uses bracket orders (entry + SL/TP)
- **Position Mode & tradeSide parameter**:
  - ONE_WAY mode: `tradeSide` parameter is omitted (as per Bitget API spec)
  - HEDGE mode: `tradeSide='open'` for new positions, `'close'` for exits

## Troubleshooting

### "BadSymbol" Error
Make sure you're using CCXT perpetual swap format:
```python
symbol = "BTC/USDT:USDT"  # Correct
# NOT "BTCUSDT" or "BTC-USDT"
```

### "Margin Coin cannot be empty" Error
This is handled automatically by the environment. If you see this, ensure your config has valid `symbol` and `product_type`.

### Margin Mode Shows "Cross" on Bitget Interface
Bitget's `set_margin_mode()` API doesn't work reliably. The environment now sets margin mode per-order by including `marginMode` in the order parameters. This is the recommended approach per [CCXT issue #21435](https://github.com/ccxt/ccxt/issues/21435). Each order will use the configured margin mode (ISOLATED or CROSSED).

### Both Long and Short Positions Opening
You're in HEDGE mode but want ONE_WAY mode:
```python
config = BitgetFuturesTradingEnvConfig(
    position_mode=PositionMode.ONE_WAY,  # Set this explicitly
)
```

### "No position to close" Message
This is normal behavior when the agent tries to close a non-existent position. The environment handles it gracefully and continues.

### Rate Limiting
Bitget has API rate limits. If you hit them:
- Increase `execute_on` timeframe (e.g., "5min" instead of "1min")
- Reduce `frames_per_batch` in collector
- Add delays between operations if needed

## Additional Resources

- [Bitget API Documentation](https://bitgetlimited.github.io/apidoc/en/mix/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Bitget Demo Trading](https://www.bitget.com/demo-trading)

## License

This example is part of TorchTrade and follows the project's license.
