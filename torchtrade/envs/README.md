# TorchTrade Environments

This directory contains the complete environment infrastructure for TorchTrade, organized into a clear hierarchical structure.

## Directory Structure

```
torchtrade/envs/
├── core/              # Base classes and fundamental abstractions
├── utils/             # Shared utility functions and helpers
├── offline/           # Backtesting environments
│   ├── infrastructure/    # Internal sampler and utilities
│   ├── longonly/          # Long-only trading environments
│   └── futures/           # Futures trading environments
├── live/              # Live trading environments
│   ├── shared/            # Shared components across providers
│   ├── alpaca/            # Alpaca integration
│   ├── binance/           # Binance Futures integration
│   └── bitget/            # Bitget Futures integration
└── transforms/        # TorchRL environment transforms
```

## Quick Start

### Offline (Backtesting)

```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

# Create a long-only sequential environment
config = SeqLongOnlyEnvConfig(
    timeframe=TimeFrame(1, TimeFrameUnit.DAY),
    window_size=50,
)
env = SeqLongOnlyEnv(df=your_dataframe, config=config)
```

### Live Trading

```python
from torchtrade.envs import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Create an Alpaca live trading environment
config = AlpacaTradingEnvConfig(
    api_key="your_key",
    api_secret="your_secret",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
)
env = AlpacaTorchTradingEnv(config=config)
```

## Architecture Overview

### Core (`core/`)

Contains the fundamental base classes that all environments inherit from:

- **TorchTradeBaseEnv**: Root base class for all environments
- **TorchTradeOfflineEnv**: Base for backtesting environments
- **TorchTradeLiveEnv**: Base for live trading environments
- **PositionState**: State management for positions
- **RewardFunction**: Reward calculation abstractions

See [core/README.md](core/README.md) for details.

### Utils (`utils/`)

Shared utilities used across all environment types:

- **Timeframe**: Time period management and conversions
- **Action Maps**: Discrete action space definitions
- **SL/TP Helpers**: Stop-loss and take-profit calculations
- **Position Sizing**: Fractional position sizing utilities
- **Metrics**: Performance metric calculations

See [utils/README.md](utils/README.md) for details.

### Offline Environments (`offline/`)

Backtesting environments for strategy development and evaluation:

- **Long-Only**: Traditional buy-and-hold style environments
- **Futures**: Leveraged long/short futures environments
- **Infrastructure**: Internal sampling and utilities (not user-facing)

See [offline/README.md](offline/README.md) for details.

### Live Environments (`live/`)

Production-ready environments for live trading:

- **Alpaca**: US equities and crypto spot trading
- **Binance**: Crypto futures trading
- **Bitget**: Crypto futures trading
- **Shared**: Common components (futures base observation)

See [live/README.md](live/README.md) for details.

## Import Guidelines

### Recommended Imports

Import from the top level when possible:

```python
# Good - top-level imports
from torchtrade.envs import (
    SeqLongOnlyEnv,
    SeqFuturesEnv,
    AlpacaTorchTradingEnv,
    TimeFrame,
    TimeFrameUnit,
)
```

### Direct Imports

Import directly from submodules when needed:

```python
# Also good - direct imports
from torchtrade.envs.offline.longonly import SeqLongOnlyEnv
from torchtrade.envs.live.binance import BinanceFuturesTorchTradingEnv
from torchtrade.envs.core import TorchTradeBaseEnv
from torchtrade.envs.utils import calculate_metrics
```

## Environment Types

### Sequential Environments

Multi-step environments where agents make decisions at each timestep:

- `SeqLongOnlyEnv`: Long-only sequential trading
- `SeqLongOnlySLTPEnv`: Long-only with stop-loss/take-profit
- `SeqFuturesEnv`: Futures sequential trading
- `SeqFuturesSLTPEnv`: Futures with stop-loss/take-profit

### One-Step Environments

Simplified environments for single-step decision making:

- `LongOnlyOneStepEnv`: Single-step long-only
- `FuturesOneStepEnv`: Single-step futures

### Live Environments

Real-time trading environments with API integration:

- `AlpacaTorchTradingEnv`: Alpaca standard environment
- `AlpacaSLTPTorchTradingEnv`: Alpaca with SL/TP
- `BinanceFuturesTorchTradingEnv`: Binance Futures
- `BinanceFuturesSLTPTorchTradingEnv`: Binance with SL/TP
- `BitgetFuturesTorchTradingEnv`: Bitget Futures
- `BitgetFuturesSLTPTorchTradingEnv`: Bitget with SL/TP

## Design Principles

1. **Clear Separation**: Core classes, utilities, and implementations are clearly separated
2. **Consistent Naming**: All provider files follow the same naming conventions
3. **Parallel Structure**: Offline longonly and futures mirror each other
4. **Import Flexibility**: Support both top-level and direct imports
5. **Maintainability**: READMEs at each level explain purpose and usage

## Migration from Old Structure

If you have code using the old import paths, here's how to migrate:

| Old Import | New Import |
|------------|------------|
| `from torchtrade.envs.base import` | `from torchtrade.envs.core.base import` |
| `from torchtrade.envs.timeframe import` | `from torchtrade.envs.utils.timeframe import` |
| `from torchtrade.envs.offline.seqlongonly import` | `from torchtrade.envs.offline.longonly.sequential import` |
| `from torchtrade.envs.alpaca.torch_env import` | `from torchtrade.envs.live.alpaca.env import` |
| `from torchtrade.envs.binance.obs_class import` | `from torchtrade.envs.live.binance.observation import` |

Or simply use the top-level imports which work with both old and new structure:

```python
from torchtrade.envs import SeqLongOnlyEnv, AlpacaTorchTradingEnv
```

## Contributing

When adding new environments or utilities:

1. Follow the established naming conventions
2. Place code in the appropriate subdirectory
3. Update relevant `__init__.py` files for exports
4. Add documentation to the appropriate README
5. Ensure imports work from both top-level and direct paths

## See Also

- [Core Base Classes](core/README.md)
- [Utilities Documentation](utils/README.md)
- [Offline Environments Guide](offline/README.md)
- [Live Environments Guide](live/README.md)
