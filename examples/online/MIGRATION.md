# Migration Guide: Algorithm Consolidation

## What Changed?

We've consolidated duplicate algorithm directories using Hydra's defaults list pattern. The examples/online directory has been restructured from 10 algorithm-specific directories to 5 algorithm directories + 1 shared environment config folder.

### Directory Changes

| Old Directory | New Command |
|---------------|-------------|
| `ppo/` | `python ppo/train.py env=sequential_sltp` |
| `ppo_futures/` | `python ppo/train.py env=sequential_futures` |
| `ppo_futures_sltp/` | `python ppo/train.py env=sequential_futures_sltp` |
| `ppo_futures_onestep/` | `python ppo/train.py env=onestep_futures` |
| `ppo_chronos/` | `python ppo_chronos/train.py` (unchanged) |
| `dsac/` | `python dsac/train.py` (unchanged) |
| `iql/` | `python iql/train.py` (unchanged) |
| `grpo_futures_onestep/` | `python grpo/train.py env=onestep_futures` |
| `long_onestep_env/` | `python grpo/train.py env=onestep_spot` |
| `rulebased/` | Deleted (not an RL algorithm) |

### New Structure

```
examples/online/
├── env/                              # Central environment configs
│   ├── sequential.yaml               # Basic sequential trading
│   ├── sequential_sltp.yaml          # Sequential with stop-loss/take-profit
│   └── onestep.yaml                  # One-step for contextual bandits
│
├── ppo/                              # PPO algorithm
│   ├── config.yaml                   # Algorithm-specific config
│   ├── env/ → ../env/                # Symlink for CLI env switching
│   ├── train.py
│   └── utils.py
├── grpo/                             # GRPO algorithm (onestep-only)
│   ├── config.yaml                   # Env embedded, no symlink
│   ├── train.py
│   └── utils.py
└── ...  (dsac, iql, ppo_chronos have env symlinks)
```

**Key Points:**
- **Algorithm configs** are real files in each algorithm directory
- **Environment configs** are centralized in `env/` directory (except GRPO)
- **One env symlink** per algorithm enables CLI environment switching
- **GRPO is special** - environment embedded directly, no CLI switching (onestep-only)
- **No spot/futures split** - users override `leverage` and `action_levels` for futures
- **All use 1Hour timeframe** by default

## Benefits

1. **Cleaner structure**: 5 algorithm directories instead of 10
2. **Easy environment switching**: Change env via command-line
3. **Less code duplication**: Shared environment configs (6 YAML files)
4. **Simpler maintenance**: Update env configs once, apply to all algorithms
5. **Flexible experimentation**: Test any algorithm with any environment

## How to Update Your Scripts

If you have custom training scripts or references to the old directories:

### Step 1: Add Defaults List to Config

```yaml
# OLD: config.yaml
env:
  name: SequentialTradingEnv
  symbol: "BTC/USD"
  time_frames: ["5Min", "15Min"]
  # ... more env parameters

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000
```

```yaml
# NEW: config.yaml
defaults:
  - env: sequential  # Choose: sequential, sequential_sltp, or onestep
  - _self_

# Optional: algorithm-specific env overrides
env:
  leverage: 5  # Override for futures
  action_levels: [-1.0, 0.0, 1.0]  # Override for futures

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000
```

### Step 2: Create or Use Environment Config

Use one of the standard configs or create a custom one:

```yaml
# examples/online/env/my_custom_env.yaml
# @package env
name: SequentialTradingEnv
symbol: "BTC/USD"
time_frames: ["1Hour"]
window_sizes: [24]
execute_on: "1Hour"
leverage: 1
action_levels: [0.0, 1.0]
exp_name: ${logger.exp_name}
seed: 0
train_envs: 5
eval_envs: 1
data_path: Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025
test_split_start: "2025-01-01"
```

**For futures trading:** Override `leverage` and `action_levels` via CLI or in algorithm config.

### Step 3: Update Config Path and Create Symlink

```python
# OLD:
@hydra.main(config_path="..", config_name="algorithm_name/config", version_base="1.1")

# NEW:
@hydra.main(config_path=".", config_name="config", version_base="1.1")
```

Then create a symlink to enable CLI environment switching:

```bash
cd examples/online/your_algorithm/
ln -sf ../env env  # Symlink to central env configs for CLI switching
```

**Why the env symlink?** It enables CLI environment switching (`env=X`) while keeping environment configs centralized.

## Available Environment Configs

| Config Name | Environment Class | SLTP | Timeframe | Use Cases |
|-------------|-------------------|------|-----------|-----------|
| `sequential` | SequentialTradingEnv | No | 1Hour | Basic sequential trading |
| `sequential_sltp` | SequentialTradingEnvSLTP | Yes | 1Hour | Sequential with bracket orders |
| `onestep` | OneStepTradingEnv | Yes | 1Hour | One-step for GRPO/contextual bandits |

**Configuring Spot vs Futures:**
- **Spot trading**: `leverage: 1` + `action_levels: [0.0, 1.0]` (default)
- **Futures trading**: Override with `env.leverage=5` + `env.action_levels='[-1.0,0.0,1.0]'`

## Usage Examples

### PPO Variants

```bash
# Default: sequential trading (spot, 1Hour)
python examples/online/ppo/train.py

# Sequential with bracket orders
python examples/online/ppo/train.py env=sequential_sltp

# Configure for futures trading
python examples/online/ppo/train.py \
    env.leverage=5 \
    env.action_levels='[-1.0,0.0,1.0]'

# Override environment and parameters
python examples/online/ppo/train.py \
    env=sequential_sltp \
    env.symbol="ETH/USD" \
    env.leverage=10 \
    optim.lr=1e-4
```

### GRPO

GRPO is specifically designed for one-step environments and has the environment config embedded directly:

```bash
# Default: one-step trading (spot, 1Hour)
python examples/online/grpo/train.py

# Configure for futures trading
python examples/online/grpo/train.py \
    env.leverage=5 \
    env.quantity_per_trade=200

# Override other parameters
python examples/online/grpo/train.py \
    env.stoploss_levels='[-0.05]' \
    env.takeprofit_levels='[0.10]'
```

**Note:** GRPO does not support environment switching - it only works with OneStepTradingEnv.

### Other Algorithms

```bash
# dSAC (default: sequential, 1Hour)
python examples/online/dsac/train.py

# IQL (default: sequential, 1Hour)
python examples/online/iql/train.py

# IQL with SLTP
python examples/online/iql/train.py env=sequential_sltp

# PPO-Chronos (default: sequential, 1Hour)
python examples/online/ppo_chronos/train.py

# Configure any algorithm for futures
python examples/online/dsac/train.py \
    env.leverage=5 \
    env.action_levels='[-1.0,0.0,1.0]'
```

## Breaking Changes

### Removed Directories

The following directories have been removed:

- ❌ `examples/online/ppo_futures/`
- ❌ `examples/online/ppo_futures_sltp/`
- ❌ `examples/online/ppo_futures_onestep/`
- ❌ `examples/online/grpo_futures_onestep/`
- ❌ `examples/online/long_onestep_env/`
- ❌ `examples/online/rulebased/` (not an RL algorithm)

### Config Path Changes

All `train.py` files now use `config_path="."` pointing to the local `config.yaml`:

- **Algorithm configs** are real files in each algorithm directory (no duplication)
- **Environment configs** are centralized in `env/` directory (single source of truth)
- **One symlink** (`env/`) per algorithm enables CLI environment switching

This approach balances config organization with CLI usability.

### Environment Config Location

Environment configs are centralized in `examples/online/env/`:

1. **Base config** comes from `env/<envname>.yaml`
2. **Algorithm-specific overrides** can be added in `<algorithm>/config.yaml`

```yaml
# Algorithm config can override env parameters
defaults:
  - env: sequential
  - _self_

env:
  leverage: 5                    # Override for futures
  action_levels: [-1.0, 0.0, 1.0]  # Override for futures
  train_envs: 10                 # Algorithm-specific override
```

## Migrating Custom Scripts

If you have custom training scripts based on the old examples:

1. **Keep your custom logic** in `train.py` and `utils.py`
2. **Update your config** to use defaults list pattern
3. **Extract env config** to a separate YAML file in `env/` (optional but recommended)
4. **Update config_path** in the `@hydra.main` decorator

Example migration:

```diff
# config.yaml
+ defaults:
+   - env: sequential_spot
+   - _self_
+
- env:
-   name: SequentialTradingEnv
-   symbol: "BTC/USD"
-   # ... 20 lines of env config

+ # Optional: algorithm-specific overrides
+ env:
+   time_frames: ["1Hour"]  # Custom timeframe

  collector:
    frames_per_batch: 100000
```

```diff
# train.py
- @hydra.main(config_path="", config_name="config", version_base="1.1")
+ @hydra.main(config_path=".", config_name="config", version_base="1.1")
  def main(cfg):
      # ... rest of code unchanged
```

```bash
# Create symlinks in your algorithm directory
cd examples/online/your_algorithm/
ln -sf ../algorithm/your_algorithm.yaml config.yaml
ln -sf ../env env
```

## Troubleshooting

### Error: "Could not find 'env/sequential_spot'"

**Cause**: Hydra can't find the env config folder.

**Solution**: Ensure you have the `env` symlink in your algorithm directory:
```bash
cd examples/online/your_algorithm/
ln -sf ../env env
```

### Error: "Missing key 'env.name'"

**Cause**: Environment config not loaded properly.

**Solution**: Check that your `defaults` list includes an env config:
```yaml
defaults:
  - env: sequential_spot
  - _self_
```

### Want to use old directory structure?

If you prefer the old structure for your custom experiments, you can:

1. Copy the old directory structure from git history
2. Or use the new structure with environment switching

The new structure is recommended for cleaner organization and easier maintenance.

## Questions?

For issues or questions about the migration:

1. Check the [Examples Documentation](https://torchtrade.github.io/torchtrade_envs/examples/)
2. Review example configs in `examples/online/env/`
3. Open an issue on [GitHub](https://github.com/TorchTrade/torchtrade/issues)
