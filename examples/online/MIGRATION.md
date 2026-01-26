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
├── env/                              # Shared environment configs
│   ├── sequential_spot.yaml
│   ├── sequential_futures.yaml
│   ├── sequential_sltp.yaml
│   ├── sequential_futures_sltp.yaml
│   ├── onestep_spot.yaml
│   └── onestep_futures.yaml
│
├── ppo/                              # Consolidated PPO (4 → 1)
├── ppo_chronos/                      # Chronos model variant
├── dsac/                             # Off-policy algorithm
├── iql/                              # Offline RL algorithm
└── grpo/                             # Consolidated GRPO (2 → 1)
```

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
  - env: sequential_spot  # or your preferred environment
  - _self_

# Optional: algorithm-specific env overrides
env:
  time_frames: ["1Hour"]  # Override if needed

collector:
  frames_per_batch: 100000
  total_frames: 100_000_000
```

### Step 2: Create or Use Environment Config

Either use an existing config from `examples/online/env/` or create a custom one:

```yaml
# examples/online/env/my_custom_env.yaml
# @package env
name: SequentialTradingEnv
trading_mode: "spot"
symbol: "BTC/USD"
time_frames: ["5Min", "15Min"]
window_sizes: [10, 10]
execute_on: "15Min"
initial_cash: [1000, 5000]
transaction_fee: 0.0025
slippage: 0.001
bankrupt_threshold: 0.1
exp_name: ${logger.exp_name}
seed: 0
train_envs: 5
eval_envs: 1
data_path: Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025
test_split_start: "2025-01-01"
```

### Step 3: Update Config Path in train.py and Create Symlink

```python
# OLD:
@hydra.main(config_path="..", config_name="algorithm_name/config", version_base="1.1")

# NEW:
@hydra.main(config_path=".", config_name="config", version_base="1.1")
```

Then create a symlink to the shared env directory:

```bash
cd examples/online/your_algorithm/
ln -sf ../env env
```

This allows Hydra to find environment configs when switching with `env=envname`.

## Available Environment Configs

| Config Name | Environment Class | Trading Mode | SLTP | Leverage | Use Cases |
|-------------|-------------------|--------------|------|----------|-----------|
| `sequential_spot` | SequentialTradingEnv | spot | No | 1x | Basic spot trading |
| `sequential_futures` | SequentialTradingEnv | futures | No | 5-6x | Basic futures trading |
| `sequential_sltp` | SequentialTradingEnvSLTP | spot | Yes | 1x | Spot with bracket orders |
| `sequential_futures_sltp` | SequentialTradingEnvSLTP | futures | Yes | 5x | Futures with bracket orders |
| `onestep_spot` | OneStepTradingEnv | spot | Yes | 1x | Contextual bandit (spot) |
| `onestep_futures` | OneStepTradingEnv | futures | Yes | 5x | Contextual bandit (futures) |

## Usage Examples

### PPO Variants

```bash
# Spot trading with bracket orders (default)
python examples/online/ppo/train.py

# Basic futures trading (no SLTP)
python examples/online/ppo/train.py env=sequential_futures

# Futures with bracket orders
python examples/online/ppo/train.py env=sequential_futures_sltp

# One-step futures (contextual bandit)
python examples/online/ppo/train.py env=onestep_futures

# Override specific parameters
python examples/online/ppo/train.py \
    env=sequential_futures \
    env.symbol="ETH/USD" \
    env.leverage=10 \
    optim.lr=1e-4
```

### GRPO Variants

```bash
# Futures one-step (default)
python examples/online/grpo/train.py

# Spot one-step
python examples/online/grpo/train.py env=onestep_spot

# Sequential futures (non-onestep)
python examples/online/grpo/train.py env=sequential_futures

# Custom configuration
python examples/online/grpo/train.py \
    env=onestep_spot \
    env.quantity_per_trade=200
```

### Other Algorithms

```bash
# dSAC (default: sequential spot)
python examples/online/dsac/train.py

# dSAC with futures
python examples/online/dsac/train.py env=sequential_futures

# IQL (default: sequential spot)
python examples/online/iql/train.py

# IQL with SLTP
python examples/online/iql/train.py env=sequential_sltp

# PPO-Chronos (default: sequential futures SLTP)
python examples/online/ppo_chronos/train.py

# PPO-Chronos with spot
python examples/online/ppo_chronos/train.py env=sequential_spot
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

All `train.py` files now use `config_path="."` instead of `config_path=""`. This enables Hydra to find the shared `env/` configs.

### Environment Config Location

Environment parameters are no longer defined directly in algorithm configs. Instead:

1. **Base config** comes from `examples/online/env/<envname>.yaml`
2. **Algorithm-specific overrides** can be added in the algorithm's `config.yaml`

```yaml
# Algorithm config can override env parameters
defaults:
  - env: sequential_spot
  - _self_

env:
  time_frames: ["1Hour"]  # Override from sequential_spot.yaml
  transaction_fee: 0.0    # Override from sequential_spot.yaml
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

## Troubleshooting

### Error: "Could not find 'env/sequential_spot'"

**Cause**: Hydra can't find the env config folder.

**Solution**: Ensure `config_path="."` in your `@hydra.main` decorator.

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
