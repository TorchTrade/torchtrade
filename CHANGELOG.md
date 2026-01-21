# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Fractional position sizing** for `SeqFuturesEnv` and `SeqLongOnlyEnv`
  - Action value now represents fraction of balance to allocate to positions
  - Default action levels: `[-1.0, -0.5, 0.0, 0.5, 1.0]` for fine-grained control
  - More intuitive semantics: `0.0` = market neutral, `±1.0` = all-in
  - Self-normalizing: automatically scales with account size
  - Customizable action levels for different granularity levels
  - **Efficient partial adjustments**: Only trades the delta when adjusting position size
    - Example: Going from 100% to 50% long only closes 50% of the position
    - Maintains weighted average entry price when adding to positions
    - Preserves entry price when reducing positions
- `position_sizing_mode` parameter to control fractional (default) vs fixed (legacy) sizing
- Comprehensive test suite for fractional position sizing in `tests/envs/test_fractional_actions.py` (18 tests)

### Changed
- **BREAKING**: Default action semantics changed for non-SLTP environments
  - `0.0` now means "market neutral" (close all positions) instead of "hold"
  - Position sizing now based on action value (fraction of balance) instead of `quantity_per_trade`
  - HOLD action is now implicit (duplicate action prevention)
  - Action space configuration moved from boolean flags to explicit `action_levels` list
- Default `position_sizing_mode` is now `"fractional"` for new environments

### Deprecated
- `include_hold_action` parameter - only used in legacy mode (`position_sizing_mode="fixed"`)
- `include_close_action` parameter - only used in legacy mode (SeqFuturesEnv only)
- `quantity_per_trade` parameter - only used in legacy mode

### Migration Guide

#### For Existing Trained Models

To maintain backward compatibility with existing trained models, set `position_sizing_mode="fixed"`:

```python
from torchtrade.envs.offline.seqfutures import SeqFuturesEnv, SeqFuturesEnvConfig

config = SeqFuturesEnvConfig(
    position_sizing_mode="fixed",  # Use legacy mode
    quantity_per_trade=0.001,
    include_hold_action=True,
    include_close_action=True,
    # ... other parameters ...
)
env = SeqFuturesEnv(df, config)
```

#### For New Training

Use the new fractional mode (default) for better action semantics:

```python
from torchtrade.envs.offline.seqfutures import SeqFuturesEnv, SeqFuturesEnvConfig

config = SeqFuturesEnvConfig(
    # position_sizing_mode="fractional" is the default
    action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],  # Optional, this is default
    leverage=5,
    initial_cash=10000,
    # ... other parameters ...
)
env = SeqFuturesEnv(df, config)
```

#### Custom Action Levels

You can customize action granularity to suit your needs:

```python
# Fine-grained control (9 actions)
config = SeqFuturesEnvConfig(
    action_levels=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
)

# Conservative (no full positions, 5 actions)
config = SeqFuturesEnvConfig(
    action_levels=[-0.5, -0.25, 0.0, 0.25, 0.5]
)

# Asymmetric (prefer long positions, 7 actions)
config = SeqFuturesEnvConfig(
    action_levels=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]
)
```

### Benefits of Fractional Mode

1. **Intuitive Semantics**: `0 = neutral`, action value has clear meaning (fraction)
2. **Self-Normalizing**: Automatically adjusts to account size changes
3. **Fine-Grained Control**: Agents can learn partial position sizing
4. **Better Risk Management**: Position sizes scale proportionally with available capital
5. **Consistency**: Same pattern for futures and long-only environments

### Technical Details

**Position Calculation Formula**:
```
position_size = (balance × |action| × leverage) / price
```

Where:
- `balance`: Current available cash
- `|action|`: Absolute value of action (fraction from 0.0 to 1.0)
- `leverage`: Configured leverage multiplier (1 for long-only)
- `price`: Current market price

**Direction**: Determined by the sign of the action
- `action > 0`: Long position
- `action < 0`: Short position
- `action = 0`: Flat (no position)

## [Previous Versions]

See git history for changes in previous versions.
