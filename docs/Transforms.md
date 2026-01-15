# TorchRL Transforms

TorchTrade provides TorchRL-compatible transforms that can be applied to any environment to enhance observations with additional features.

## Overview

Transforms follow the TorchRL `Transform` API and can be composed using `Compose` and applied via `TransformedEnv`:

```python
from torchrl.envs import TransformedEnv, Compose
from torchtrade.envs.transforms import MarketRegimeTransform

env = TransformedEnv(
    base_env,
    Compose(
        MarketRegimeTransform(in_keys=["market_data_1Minute_12"]),
        # Add more transforms here
    )
)
```

## Available Transforms

### MarketRegimeTransform

**Purpose**: Add market regime features for context-aware trading strategies.

The `MarketRegimeTransform` computes 7 regime indicators from price and volume data:

1. **vol_regime** (categorical): Volatility regime (0=low, 1=medium, 2=high)
2. **trend_regime** (categorical): Trend regime (-1=downtrend, 0=sideways, 1=uptrend)
3. **volume_regime** (categorical): Volume regime (0=low, 1=normal, 2=high)
4. **position_regime** (categorical): Price position (0=oversold, 1=neutral, 2=overbought)
5. **volatility** (continuous): Current volatility value
6. **trend_strength** (continuous): Trend strength indicator
7. **volume_ratio** (continuous): Volume ratio vs average

These features enable agents to learn **regime-conditional policies** - different strategies for different market conditions.

#### Basic Usage

```python
from torchrl.envs import TransformedEnv
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.transforms import MarketRegimeTransform

# Create base environment
config = SeqLongOnlyEnvConfig(
    time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
    window_sizes=[12],
    execute_on=TimeFrame(1, TimeFrameUnit.Minute),
    initial_cash=10000,
)
base_env = SeqLongOnlyEnv(df, config)

# Add regime features
env = TransformedEnv(
    base_env,
    MarketRegimeTransform(
        in_keys=["market_data_1Minute_12"],  # Input market data key
        price_feature_idx=3,                  # Index of close price in OHLCV
        volume_feature_idx=4,                 # Index of volume in OHLCV
    )
)

# Reset and inspect
td = env.reset()
print("Regime features:", td["regime_features"])
# Output: tensor([1.0, 0.0, 1.0, 1.0, 0.023, 0.001, 1.05])
#                 [vol, trend, volume, position, vol_cont, trend_cont, vol_ratio]
```

#### Configuration Parameters

```python
MarketRegimeTransform(
    in_keys: List[str],                    # Market data keys to process
    price_feature_idx: int = 3,            # Index of price feature (default: close)
    volume_feature_idx: int = 4,           # Index of volume feature
    volatility_window: int = 20,           # Lookback for volatility calculation
    trend_window: int = 50,                # Lookback for trend calculation
    volume_window: int = 20,               # Lookback for volume calculation
    position_window: int = 252,            # Lookback for price position (52 weeks)
    vol_percentiles: List[float] = [0.33, 0.67],      # Volatility thresholds
    trend_thresholds: List[float] = [-0.02, 0.02],    # Trend thresholds
    volume_thresholds: List[float] = [0.7, 1.3],      # Volume thresholds
    position_percentiles: List[float] = [0.33, 0.67], # Position thresholds
)
```

#### Custom Configuration Example

```python
# More responsive regime detection
env = TransformedEnv(
    base_env,
    MarketRegimeTransform(
        in_keys=["market_data_1Minute_12"],
        volatility_window=10,        # Shorter = more responsive
        trend_window=30,             # Shorter = more responsive
        trend_thresholds=[-0.01, 0.01],  # Tighter = more sensitive
        volume_thresholds=[0.5, 1.5],    # Wider = clearer extremes
    )
)
```

#### Regime-Conditional Policy Example

```python
# Simple rule-based regime-aware strategy
def regime_conditional_policy(observation):
    regime = observation["regime_features"]

    vol_regime = regime[0].item()      # 0=low, 1=med, 2=high
    trend_regime = regime[1].item()    # -1=down, 0=sideways, 1=up

    # Context-dependent logic
    if trend_regime == 1.0 and vol_regime <= 1.0:
        # Uptrend + low/medium volatility → aggressive long
        return 2  # BUY
    elif trend_regime == -1.0:
        # Downtrend → exit
        return 0  # SELL
    elif vol_regime == 2.0:
        # High volatility → conservative
        return 1  # HOLD
    else:
        return 1  # HOLD (default)
```

In RL training, neural networks learn much more sophisticated regime-action mappings than simple rules.

#### Multi-Timeframe Usage

```python
# Compute regime features from primary execution timeframe
config = SeqLongOnlyEnvConfig(
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 12, 12],
    execute_on=TimeFrame(1, TimeFrameUnit.Minute),
    initial_cash=10000,
)
base_env = SeqLongOnlyEnv(df, config)

# Use execution timeframe for regime detection
env = TransformedEnv(
    base_env,
    MarketRegimeTransform(
        in_keys=["market_data_1Minute_12"],  # Use execution timeframe
    )
)
```

**Note**: You can create multiple regime transforms for different timeframes if you want multi-scale regime awareness.

#### Integration with Training

```python
from torchrl.envs import Compose, InitTracker, RewardSum

# Full training pipeline with regime features
env = TransformedEnv(
    base_env,
    Compose(
        MarketRegimeTransform(
            in_keys=["market_data_1Minute_12"],
        ),
        InitTracker(),  # Track episode initialization
        RewardSum(),    # Track cumulative reward
    )
)

# Train with any RL algorithm (PPO, IQL, GRPO, etc.)
# The policy network automatically receives regime_features in observations
```

#### Expected Behavior

With regime features, agents should learn:

1. **Volatility adaptation**:
   - Low volatility → use aggressive actions (wide stops, large targets)
   - High volatility → use conservative actions (tight stops, quick profits)

2. **Trend adaptation**:
   - Strong uptrend → favor long positions
   - Strong downtrend → favor short positions (or exit for long-only)
   - Sideways → mean reversion or selective trading

3. **Volume adaptation**:
   - High volume → confident entries (strong signal)
   - Low volume → conservative or wait (weak signal)

4. **Position adaptation**:
   - Oversold → look for long opportunities
   - Overbought → take profits or avoid longs
   - Neutral → normal trading

#### Advantages

**vs No Regime Features:**
- ✅ Context-aware decisions
- ✅ Better generalization to new market conditions
- ✅ Reduced overfitting to training period characteristics
- ✅ More interpretable (can analyze regime-action associations)

**vs Manual Feature Engineering:**
- ✅ Standardized implementation across all environments
- ✅ Composable with other transforms
- ✅ Automatically updates observation specs
- ✅ Handles batched observations correctly

#### Example Output

```python
td = env.reset()
print("Regime features:", td["regime_features"])

# Example output:
# tensor([2.0, 1.0, 0.0, 1.0, 0.045, 0.025, 0.65])
#
# Interpretation:
# [0] vol_regime = 2.0        → High volatility
# [1] trend_regime = 1.0      → Uptrend
# [2] volume_regime = 0.0     → Low volume
# [3] position_regime = 1.0   → Neutral price position
# [4] volatility = 0.045      → 4.5% volatility
# [5] trend_strength = 0.025  → 2.5% trend strength
# [6] volume_ratio = 0.65     → 65% of average volume
#
# Agent might learn: "High vol uptrend with low volume → be cautious, wait for volume confirmation"
```

---

### ChronosEmbeddingTransform

**Purpose**: Use pretrained Chronos foundation models to extract time series embeddings.

The `ChronosEmbeddingTransform` uses Amazon's Chronos T5-based forecasting models to embed market data into fixed-size feature vectors, similar to how VC1Transform works for vision.

#### Basic Usage

```python
from torchtrade.envs.transforms import ChronosEmbeddingTransform

env = TransformedEnv(
    base_env,
    ChronosEmbeddingTransform(
        in_keys=["market_data_1Minute_12"],
        out_keys=["chronos_embedding"],
        model_name="amazon/chronos-t5-large",
        aggregation="mean",
    )
)
```

See the [Chronos embedding example](../examples/transforms/chronos_embedding_example.py) for detailed usage.

---

### CoverageTracker

**Purpose**: Track which states/transitions have been visited during training.

The `CoverageTracker` helps analyze exploration coverage and identify underexplored regions of the state space.

```python
from torchtrade.envs.transforms import CoverageTracker

env = TransformedEnv(
    base_env,
    CoverageTracker()
)

# After training
coverage_stats = env.get_coverage_stats()
print(f"Unique states visited: {coverage_stats['unique_states']}")
```

---

## Composing Multiple Transforms

Transforms can be composed to add multiple features:

```python
from torchrl.envs import Compose

env = TransformedEnv(
    base_env,
    Compose(
        MarketRegimeTransform(
            in_keys=["market_data_1Minute_12"],
        ),
        ChronosEmbeddingTransform(
            in_keys=["market_data_5Minute_12"],
            out_keys=["chronos_5m"],
            model_name="amazon/chronos-t5-small",
        ),
        InitTracker(),
        RewardSum(),
    )
)

# Observations now include:
# - regime_features (7 elements)
# - chronos_5m (embedding_dim elements)
# - account_state (7 elements)
# - Original market data (if not deleted)
```

---

## Creating Custom Transforms

To create your own transform, inherit from `torchrl.envs.transforms.Transform`:

```python
from torchrl.envs.transforms import Transform
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDictBase
import torch

class MyCustomTransform(Transform):
    """Custom transform that adds new features."""

    def __init__(self, in_keys: List[str], out_keys: List[str], **kwargs):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        # Store parameters
        self.param = kwargs.get('param', default_value)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Process tensordict and add new features."""
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key not in tensordict.keys():
                continue

            input_data = tensordict.get(in_key)

            # Transform logic here
            output_data = self._transform(input_data)

            tensordict.set(out_key, output_data)

        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        """Apply transform to reset observations."""
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        """Update observation spec with new features."""
        spec = observation_spec.clone()

        for out_key in self.out_keys:
            spec.set(
                out_key,
                UnboundedContinuousTensorSpec(
                    shape=(output_dim,),
                    dtype=torch.float32
                )
            )

        return spec

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Implement your transformation logic."""
        # Your logic here
        return transformed_data
```

Key methods to implement:
1. `__init__`: Initialize parameters
2. `_call`: Main transformation logic
3. `_reset`: Handle reset observations
4. `transform_observation_spec`: Update observation specs

See [MarketRegimeTransform source](../torchtrade/envs/transforms/market_regime.py) for a complete example.

---

## Examples

- [Market Regime Example](../examples/transforms/market_regime_example.py)
- [Chronos Embedding Example](../examples/transforms/chronos_embedding_example.py)

---

## Best Practices

1. **Lazy initialization**: Load heavy models (like Chronos) lazily on first use
2. **Batched support**: Handle both batched and unbatched observations
3. **Spec transformation**: Always implement `transform_observation_spec`
4. **Caching**: Cache computed specs to avoid recomputation
5. **Error handling**: Gracefully handle missing keys
6. **Testing**: Write comprehensive tests for all transform behavior

---

## Related Issues

- Issue #55: Market Regime Features for Context-Aware Trading
