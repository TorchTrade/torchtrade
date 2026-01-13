# PPO Training with Chronos Embeddings

This example demonstrates PPO training using pretrained Chronos models to embed market data observations before feeding them to the policy network.

## Overview

Instead of directly processing raw OHLCV data, this example uses Amazon's Chronos T5-based forecasting models to extract meaningful embeddings from time series market data. The embeddings are then concatenated and fed into the policy network.

**Key Features:**
- Uses `ChronosEmbeddingTransform` to convert market data into fixed-size embeddings
- Supports any Chronos model size (tiny, mini, small, base, large)
- Automatically handles multi-timeframe observations
- Combines Chronos embeddings with account state features

## Architecture

```
Market Data (OHLCV) → Chronos Embedding → Concatenate with Account State → MLP → Policy/Value Heads
                          Transform
```

The pipeline:
1. Market data observations are embedded using the pretrained Chronos model
2. Account state features are kept as-is and normalized
3. All features are concatenated
4. A simple MLP processes the concatenated features
5. Policy and value heads predict actions and state values

## Configuration

The example uses a 15-minute timeframe for both observation and execution:

```yaml
env:
  time_frames: [15]     # 15-minute bars
  window_sizes: [32]    # 32 timesteps lookback
  execute_on: [15, "Min"]  # Execute trades on 15-minute bars

model:
  chronos_model: "amazon/chronos-t5-small"  # Chronos small model
```

### Available Chronos Models

- `amazon/chronos-t5-tiny` (8M params) - For testing and CI
- `amazon/chronos-t5-mini` (20M params) - Small deployments
- `amazon/chronos-t5-small` (46M params) - **Default** - Balanced
- `amazon/chronos-t5-base` (200M params) - Standard
- `amazon/chronos-t5-large` (710M params) - Best performance

## Usage

### Train the agent

```bash
cd examples/online/ppo_chronos
python train.py
```

### Override configuration

```bash
# Use a different Chronos model
python train.py model.chronos_model="amazon/chronos-t5-large"

# Change learning rate
python train.py optim.lr=1e-5

# Use different timeframe
python train.py env.time_frames=[30] env.window_sizes=[64] env.execute_on=[30,Min]

# Disable W&B logging
python train.py logger.backend=null
```

## Features vs Raw Market Data

**Advantages of Chronos Embeddings:**
- Pretrained on diverse time series data
- Captures temporal patterns automatically
- Fixed-size representation regardless of window size
- May generalize better to unseen market conditions

**Disadvantages:**
- Requires model download (~46MB for small, ~710MB for large)
- Slower inference compared to raw features
- Less interpretable than handcrafted features
- Fixed preprocessing (can't customize normalization)

## Performance Notes

- **Memory**: Chronos models require additional GPU memory. The small model uses ~100MB.
- **Speed**: Embedding computation adds overhead. Expect ~20-30% slower collection vs raw features.
- **Recommendation**: Start with `chronos-t5-small` for initial experiments, upgrade to `large` for best performance.

## Extending This Example

### Use Multiple Timeframes

```yaml
env:
  time_frames: [5, 15, 60]
  window_sizes: [32, 32, 24]
  freqs: ["Min", "Min", "Min"]
```

Each timeframe will get its own Chronos embedding, and all will be concatenated.

### Custom Aggregation

In `utils.py`, change the aggregation method:

```python
chronos_transform = ChronosEmbeddingTransform(
    in_keys=[market_key],
    out_keys=[out_key],
    model_name=cfg.model.chronos_model,
    aggregation="mean",  # Options: "mean", "max", "concat"
    ...
)
```

- `"concat"`: Concatenate embeddings across features (default, highest info)
- `"mean"`: Average embeddings (smaller representation)
- `"max"`: Max-pooling (highlights extreme values)

### Keep Raw Features

Set `del_keys=False` to keep both raw and embedded features:

```python
chronos_transform = ChronosEmbeddingTransform(
    in_keys=[market_key],
    out_keys=[out_key],
    del_keys=False,  # Keep raw market data
    ...
)
```

Then update the model to process both.

## Dependencies

Requires the `chronos-forecasting` package:

```bash
pip install chronos-forecasting
```

This is automatically installed when you install TorchTrade with the `[chronos]` extra:

```bash
pip install -e ".[chronos]"
```

## References

- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [TorchRL Transforms Documentation](https://pytorch.org/rl/stable/reference/envs.html#transforms)
