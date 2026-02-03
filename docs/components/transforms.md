# Transforms

TorchRL transforms are composable modules that modify environment observations, rewards, or actions. TorchTrade extends TorchRL's transform system with domain-specific transforms for trading.

## Available Transforms

| Transform | Purpose | Use Case |
|-----------|---------|----------|
| [**CoverageTracker**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/envs/transforms/coverage_tracker.py) | Track dataset coverage during training | Monitor exploration, detect overfitting |
| [**ChronosEmbeddingTransform**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/envs/transforms/chronos_embedding.py) | Embed time series with Chronos T5 models | Replace raw OHLCV with learned representations |
| [**TimestampTransform**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/envs/transforms/timestamp.py) | Add Unix timestamps to TensorDicts | Create offline datasets from live trading runs |

---

## CoverageTracker

Tracks which portions of the dataset are visited during training with random episode resets. Monitors both episode start diversity (reset coverage) and full trajectory coverage (state coverage). Auto-detects environment settings and only activates when `random_start=True`.

| Metric | Range | Meaning |
|--------|-------|---------|
| `reset_coverage` | [0, 1] | Fraction of positions used as episode starts |
| `state_coverage` | [0, 1] | Fraction of all states visited during episodes |
| `reset_entropy` / `state_entropy` | [0, log(N)] | Uniformity of visit distribution |

```python
from torchtrade.envs.transforms import CoverageTracker
from torchrl.collectors import SyncDataCollector

# Use as postproc in collector (NOT in environment transform chain)
coverage_tracker = CoverageTracker()

collector = SyncDataCollector(
    env, policy,
    frames_per_batch=1000,
    total_frames=100000,
    postproc=coverage_tracker,
)

for batch in collector:
    # ... train on batch ...
    stats = coverage_tracker.get_coverage_stats()
    if stats["enabled"]:
        logger.log({
            "train/reset_coverage": stats["reset_coverage"],
            "train/state_coverage": stats["state_coverage"],
        })
```

---

## ChronosEmbeddingTransform

Embeds time series observations using pretrained [Chronos T5](https://arxiv.org/abs/2403.07815) forecasting models. Replaces raw OHLCV data with learned representations. Model is lazy-loaded on first use.

| Model | Parameters | Embedding Dim | Memory |
|-------|------------|---------------|--------|
| chronos-t5-tiny | 8M | 512 | ~1GB |
| chronos-t5-mini | 20M | 512 | ~2GB |
| chronos-t5-small | 46M | 768 | ~3GB |
| chronos-t5-base | 200M | 768 | ~5GB |
| chronos-t5-large | 710M | 1024 | ~12GB |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_keys` | Required | Market data keys to embed |
| `out_keys` | Required | Output embedding keys |
| `model_name` | `"amazon/chronos-t5-large"` | HuggingFace model ID |
| `aggregation` | `"mean"` | `"mean"`, `"max"`, or `"concat"` for multi-feature embeddings |
| `del_keys` | `True` | Remove input keys after transformation |

```python
from torchrl.envs import TransformedEnv, Compose
from torchtrade.envs.transforms import ChronosEmbeddingTransform

env = TransformedEnv(
    base_env,
    Compose(
        ChronosEmbeddingTransform(
            in_keys=["market_data_1Minute_12"],
            out_keys=["chronos_embedding"],
            model_name="amazon/chronos-t5-base",
            aggregation="mean",
        ),
    )
)
# Observation: {"market_data_1Minute_12": (12, 5)} → {"chronos_embedding": (768,)}
```

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

---

## TimestampTransform

Adds Unix timestamps to TensorDicts on reset and step. Useful for creating offline datasets from live trading runs, debugging latency, and correlating trading decisions with real-world events.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `out_key` | `"timestamp"` | Key to store the timestamp |

```python
from torchrl.envs import TransformedEnv
from torchtrade.envs.transforms import TimestampTransform

env = TransformedEnv(base_env, TimestampTransform())

td = env.reset()           # td["timestamp"] = 1738617600.123
td = env.step(td)          # td["next", "timestamp"] = 1738617600.456

# Convert to datetime
from datetime import datetime
dt = datetime.fromtimestamp(td["timestamp"])
```

**Note:** Timestamps are wall-clock time (when `step()`/`reset()` is called), not execution timeframe timestamps. For execution timestamps, use the market data timestamps in observations.

---

## See Also

- [TorchRL Transforms](https://pytorch.org/rl/reference/envs.html#transforms) - Built-in transforms (VecNorm, ActionMask, etc.)
- [Feature Engineering](../guides/custom-features.md) - Manual feature engineering
- [Example: PPO + Chronos](../examples/index.md) - Training with Chronos embeddings
