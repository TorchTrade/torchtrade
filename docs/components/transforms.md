# Transforms

TorchRL transforms are composable modules that modify environment observations, rewards, or actions. TorchTrade extends TorchRL's transform system with domain-specific transforms for trading environments.

## Available Transforms

| Transform | Type | Purpose | Use Case |
|-----------|------|---------|----------|
| **CoverageTracker** | Monitoring | Track dataset coverage during training | Monitor exploration, detect overfitting |
| **ChronosEmbeddingTransform** | Feature Extraction | Embed time series with Chronos T5 models | Replace raw OHLCV with learned representations |

---

## CoverageTracker

Transform that tracks which portions of the dataset are visited during training with random episode resets. Monitors both episode start diversity and full trajectory coverage.

### Key Features

- **Dual coverage tracking**:
    - **Reset coverage**: Which starting positions are used for episode starts
    - **State coverage**: All timesteps visited during episodes
- **Entropy metrics**: Measures uniformity of visit distribution
- **Zero overhead**: Tracking happens in postproc, outside critical environment path
- **Auto-detection**: Only activates for training environments with `random_start=True`

### When to Use

- **Dataset exploration analysis** - Ensure comprehensive coverage of training data
- **Overfitting detection** - Identify if agent concentrates on specific market conditions
- **Curriculum learning** - Track coverage progression across training phases
- **Data efficiency** - Verify all collected data is being utilized

### Key Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `reset_coverage` | [0.0, 1.0] | Fraction of positions used as episode starts (0.8+ is good) |
| `state_coverage` | [0.0, 1.0] | Fraction of all states visited (should be >> reset_coverage) |
| `reset_entropy` | [0, log(N)] | Uniformity of episode start distribution (higher = more uniform) |
| `state_entropy` | [0, log(N)] | Uniformity of state visit distribution (higher = better exploration) |

**Interpretation Guide**:

- **Good pattern**: `reset_coverage=0.3`, `state_coverage=0.9` → Starting from 30% of positions but exploring 90% of dataset through trajectories
- **Warning**: `reset_coverage=0.5`, `state_coverage=0.5` → Only seeing states near episode starts, not exploring forward in time
- **Ideal**: Both high coverage (>0.8) and high entropy (close to log(N))

### Usage Example

```python
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, Compose, InitTracker, DoubleToFloat, RewardSum
from torchtrade.envs.transforms import CoverageTracker

# Create environment with standard transforms
env = TransformedEnv(
    base_env,
    Compose(
        InitTracker(),
        DoubleToFloat(),
        RewardSum(),
    )
)

# Create coverage tracker for postproc (NOT in environment transform chain)
coverage_tracker = CoverageTracker()

# Use as postproc in collector
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=1000,
    total_frames=100000,
    postproc=coverage_tracker,  # Process batches after collection
)

# During training loop
for batch in collector:
    # ... train on batch ...

    # Log coverage metrics periodically
    stats = coverage_tracker.get_coverage_stats()
    if stats["enabled"]:
        logger.log({
            "train/reset_coverage": stats["reset_coverage"],
            "train/reset_entropy": stats["reset_entropy"],
            "train/state_coverage": stats["state_coverage"],
            "train/state_entropy": stats["state_entropy"],
        })
```

### Configuration

CoverageTracker requires no configuration - it auto-detects environment settings:

- **Initialization**: Automatically detects dataset size from environment sampler
- **Activation**: Only tracks when `env.random_start=True` (training mode)
- **Deactivation**: Silently disabled for sequential/eval environments

### Coverage Statistics

Full statistics returned by `get_coverage_stats()`:

```python
{
    "enabled": True,
    "total_positions": 100000,  # Dataset size

    # Reset coverage (episode start diversity)
    "reset_visited": 25000,
    "reset_coverage": 0.25,
    "total_resets": 50000,
    "reset_mean_visits": 2.0,
    "reset_max_visits": 150,
    "reset_min_visits": 0,
    "reset_std_visits": 5.2,
    "reset_entropy": 4.3,

    # State coverage (full trajectory coverage)
    "state_visited": 85000,
    "state_coverage": 0.85,
    "total_states": 500000,
    "state_mean_visits": 5.8,
    "state_max_visits": 500,
    "state_min_visits": 0,
    "state_std_visits": 12.1,
    "state_entropy": 6.1,
}
```

**Reference**: [`torchtrade/envs/transforms/coverage_tracker.py`](https://github.com/TorchTrade/TorchTrade/blob/main/torchtrade/envs/transforms/coverage_tracker.py)

---

## ChronosEmbeddingTransform

Transform that embeds time series observations using pretrained Chronos T5 forecasting models. Replaces raw OHLCV data with learned representations, similar to how VC1Transform works for vision.

### Key Features

- **Pretrained representations**: Leverages Amazon's Chronos T5 models trained on diverse time series
- **Lazy loading**: Model loaded on first use, not during initialization
- **Multiple model sizes**: From tiny (8M params) to large (710M params)
- **Flexible aggregation**: Mean, max, or concat for multi-feature embeddings
- **GPU acceleration**: Automatic device detection with bfloat16 precision

### When to Use

- **Limited training data** - Pretrained representations generalize better than learning from scratch
- **Transfer learning** - Embeddings trained on diverse time series work across assets
- **Feature engineering alternative** - Replace manual technical indicators with learned features
- **Model compression** - Reduce observation dimension while preserving information

### Available Models

| Model | Parameters | Embedding Dim | Memory | Use Case |
|-------|------------|---------------|--------|----------|
| chronos-t5-tiny | 8M | 512 | ~1GB | Testing, CI |
| chronos-t5-mini | 20M | 512 | ~2GB | Small deployments |
| chronos-t5-small | 46M | 768 | ~3GB | Balanced |
| chronos-t5-base | 200M | 768 | ~5GB | Standard (recommended) |
| chronos-t5-large | 710M | 1024 | ~12GB | Best performance |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_keys` | list[str] | Required | Market data keys to embed (e.g., `["market_data_1Minute_12"]`) |
| `out_keys` | list[str] | Required | Output embedding keys (e.g., `["chronos_embedding"]`) |
| `model_name` | str | "amazon/chronos-t5-large" | HuggingFace model identifier |
| `aggregation` | str | "mean" | How to aggregate multi-feature embeddings ("mean", "max", "concat") |
| `del_keys` | bool | True | Remove input keys after transformation |
| `device` | str | Auto-detect | Device for inference ("cuda", "cpu", "mps") |
| `torch_dtype` | dtype | torch.bfloat16 | Model precision (bfloat16 recommended for memory) |

### Usage Example

```python
from torchrl.envs import TransformedEnv, Compose
from torchtrade.envs.transforms import ChronosEmbeddingTransform

# Create environment with Chronos embedding
env = TransformedEnv(
    base_env,
    Compose(
        ChronosEmbeddingTransform(
            in_keys=["market_data_1Minute_12"],
            out_keys=["chronos_embedding"],
            model_name="amazon/chronos-t5-large",
            aggregation="mean",  # Average across OHLCV features
        ),
        # ... other transforms ...
    )
)

# Observation spec automatically updated
# Before: {"market_data_1Minute_12": (12, 5)}
# After:  {"chronos_embedding": (1024,)}  # For chronos-t5-large
```

### Aggregation Strategies

When input has multiple features (e.g., OHLCV has 5 features), choose aggregation method:

**Mean** (default):
```python
aggregation="mean"  # Output: (embedding_dim,)
# Averages embeddings across features
# Best for: Reducing dimensionality, treating features equally
```

**Max**:
```python
aggregation="max"  # Output: (embedding_dim,)
# Takes element-wise maximum across feature embeddings
# Best for: Highlighting strongest signals
```

**Concat**:
```python
aggregation="concat"  # Output: (num_features * embedding_dim,)
# Concatenates all feature embeddings
# Best for: Preserving feature-specific information, larger models
```

### Integration with append_transform

For environments already created, use `append_transform`:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Create environment
config = SeqLongOnlyEnvConfig(...)
env = SeqLongOnlyEnv(config)

# Append Chronos transform
env = env.append_transform(
    ChronosEmbeddingTransform(
        in_keys=env.market_data_keys,  # Use environment's market data keys
        out_keys=["chronos_embedding"],
        model_name="amazon/chronos-t5-base",
    )
)
```

### Performance Considerations

**Memory**:
- Use `torch_dtype=torch.bfloat16` to reduce memory footprint
- Smaller models (mini/small) for resource-constrained environments
- Embeddings are computed on CPU then moved to GPU to avoid OOM errors

**Speed**:
- First forward pass loads model (1-5 seconds depending on size)
- Subsequent passes are fast (~10-50ms per batch depending on model)
- Use vLLM backend for production deployment (not yet supported)

**Example Installation**:
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**Reference**: [`torchtrade/envs/transforms/chronos_embedding.py`](https://github.com/TorchTrade/TorchTrade/blob/main/torchtrade/envs/transforms/chronos_embedding.py)

---

## Quick Reference

### Transform Selection Guide

| Scenario | Recommended Transform | Notes |
|----------|----------------------|-------|
| Monitor data coverage | CoverageTracker | Use as collector postproc |
| Pretrained features | ChronosEmbeddingTransform | Replace raw OHLCV with embeddings |
| Standard preprocessing | TorchRL built-ins | DoubleToFloat, RewardSum, etc. |

### Common Patterns

**Pattern 1: Coverage Monitoring (Training)**
```python
# Create postproc transform (NOT in env transform chain)
coverage = CoverageTracker()

# Use in collector
collector = SyncDataCollector(
    env, policy,
    postproc=coverage,
)

# Log periodically
stats = coverage.get_coverage_stats()
```

**Pattern 2: Chronos Embedding (Feature Engineering)**
```python
# Replace raw OHLCV with learned embeddings
env = TransformedEnv(
    base_env,
    Compose(
        ChronosEmbeddingTransform(
            in_keys=["market_data_1Minute_12"],
            out_keys=["embedding"],
            model_name="amazon/chronos-t5-base",
        ),
        DoubleToFloat(),
        RewardSum(),
    )
)
```

**Pattern 3: Multi-Timeframe Embedding**
```python
# Embed multiple timeframes separately
ChronosEmbeddingTransform(
    in_keys=["market_data_1Minute_12", "market_data_5Minute_8"],
    out_keys=["embed_1min", "embed_5min"],
    aggregation="concat",  # Preserve all feature info
)
# Policy network concatenates embed_1min + embed_5min
```

---

## Implementation Notes

### CoverageTracker Design

- **Postproc usage**: Designed to run outside environment step loop for zero overhead
- **Dual tracking**: Separates episode start diversity from full trajectory coverage
- **Batch aggregation**: Uses `torch.unique` for efficient batch processing
- **Auto-initialization**: Detects dataset size from environment or first batch

### ChronosEmbeddingTransform Design

- **Lazy loading**: Model loaded on first forward pass, not __init__
- **CPU embedding**: Chronos requires CPU for embedding extraction (library limitation)
- **Observation spec update**: Automatically updates environment specs with embedding dimensions
- **Batch processing**: Handles both single and batched observations from parallel environments

---

## See Also

- [TorchRL Transforms Documentation](https://pytorch.org/rl/reference/envs.html#transforms) - Built-in transforms
- [Feature Engineering](../guides/custom-features.md) - Manual feature engineering patterns
- [Loss Functions](losses.md) - Training objectives that work with transforms
- [Example: Chronos Embedding](https://github.com/TorchTrade/TorchTrade/blob/main/examples/transforms/chronos_embedding_example.py) - Complete usage example
