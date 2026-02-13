# PPO Vectorized Environment Example

> **Experimental:** `VectorizedSequentialTradingEnv` is still experimental. While it passes extensive scalar-vectorized equivalence tests, it has not been battle-tested in production training runs yet. Use with caution and verify results against the standard `SequentialTradingEnv`.

This example demonstrates how to train a PPO agent using the `VectorizedSequentialTradingEnv` for significantly higher throughput compared to the standard `ParallelEnv` approach.

## Key Differences from Standard PPO

1. **Environment**: Uses `VectorizedSequentialTradingEnv` instead of `ParallelEnv` with `SequentialTradingEnv`
2. **Performance**: Achieves 20-400x higher throughput by eliminating inter-process communication overhead
3. **Model**: Default network type is `batchnorm_mlp` which works best with flattened observations
4. **Configuration**: Higher default `train_envs` (64) to take advantage of vectorization

## Usage

### Basic Training
```bash
cd examples/online_rl/ppo_vectorized
python train.py
```

### With Configuration Overrides
```bash
python train.py env.train_envs=128 model.hidden_size=256 optim.lr=1e-4
```

### Test Performance
```bash
# Compare vectorized vs standard
python ../../benchmarks/bench_sequential.py --benchmark vectorized
```

## Configuration

### Environment (`env/vectorized_sequential.yaml`)
- `train_envs`: Number of vectorized environments (default: 64)
- `time_frames`: Market data timeframes 
- `window_sizes`: Observation window sizes
- `action_levels`: Trading action levels [0.0, 1.0] for spot mode

### Model (`config.yaml`)
- `network_type`: "batchnorm_mlp" (recommended for vectorized)
- `hidden_size`: Neural network hidden layer size
- `num_layers`: Number of hidden layers

### Training
- `collector.frames_per_batch`: Frames per collection batch
- `collector.total_frames`: Total training frames
- `loss.ppo_epochs`: PPO update epochs per batch

## Performance Expectations

The vectorized environment should provide:
- **20-100x** higher throughput for small numbers of environments (8-32)
- **100-400x** higher throughput for large numbers of environments (64-256)
- **No IPC overhead** - all environments run in the same process
- **Better GPU utilization** with larger batch sizes

## Limitations

- SLTP (stop-loss/take-profit) bracket orders are not yet supported
- No domain randomization for `initial_cash` (tuple form)

## Monitoring

Training logs to Weights & Biases by default:
- Project: `TorchTrade-Vectorized`
- Group: `VectorizedSequentialTradingEnv`

Key metrics to monitor:
- `time/SPS-collecting`: Steps per second during data collection
- `time/SPS-total`: Overall steps per second
- `train/reward`: Training episode rewards
- `eval/reward`: Test episode rewards