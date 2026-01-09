# Local LLM Trading Actor Examples

This directory contains examples for using local language models (LLMs) for trading decisions using the [unsloth](https://github.com/unslothai/unsloth) library for fast inference.

## Overview

The `LocalLLMActor` provides an alternative to cloud-based APIs (like OpenAI) by running quantized LLMs locally. This approach offers:

- **Privacy**: Your trading data never leaves your machine
- **Cost**: No API fees for inference
- **Latency**: Faster inference without network requests
- **Control**: Full control over model behavior and fine-tuning

## Quick Start

### Installation

```bash
# Install unsloth and dependencies
pip install unsloth

# Or install with all optional dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install transformers and related packages
pip install torch transformers datasets trl
```

### Basic Usage

```python
from torchtrade.actor.local_llm_actor import LocalLLMActor

# Create actor with default Qwen3-0.6B model (4-bit quantized)
actor = LocalLLMActor(
    model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    load_in_4bit=True,
    debug=True
)

# Use with TorchRL environment
result = actor(tensordict)
action = result.get("action")  # 0=sell, 1=hold, 2=buy
```

## Examples

### 1. Basic Inference (`inference_example.py`)

Demonstrates how to use `LocalLLMActor` for trading decisions with mock market data.

```bash
python examples/llm/local/inference_example.py
```

**What it does:**
- Loads the Qwen3-0.6B 4-bit quantized model
- Creates sample market data (multi-timeframe OHLCV)
- Generates a trading decision with reasoning
- Shows the model's thinking process

### 2. Fine-tuning (`finetune_example.py`)

Shows how to fine-tune the local LLM on trading-specific data using LoRA.

```bash
python examples/llm/local/finetune_example.py
```

**What it does:**
- Loads the base Qwen3 model
- Configures LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Trains on example trading scenarios
- Saves the fine-tuned model

**Training data format:**
```python
{
    "instruction": "System prompt with trading rules",
    "input": "Account state + market data",
    "output": "<think>reasoning</think>\n<answer>action</answer>"
}
```

### 3. Data Collection for Fine-tuning

To create your own training data:

1. **Option A: Learn from successful strategies**
   ```python
   # Run a profitable rule-based or RL strategy
   # Collect (state, action, outcome) tuples
   # Filter for successful trades
   # Convert to prompt-response format
   ```

2. **Option B: Use existing LLM actor**
   ```python
   # Run LLMActor (GPT-based) to collect expert demonstrations
   # Use examples/live/alpaca/collect_live_llm.py
   # Convert collected trajectories to training format
   ```

3. **Option C: Human annotation**
   ```python
   # Show human experts market scenarios
   # Collect their decisions and reasoning
   # Format as training examples
   ```

## Model Selection

### Recommended Models

| Model | Size | Memory | Speed | Notes |
|-------|------|--------|-------|-------|
| `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` | 0.6B | ~2GB | Fast | **Recommended for Pi** |
| `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | 1.5B | ~4GB | Medium | Better reasoning |
| `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | 3B | ~6GB | Slower | High quality |

### Raspberry Pi Deployment

The Qwen3-0.6B 4-bit model can run on Raspberry Pi 4/5 with:
- At least 4GB RAM (8GB recommended)
- ~2GB storage for model weights
- Expected inference time: 2-5 seconds per decision

```python
# Raspberry Pi configuration
actor = LocalLLMActor(
    model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    max_seq_length=1024,  # Reduce for lower memory
    load_in_4bit=True,
    device="cpu"
)
```

## Integration with Trading Environments

### Live Trading (Alpaca)

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.actor.local_llm_actor import LocalLLMActor

# Create environment
config = AlpacaTradingEnvConfig(symbol="BTC/USD", paper=True)
env = AlpacaTorchTradingEnv(config, api_key="...", api_secret="...")

# Create local LLM actor
actor = LocalLLMActor()

# Collect data
from torchrl.collectors import SyncDataCollector
collector = SyncDataCollector(env, actor, frames_per_batch=1, total_frames=100)

for data in collector:
    print(f"Action: {data['action']}, Reward: {data['reward']}")
```

### Backtesting (Offline)

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.actor.local_llm_actor import LocalLLMActor

# Load historical data
import pandas as pd
df = pd.read_parquet("data/btcusd_1m.parquet")

# Create environment
config = SeqLongOnlyEnvConfig(initial_cash=10000)
env = SeqLongOnlyEnv(config, data=df)

# Test with local LLM
actor = LocalLLMActor()
tensordict = env.reset()

for _ in range(100):
    tensordict = actor(tensordict)
    tensordict = env.step(tensordict)
    if tensordict["done"]:
        break

print(f"Final reward: {tensordict['episode_reward']}")
```

## Performance Comparison

| Actor Type | Latency | Cost | Privacy | Customizable |
|------------|---------|------|---------|--------------|
| GPT-4 API | ~1-2s | High | ❌ | Limited |
| Local Qwen3-0.6B | ~0.5-2s | Free | ✅ | Full |
| Local Qwen2.5-1.5B | ~1-3s | Free | ✅ | Full |

## Fine-tuning Tips

1. **Start with good base data**: Use successful trades from profitable strategies
2. **Diverse scenarios**: Include bull markets, bear markets, sideways, volatile periods
3. **Clear reasoning**: Annotate why each action was taken
4. **Iterative improvement**: Fine-tune → test → collect → repeat
5. **Regularization**: Use LoRA dropout to prevent overfitting

## Troubleshooting

### Out of Memory

```python
# Reduce sequence length
actor = LocalLLMActor(max_seq_length=1024)

# Use smaller model
actor = LocalLLMActor(model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")

# Force CPU usage
actor = LocalLLMActor(device="cpu")
```

### Slow Inference

```python
# Enable Flash Attention (if supported)
FastLanguageModel.for_inference(model, use_flash_attention=True)

# Reduce max_new_tokens
# Edit in local_llm_actor.py: max_new_tokens=128 instead of 256

# Use smaller model
actor = LocalLLMActor(model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
```

### Model Download Issues

```python
# Use HuggingFace cache
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3-0.6B-unsloth-bnb-4bit")

# Then load from cache
actor = LocalLLMActor(model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
```

## Next Steps

1. **Experiment with models**: Try different model sizes and architectures
2. **Collect real data**: Use `collect_live_llm.py` with GPT to generate training data
3. **Fine-tune**: Use `finetune_example.py` with your collected data
4. **Backtest**: Test your fine-tuned model on historical data
5. **Deploy**: Run on paper trading before going live

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [TorchRL Documentation](https://pytorch.org/rl/)

## Contributing

Found a bug or have an improvement? Please open an issue or PR on the TorchTrade repository!
