# Local LLM Trading Actors

This directory contains examples demonstrating how to use local Large Language Models (LLMs) as trading actors with TorchTrade.

## Overview

The `LocalLLMActor` enables trading with local LLMs using either [vllm](https://github.com/vllm-project/vllm) (GPU-optimized) or [transformers](https://github.com/huggingface/transformers) (CPU/GPU compatible) backends. This provides:

- **Privacy**: Models run locally without API calls
- **Cost-effective**: No per-token API costs
- **Customizable**: Use any HuggingFace model or fine-tuned model
- **Performance**: vllm provides 10-20x faster inference than standard transformers

## Installation

Install the required dependencies:

```bash
pip install -e ".[llm_local]"
```

This installs:
- `vllm>=0.6.0` - Fast GPU inference engine
- `transformers>=4.30.0` - HuggingFace transformers library
- `accelerate>=0.20.0` - Distributed/quantized model loading
- `bitsandbytes>=0.41.0` - 4-bit/8-bit quantization

### Hardware Requirements

**Minimum (CPU-only with transformers)**:
- 8GB RAM
- Any CPU (Qwen2.5-0.5B runs on Raspberry Pi 4)

**Recommended (GPU with vllm)**:
- NVIDIA GPU with 8GB+ VRAM (RTX 3060, 4060, etc.)
- CUDA 11.8 or newer
- 16GB system RAM

## Recommended Models

### Tiny Models (Fast, Low Memory)
| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | ~2GB | Ultra-fast, runs on Raspberry Pi |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~4GB | High quality for size |

### Medium Models (Better Performance)
| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3GB | Good balance of speed/quality |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~8GB | Strong reasoning capabilities |
| `meta-llama/Llama-3.2-8B-Instruct` | 8B | ~10GB | Excellent instruction following |

### Large Models (Best Quality)
| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~16GB | Professional-grade reasoning |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | ~40GB | State-of-the-art (requires A100) |

## Examples

### 1. Basic Inference Example (`inference_example.py`)

A simple demonstration showing LocalLLMActor with SeqLongOnlyEnv (3-action trading).

**What it demonstrates**:
- Loading local LLM (Qwen2.5-0.5B-Instruct)
- Basic buy/sell/hold trading
- Multi-timeframe observations
- Extracting reasoning traces

**Run it**:
```bash
python examples/llm/local/inference_example.py
```

**Expected output**:
- Model initialization logs
- 10 trading steps with actions and reasoning
- Episode summary with rewards

**Customization**:
```python
# Use a different model
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-1.5B-Instruct",  # Change model
    backend="vllm",
    debug=True,  # See full prompts/responses
)

# Use CPU (transformers fallback)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    backend="transformers",
    device="cpu",
)

# Use quantization (reduce memory)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",  # 4-bit quantization
)
```

### 2. Futures SLTP Example (`futures_sltp_example.py`)

Advanced demonstration using SeqFuturesSLTPEnv (futures with leverage and bracket orders).

**What it demonstrates**:
- Futures trading with leverage (5x)
- Combinatorial action space (long/short with SL/TP)
- Multi-timeframe observations (1m, 5m, 15m, 1h)
- Episode metrics and performance analysis

**Run it**:
```bash
python examples/llm/local/futures_sltp_example.py
```

**Expected output**:
- Action space breakdown (hold, long with SL/TP, short with SL/TP)
- 50 trading steps with position updates
- Action distribution analysis
- Performance metrics (return, Sharpe, drawdown)

**Key features**:
```python
# Combinatorial action space
action_map = futures_sltp_action_map(
    stoploss_levels=(-0.02, -0.05),     # -2%, -5%
    takeprofit_levels=(0.05, 0.1),      # +5%, +10%
)
# Results in: 1 (hold) + 2×2 (long) + 2×2 (short) = 9 actions

# LocalLLMActor handles this automatically
actor = LocalLLMActor(
    action_space_type="futures_sltp",
    action_map=action_map,  # Pass the action map
)
```

## Performance Tips

### 1. Use vllm for Production
```python
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    backend="vllm",  # 10-20x faster than transformers
    device="cuda",
)
```

### 2. Enable Quantization for Larger Models
```python
# 4-bit quantization (1/4 memory, minimal quality loss)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",
)

# 8-bit quantization (1/2 memory, better quality)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="8bit",
)
```

### 3. Batch Multiple Environments (Advanced)
For production systems, batch multiple environments to maximize GPU utilization:
```python
# Coming soon: MultiEnv support
```

### 4. Adjust Temperature for Exploration
```python
# Low temperature = more deterministic (exploitation)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.3,
)

# High temperature = more random (exploration)
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=1.0,
)
```

## Troubleshooting

### vllm installation fails
```bash
# vllm requires CUDA 11.8+
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# Or use transformers fallback (CPU/GPU compatible)
actor = LocalLLMActor(backend="transformers", device="cpu")
```

### Out of memory (OOM)
```bash
# Option 1: Use smaller model
actor = LocalLLMActor(model="Qwen/Qwen2.5-0.5B-Instruct")

# Option 2: Enable quantization
actor = LocalLLMActor(model="Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Option 3: Use CPU (slower but works)
actor = LocalLLMActor(backend="transformers", device="cpu")
```

### Model download is slow
Models are cached in `~/.cache/huggingface/`. First download takes time but subsequent runs are instant.

```bash
# Pre-download a model
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

### Actions are always "hold" or invalid
Enable debug mode to see prompts and responses:
```python
actor = LocalLLMActor(debug=True)
```

Common issues:
- Model not following `<answer>action</answer>` format → Try a better instruction-tuned model
- Model hallucinating actions → Reduce temperature or use smaller, more focused model

## Action Space Handling

### Standard 3-Action (SeqLongOnly, SeqFutures)
```python
actor = LocalLLMActor(
    action_space_type="standard",  # Default
)
# Actions: buy, sell, hold
# Model outputs: <answer>buy</answer>, <answer>sell</answer>, <answer>hold</answer>
```

### SLTP (SeqLongOnlySLTP, SeqFuturesSLTP)
```python
from torchtrade.envs.offline.seqfuturessltp import futures_sltp_action_map

action_map = futures_sltp_action_map(
    stoploss_levels=(-0.02, -0.05),
    takeprofit_levels=(0.05, 0.1),
)

actor = LocalLLMActor(
    action_space_type="futures_sltp",
    action_map=action_map,
)
# Actions: 0, 1, 2, ..., 8
# Model outputs: <answer>0</answer>, <answer>1</answer>, etc.
```

## Advanced: Fine-Tuning (Coming Soon)

Fine-tune a local model on historical trading data to improve decision quality:

```bash
# Coming soon: Fine-tuning guide
# See: examples/llm/local/finetune/ (planned)
```

## Comparison: Local LLM vs OpenAI API

| Feature | LocalLLMActor | LLMActor (OpenAI) |
|---------|--------------|-------------------|
| **Cost** | Free (after hardware) | $$ per 1M tokens |
| **Privacy** | Fully local | Data sent to API |
| **Latency** | ~50-200ms (vllm) | ~500-2000ms (network) |
| **Quality** | Model-dependent | GPT-4 level |
| **Customization** | Full control | Limited |
| **Hardware** | GPU recommended | None required |

## Contributing

Have a cool example or improvement? Submit a PR!

## Support

- Issues: https://github.com/TorchTrade/torchrl_alpaca_env/issues
- Discussions: https://github.com/TorchTrade/torchrl_alpaca_env/discussions

## License

MIT
