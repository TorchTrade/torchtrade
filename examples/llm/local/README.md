# Local LLM Trading Examples

Use local LLMs as trading actors with TorchTrade via [vLLM](https://github.com/vllm-project/vllm) or [transformers](https://github.com/huggingface/transformers).

## Installation

```bash
pip install -e ".[llm]"
```

## Examples

### Offline Backtesting (`offline.py`)

Run a local LLM through a `SequentialTradingEnv` with historical data:

```bash
python examples/llm/local/offline.py
```

### Live Trading (`online.py`)

Run a local LLM on Alpaca paper trading, collecting trajectories into a replay buffer:

```bash
# Set API keys in .env
python examples/llm/local/online.py
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`.

## Recommended Models

| Model | Size | VRAM | Notes |
|-------|------|------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 500M | ~2GB | Fast, good for testing |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3GB | Good balance |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~8GB | Strong reasoning |

Use `quantization="4bit"` to reduce memory for larger models.

## Hardware

- **Minimum**: 8GB RAM, any CPU (use `backend="transformers"`, `device="cpu"`)
- **Recommended**: NVIDIA GPU 8GB+ VRAM, CUDA 11.8+ (use `backend="vllm"`)
