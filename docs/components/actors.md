# Trading Actors

Trading actors implement the policy interface for TorchTrade environments. Beyond standard neural network policies, TorchTrade provides rule-based strategies, LLM-powered agents, and human-in-the-loop interfaces.

## Available Actors

| Actor | Type | Use Case | Key Features |
|-------|------|----------|--------------|
| **RuleBasedActor** | Deterministic Strategy | Baselines, debugging, research benchmarks | Technical indicators, no learning required |
| **MeanReversionActor** | Rule-Based (Bollinger + Stoch RSI) | Ranging markets, baseline comparisons | ~0.3-0.8 Sharpe on sideways markets |
| **LLMActor** | LLM (OpenAI API) | Research, rapid prototyping | GPT-4/5 for decision-making |
| **LocalLLMActor** | LLM (Local inference) | Production, privacy, cost efficiency | vLLM or transformers backend, quantization support |

---

## RuleBasedActor (Base Class)

Abstract base class for implementing deterministic trading strategies using technical indicators. Rule-based actors provide strong baselines and help validate environment design without RL complexity.

### Key Features

- **Deterministic**: No randomness, reproducible results
- **Preprocessing pattern**: Compute indicators on full dataset upfront
- **Feature extraction**: Helper methods to access preprocessed indicators
- **Debugging support**: Built-in debug mode for step-by-step analysis

### When to Use

- **Baselines**: Compare RL agents against known strategies
- **Environment validation**: Verify reward functions and environment mechanics
- **Research benchmarks**: Establish performance floors
- **Strategy prototyping**: Test ideas before implementing in RL

### Implementation Pattern

Rule-based actors follow a two-phase pattern:

1. **Preprocessing**: Compute all technical indicators on full dataset
2. **Decision-making**: Extract features and apply rules at each step

```python
from torchtrade.actor.rulebased.base import RuleBasedActor

class MyStrategy(RuleBasedActor):
    def get_preprocessing_fn(self):
        """Return function that computes indicators on full dataset."""
        def preprocess(df):
            # Compute indicators (e.g., moving averages, RSI)
            df["features_sma_20"] = df["close"].rolling(20).mean()
            df["features_rsi_14"] = compute_rsi(df["close"], 14)
            return df
        return preprocess

    def select_action(self, observation):
        """Apply trading rules based on current observation."""
        data = self.extract_market_data(observation)
        sma = self.get_feature(data, "features_sma_20")[-1]
        rsi = self.get_feature(data, "features_rsi_14")[-1]

        # Trading logic
        if rsi < 30 and price < sma:
            return 2  # Buy
        elif rsi > 70 and price > sma:
            return 0  # Sell
        return 1  # Hold
```

---

## MeanReversionActor

Concrete implementation of RuleBasedActor using Bollinger Bands and Stochastic RSI for mean reversion trading.

### Strategy Logic

- **Buy signal**: Price below lower Bollinger Band AND Stoch RSI bullish crossover from oversold (<20) AND volume confirmation
- **Sell signal**: Price above upper Bollinger Band AND Stoch RSI bearish crossover from overbought (>80) AND volume confirmation
- **Hold**: Otherwise

### Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Sharpe Ratio** | 0.3 - 0.8 | Varies by market regime |
| **Action Distribution** | 30% long, 30% short, 40% hold | Balanced strategy |
| **Best Markets** | Ranging/sideways | Mean reversion works when prices oscillate |
| **Worst Markets** | Strong trends | Gets caught in persistent directional moves |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bb_window` | int | 20 | Bollinger Bands period |
| `bb_std` | float | 2.0 | Bollinger Bands standard deviations |
| `stoch_rsi_window` | int | 14 | Stochastic RSI period |
| `stoch_k_window` | int | 3 | Stochastic %K smoothing window |
| `stoch_d_window` | int | 3 | Stochastic %D smoothing window |
| `oversold_threshold` | float | 20.0 | Stoch RSI oversold level |
| `overbought_threshold` | float | 80.0 | Stoch RSI overbought level |
| `execute_timeframe` | TimeFrame | 5Minute | Timeframe for feature extraction |

### Technical Indicator References

- **Bollinger Bands**: Bollinger, J. (2001). "Bollinger on Bollinger Bands". McGraw-Hill Education
- **Stochastic RSI**: Introduced by Stanley Kroll and Tushar Chande (1994) in "The New Technical Trader"

### Usage Example

```python
from torchtrade.actor.rulebased.meanreversion import MeanReversionActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

# Create actor
actor = MeanReversionActor(
    market_data_keys=["market_data_1Minute_12", "market_data_5Minute_8"],
    features_order=["open", "high", "low", "close", "volume",
                    "features_bb_middle", "features_bb_std", "features_bb_upper",
                    "features_bb_lower", "features_bb_position",
                    "features_stoch_rsi_k", "features_stoch_rsi_d"],
    bb_window=20,
    oversold_threshold=20,
    debug=False,
)

# Get preprocessing function
preprocessing_fn = actor.get_preprocessing_fn()

# Create environment with preprocessing
config = SequentialTradingEnvConfig(
    df=train_df,
    preprocessing_fn=preprocessing_fn,  # Compute indicators upfront
    # ... other config ...
)
env = SequentialTradingEnv(df, config)

# Use actor to trade
observation = env.reset()
while not done:
    action = actor.select_action(observation)
    observation, reward, done, info = env.step(action)
```

**Code Reference**: `torchtrade/actor/rulebased/meanreversion/actor.py`

---

## LLMActor

LLM-based trading actor using OpenAI API (GPT-4, GPT-5) for decision-making. The actor constructs prompts from market data and account state, queries the LLM, and parses actions from structured responses.

### Key Features

- **Natural language reasoning**: LLM explains decisions in `<think>` tags
- **Flexible action spaces**: Supports standard (buy/sell/hold) and custom actions
- **Structured prompts**: Market data formatted as tables for readability
- **OpenAI API**: Uses latest GPT models

### When to Use

- **Research**: Explore LLM capabilities for trading
- **Rapid prototyping**: Test strategies without training neural networks
- **Interpretability**: Get explanations for each decision
- **Multi-modal agents**: Combine with other data sources (news, sentiment)

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `market_data_keys` | list[str] | Required | TensorDict keys for market data (e.g., `["market_data_1Minute_12"]`) |
| `account_state` | list[str] | Required | Account state field names (e.g., `["cash", "position_size", ...]`) |
| `model` | str | "gpt-5-nano" | OpenAI model identifier |
| `symbol` | str | "BTC/USD" | Trading symbol for prompt context |
| `execute_on` | str | "5Minute" | Timeframe for prompt context |
| `action_dict` | dict | `{"buy": 2, "sell": 0, "hold": 1}` | Action name to index mapping |
| `debug` | bool | False | Print prompts and responses |

### Usage Example

```python
from torchtrade.actor import LLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

# Create environment
config = SequentialTradingEnvConfig(...)
env = SequentialTradingEnv(df, config)

# Create LLM actor (uses environment attributes)
actor = LLMActor(
    market_data_keys=env.market_data_keys,
    account_state=env.account_state,
    symbol=config.symbol,
    execute_on=config.execute_on,
    model="gpt-4-turbo",
    debug=True,  # Print prompts/responses
)

# Trade
observation = env.reset()
output = actor(observation)  # Returns tensordict with "action" and "thinking"
action = output["action"]
thinking = output.get("thinking", "")  # LLM's reasoning
```

### Prompt Format

LLMActor constructs prompts with:

**System Prompt**:
```
You are a disciplined trading agent for BTC/USD on the 5Minute timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action from: buy, sell, hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
```

**User Prompt** (example):
```
Current account state:
cash: 10000.0
position_size: 0.0
position_value: 0.0
entry_price: 0.0
current_price: 50000.0
unrealized_pnlpct: 0.0
holding_time: 0.0

---
Current market data:

market_data_5Minute_8:

   close |     open |     high |      low |   volume

50123.4 | 50100.2 | 50150.0 | 50050.0 |   125.3
50156.1 | 50123.4 | 50200.5 | 50100.0 |   142.7
...
```

**LLM Response**:
```
<think>
The price has been trending upward over the last 8 candles, with increasing volume.
Current price is near resistance at 50200. RSI would likely be overbought.
I should wait for a pullback before entering.
</think>
<answer>hold</answer>
```

**Requirements**: `.env` file with `OPENAI_API_KEY`

**Code Reference**: `torchtrade/actor/frontier_llm_actor.py`

---

## LocalLLMActor

Local LLM-based actor using vLLM or transformers for inference. Similar to LLMActor but runs models locally for privacy, cost efficiency, and production deployment.

### Key Features

- **Local inference**: No API calls, full control over model
- **Multiple backends**:
    - **vLLM**: High-throughput inference with PagedAttention (recommended)
    - **transformers**: HuggingFace compatibility, easier setup
- **Quantization support**: 4-bit and 8-bit quantization for memory efficiency
- **Auto-detection**: Automatically detects environment type (standard, SLTP, futures)
- **Flexible action spaces**: Supports standard 3-action, SLTP, and futures environments

### Backend Comparison

| Backend | Speed | Setup | Use Case |
|---------|-------|-------|----------|
| **vLLM** | 5-10x faster | Requires CUDA | Production, high-throughput |
| **transformers** | Baseline | CPU/GPU/MPS | Development, compatibility |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "Qwen/Qwen2.5-0.5B-Instruct" | HuggingFace model ID |
| `backend` | str | "vllm" | Inference backend ("vllm" or "transformers") |
| `device` | str | Auto-detect | Device for inference ("cuda", "cpu", "mps") |
| `quantization` | str | None | Quantization mode (None, "4bit", "8bit") |
| `max_tokens` | int | 512 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `gpu_memory_utilization` | float | 0.9 | Fraction of GPU memory for vLLM (0.0-1.0) |
| `action_space_type` | str | "standard" | Action space ("standard", "sltp", "futures_sltp") |
| `action_map` | dict | None | Required for SLTP environments |
| `debug` | bool | False | Print prompts and responses |

### Usage Example

**Standard 3-Action Environment**:
```python
from torchtrade.actor import LocalLLMActor

# Create local LLM actor
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    backend="vllm",
    quantization="4bit",  # Use 4-bit quantization for memory
    temperature=0.7,
    debug=False,
)

# Use like any actor
observation = env.reset()
output = actor(observation)
action = output["action"]
```

**SLTP Environment (Bracket Orders)**:
```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP

# Get action map from environment
env = SequentialTradingEnvSLTP(df, config)
action_map = env.action_map  # Dict mapping action indices to (side, sl, tp)

# Create actor for SLTP
actor = LocalLLMActor(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    action_space_type="sltp",
    action_map=action_map,
    backend="transformers",  # Fallback to transformers
)
```

**Futures Environment with Leverage**:
```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP

env = SequentialTradingEnvSLTP(df, config)

actor = LocalLLMActor(
    model="Qwen/Qwen2.5-3B-Instruct",
    action_space_type="futures_sltp",
    action_map=env.action_map,
    backend="vllm",
    gpu_memory_utilization=0.8,
)
```

### Quantization Benefits

| Quantization | Memory Reduction | Speed | Quality Loss |
|--------------|------------------|-------|--------------|
| None (fp16) | Baseline | Baseline | None |
| 8-bit | ~2x | ~1.5x faster | Minimal |
| 4-bit | ~4x | ~2x faster | Small (~2-3% accuracy) |

**Recommendation**: Use 4-bit for models >1B parameters to fit in consumer GPUs.

### Model Recommendations

| Model | Size | Memory (4-bit) | Use Case |
|-------|------|----------------|----------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | ~500MB | Quick testing |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~1.5GB | Lightweight production |
| Qwen/Qwen2.5-3B-Instruct | 3B | ~2.5GB | Balanced performance |
| Qwen/Qwen2.5-7B-Instruct | 7B | ~5GB | Best quality |

**Installation**:
```bash
# For vLLM backend (recommended)
pip install vllm

# For transformers backend
pip install transformers accelerate bitsandbytes
```

**Code Reference**: `torchtrade/actor/local_llm_actor.py`

---

## Quick Reference

### Actor Selection Guide

| Scenario | Recommended Actor | Notes |
|----------|-------------------|-------|
| Baseline comparison | MeanReversionActor | Known strategy, deterministic |
| RL training | Neural network policy | Standard TorchRL actor-critic |
| Research prototyping | LLMActor | Fast iteration, interpretable |
| Production LLM | LocalLLMActor | Local inference, cost-efficient |

### Common Patterns

**Pattern 1: Rule-Based Baseline**
```python
# Test environment with known strategy first
actor = MeanReversionActor(...)
preprocessing_fn = actor.get_preprocessing_fn()
env = SequentialTradingEnv(df, config)
# ... run episodes ...
```

**Pattern 2: LLM Exploration**
```python
# Quickly test ideas without training
actor = LocalLLMActor(model="Qwen/Qwen2.5-1.5B", quantization="4bit")
# ... trade ...
```

---

## Implementation Notes

### RuleBasedActor Preprocessing

Rule-based actors compute all indicators upfront on the full dataset:

```python
preprocessing_fn = actor.get_preprocessing_fn()
df = preprocessing_fn(df)  # Compute features once
# Features stored in DataFrame, accessed during episodes
```

This is efficient because indicators like Bollinger Bands require historical data, so computing them once is faster than recalculating at each step.

### LLM Prompt Engineering

Both LLMActor and LocalLLMActor use structured prompts:

- **System prompt**: Defines role, action space, output format
- **User prompt**: Current state (account + market data)
- **Response format**: `<think>reasoning</think><answer>action</answer>`

The `<think>` tags encourage chain-of-thought reasoning, improving decision quality.

### LocalLLMActor Backend Selection

vLLM is faster but requires:
- CUDA GPU
- Linux (Windows via WSL)
- `pip install vllm`

transformers works everywhere:
- CPU/GPU/MPS (Apple Silicon)
- Cross-platform
- `pip install transformers`

If vLLM import fails, LocalLLMActor automatically falls back to transformers.

---

## See Also

- [Environments](../environments/offline.md) - Compatible environment types
- [TorchRL Actors](https://pytorch.org/rl/reference/modules.html#actors) - Building neural network policies
