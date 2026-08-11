# Trading Actors

Trading actors implement the policy interface for TorchTrade environments. Beyond standard neural network policies, TorchTrade provides rule-based strategies and LLM-powered agents.

## Available Actors

| Actor | Type | Use Case |
|-------|------|----------|
| [**RuleBasedActor**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/rulebased/base.py) | Deterministic Strategy | Baselines, debugging, research benchmarks |
| [**MeanReversionActor**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/rulebased/meanreversion/actor.py) | Rule-Based (Bollinger + Stoch RSI) | Ranging markets, baseline comparisons |
| [**FrontierLLMActor**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/frontier_llm_actor.py) | LLM (API) | Research, rapid prototyping with GPT/Claude |
| [**LocalLLMActor**](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/local_llm_actor.py) | LLM (Local inference) | Production, privacy, cost efficiency |

---

## RuleBasedActor

Abstract base class for deterministic trading strategies. Follows a two-phase pattern: **preprocess** (compute indicators on full dataset upfront) then **decide** (extract features and apply rules at each step).

```python
from torchtrade.actor.rulebased.base import RuleBasedActor

class MyStrategy(RuleBasedActor):
    def get_preprocessing_fn(self):
        def preprocess(df):
            df["features_sma_20"] = df["close"].rolling(20).mean()
            df["features_rsi_14"] = compute_rsi(df["close"], 14)
            return df
        return preprocess

    def select_action(self, observation):
        data = self.extract_market_data(observation)
        sma = self.get_feature(data, "features_sma_20")[-1]
        rsi = self.get_feature(data, "features_rsi_14")[-1]

        if rsi < 30 and price < sma:
            return 2  # Buy
        elif rsi > 70 and price > sma:
            return 0  # Sell
        return 1  # Hold
```

### MeanReversionActor

Concrete implementation using Bollinger Bands and Stochastic RSI. Buys when price is below lower band with oversold Stoch RSI, sells when above upper band with overbought Stoch RSI.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bb_window` | 20 | Bollinger Bands period |
| `bb_std` | 2.0 | Bollinger Bands standard deviations |
| `stoch_rsi_window` | 14 | Stochastic RSI period |
| `oversold_threshold` | 20.0 | Stoch RSI oversold level |
| `overbought_threshold` | 80.0 | Stoch RSI overbought level |

See `examples/rule_based/` for offline and live usage examples.

---

## FrontierLLMActor

LLM-based actor using frontier model APIs (OpenAI, Anthropic, etc.) for trading decisions. Constructs prompts from market data and account state, queries the LLM, and parses actions from structured `<think>...<answer>` responses.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-5-nano"` | Model identifier |
| `symbol` | `"BTC/USD"` | Trading symbol for prompt context |
| `action_dict` | `{"buy": 2, "sell": 0, "hold": 1}` | Action name to index mapping |
| `debug` | `False` | Print prompts and responses |

```python
from torchtrade.actor import FrontierLLMActor

actor = FrontierLLMActor(
    market_data_keys=env.market_data_keys,
    account_state=env.account_state,
    model="gpt-4-turbo",
)

observation = env.reset()
output = actor(observation)  # Returns tensordict with "action" and "thinking"
```

Requires `OPENAI_API_KEY` in `.env`. See `examples/llm/frontier/` for offline and live examples.

---

## LocalLLMActor

Local LLM-based actor using vLLM or transformers for inference. Same prompt interface as FrontierLLMActor but runs models locally.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"Qwen/Qwen2.5-0.5B-Instruct"` | HuggingFace model ID |
| `backend` | `"vllm"` | `"vllm"` (faster, CUDA) or `"transformers"` (portable) |
| `quantization` | `None` | `None`, `"4bit"`, or `"8bit"` |
| `max_tokens` | `512` | Maximum tokens to generate |
| `temperature` | `0.7` | Sampling temperature |
| `action_space_type` | `"standard"` | `"standard"`, `"sltp"`, or `"futures_sltp"` |

```python
from torchtrade.actor import LocalLLMActor

actor = LocalLLMActor(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    backend="vllm",
    quantization="4bit",
)

output = actor(observation)
```

For SLTP environments, pass `action_space_type="sltp"` and `action_map=env.action_map`. See `examples/llm/local/` for offline and live examples.

```bash
pip install torchtrade[llm]  # Installs vllm, transformers, bitsandbytes
```

### Batched / Parallel-Env Inference

`LocalLLMActor` accepts a batched tensordict (`batch_size=[N]`), such as one
produced by `ParallelEnv`, and generates N trading decisions in a single pass:
it builds N prompts, runs one vLLM call, and writes N actions back into the
tensordict. A single, unbatched observation still works exactly as before, so
existing offline/live scripts require no changes. `generate_batch` is the
extension point for adding batched generation to new backends. See
`examples/llm/local/parallel.py` for a full example.

### Action Extraction

Both LLM actors parse the model's chosen action from a `<answer>N</answer>` tag.
That logic lives in one reusable pure function:

```python
from torchtrade.actor.parsers import extract_action

idx = extract_action("<think>...</think><answer>2</answer>", num_actions=3)  # -> 2
```

It returns the integer `N` when `0 <= N < num_actions`, and falls back to action
`0` (logging a warning) when the tag is missing or out of range — a trading agent
must always emit a valid action.

### Tool use

Configure `tools=[...]` to let the actor call tools before deciding. Tools run
only when configured (live path); without them the actor is single-shot as before.

```python
from torchtrade.actor import LocalLLMActor
from torchtrade.actor.tools import GoogleNewsTool, PolymarketTool

actor = LocalLLMActor(
    model="Qwen/Qwen2.5-0.5B-Instruct", backend="vllm",
    market_data_keys=env.market_data_keys,
    account_state_labels=env.account_state,
    action_levels=env.action_levels,
    symbol="BTC/USD",
    tools=[
        GoogleNewsTool(symbol="BTC/USD"),
        PolymarketTool(symbol="BTC/USD"),
    ],
    max_tool_iters=3,
)
```

Pass tools individually rather than bundling sources behind flags — the list is
the configuration. Several tools still cost a single tool iteration, because the
model may emit more than one `<tool>` block per turn and they all resolve into
one `<tool_results>` block.

The model calls a tool with `<tool name="google_news">{"query": "Bitcoin"}</tool>`
(torchrl `XMLBlockParser` convention) and receives a `<tool_results>...</tool_results>`
block, then continues until it emits `<answer>N</answer>`. Only conversations that
call a tool are re-generated, so batched multi-symbol inference stays efficient.
Tool use requires `backend="vllm"` (the transformers backend can't halt at
`</tool>`) and the `[llm]` extra (adds `feedparser`).

#### `GoogleNewsTool`

Recent headlines for the traded asset, from Google News RSS — free, no key. Requires
the `[llm]` extra (adds `feedparser`).

- **Headlines are a third-party trust boundary.** `title`, `source` and `published`
  are authored by whoever gets a story indexed by Google News, and unlike
  `PolymarketTool` there is no volume floor or other content filter deciding what
  reaches the prompt — only a `top_n` count cap. Every rendered field is collapsed to
  a single capped line before it enters the context, because a newline would
  otherwise let one entry occupy two numbered rows and fabricate a headline the model
  cannot distinguish from genuine tool output (#308).
- **That guard covers row forgery and length, not inline markup.** A literal
  `</tool_results>` in a headline would still close the results block early; tracked
  in #330.
- **This is a live-path tool.** It returns news as of *now*, so using it during
  offline replay would show a historical episode present-day headlines.

#### `PolymarketTool`

Prediction-market odds for the traded asset, from Polymarket's public Gamma API —
free, no key, no account. It reuses the `MarketScanner` that backs
`PolymarketBetEnv`, so it inherits that client's fetching, retry and filtering
machinery — but sets its own budget and thresholds, which differ from the live
env's (see below).

```python
PolymarketTool(symbol="BTC/USD", top_n=5, min_volume_24h=10_000,
               min_liquidity=5_000, timeout=5.0)
```

The keyword defaults to the traded symbol via `symbol_to_query()` (`BTC` →
`"Bitcoin"`); the model can override it per call with
`<tool name="polymarket">{"query": "Fed"}</tool>`. Output is a ranked list of
questions with the YES probability and 24h volume:

```
Prediction markets for 'Bitcoin':
1. Will the price of Bitcoin be above $64,000 on August 10? — YES 97.0% · 24h vol $74,568
2. Bitcoin Up or Down on August 10? — YES 32.4% · 24h vol $131,842
```

A row of strike-based markets is effectively a market-implied price distribution,
which is information OHLCV cannot express. Probabilities are rendered to one
decimal on purpose: a market at 0.9962 near resolution must not read as `100%`.

Four things to keep in mind:

- **`min_volume_24h` / `min_liquidity` are a content filter, not just noise
  reduction.** Market questions are user-authored and land in the model's context
  verbatim, so low-volume markets are the injection surface. Every free-text
  field the tool renders — market questions *and* the model's own `query` — is
  collapsed to a single capped line, because a newline would otherwise let one
  field render as a second numbered row and fabricate a market. Lower these
  floors deliberately.
- **An empty result does not mean no markets exist.** `MarketScanner.scan()` logs
  and returns `[]` when the Gamma API is unreachable, so the tool cannot tell an
  outage from a genuinely empty result and deliberately says so rather than
  asserting an absence it never verified.
- **The tool blocks the collection step.** `_resolve_tools` resolves tool calls
  sequentially across the batch, so per-call latency is serialised onto the
  policy call inside `SyncDataCollector`/`env.rollout()`. The tool spends
  `timeout=5.0` over 2 attempts — roughly 11s worst case, against the scanner's
  default of roughly 48s. Budget that against **your** `execute_on` cadence, not
  against `PolymarketBetEnv`'s: with `max_tool_iters=3` the worst case is ~33s
  per conversation, which is comfortable on a `1Hour` step and most of the
  budget on a `1Min` one.
- **This is a live-path tool.** It returns markets that are open *now*, so using
  it during offline replay would show a historical episode present-day
  probabilities. Restrict it to live trading, as with `GoogleNewsTool`.

---

## See Also

- [Examples: LLM Actors](../examples/index.md#llm-actors) - Full example scripts
- [Examples: Rule-Based Actors](../examples/index.md#rule-based-actors) - Mean reversion examples
- [TorchRL Actors](https://pytorch.org/rl/reference/modules.html#actors) - Neural network policies
