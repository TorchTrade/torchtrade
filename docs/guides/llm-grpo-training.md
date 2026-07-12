# Fine-tuning an LLM trading actor with GRPO (`LLMTrainer`)

`LLMTrainer` fine-tunes a local LLM to make better `<answer>N</answer>` trading decisions with
**GRPO** (group-relative policy optimization) on your historical data. The trading environment
provides the reward; you pass a dataset and pick what to train.

```python
from torchtrade.envs.offline import OneStepTradingEnvConfig
from torchtrade.llm.train import LLMTrainer

config = OneStepTradingEnvConfig(symbol="BTC/USD", time_frames=["1Day"], window_sizes=[30],
                                 execute_on="1Day", action_levels=[-1, 0, 1], random_start=False)

adapter = LLMTrainer(
    df=df, config=config,
    feature_preprocessing_fn=my_features,  # what features to COMPUTE (features_* columns)
    feature_keys=["features_return", "close"],  # which of them the LLM SEES in the prompt
    model="unsloth/Qwen3-8B-Base-bnb-4bit",  # default; one 4-bit ckpt for rollouts + QLoRA
    method="qlora",          # "full" | "lora" | "qlora"
    num_generations=8,       # K completions per bar (the GRPO group); K is cheap now (see Memory)
    logprob_chunk_size=1024, # chunk size for the memory-bounded log-prob/entropy (see Memory)
    constrain_actions=True,  # guided decoding -> every completion is a parseable action (recommended)
    reward_fn=None,          # default: OneStepTradingEnv score; override to customize
    system_prompt=None,      # default: the actor's system prompt; override to customize
    user_prompt_fn=None,     # default: the actor's user prompt; the "pre-prompt" override
    loss="grpo",             # registry name or a factory f(actor) -> LossModule
    use_wandb=True,          # logs reward/advantage/loss/tokens_per_s + a sample-completions table
).train()
# -> load `adapter` into LocalLLMActor for eval / live inference.
```

The snippet above is the minimal call (see `examples/llm/train/grpo_finetune.py`). Everything
else is optional and follows the same "pass to override, omit for our default" pattern.

## Customizing

### Features — what the LLM sees

`feature_preprocessing_fn` receives the resampled OHLCV frame and returns `features_*` columns;
`feature_keys` selects which columns are rendered into the prompt. These are the *same* hooks
`LocalLLMActor` uses at inference, so **your training prompt matches your inference prompt**. Omit
both to fall back to the raw market-data keys.

```python
def my_features(df):
    df = df.copy().reset_index(drop=False)
    df["features_return"] = df["close"].pct_change().fillna(0.0)
    df["features_sma_ratio"] = (df["close"] / df["close"].rolling(10).mean()).fillna(1.0)
    return df

LLMTrainer(df=df, config=config,
           feature_preprocessing_fn=my_features,           # what to COMPUTE
           feature_keys=["features_return", "features_sma_ratio", "close"])  # what the LLM SEES
```

### Reward — how a decision is scored

By default each decision is scored by `OneStepTradingEnv.score(bar_index, action)` (the realized
return of rolling that action to its SL/TP/horizon). Pass `reward_fn(action, bar_index, env) -> float`
to score it your way — e.g. penalize holding to encourage the model to take positions:

```python
def my_reward(action, bar_index, env):
    r = env.score(bar_index, action)
    return r - 0.001 if action == 0 else r    # small penalty for "hold"

LLMTrainer(df=df, config=config, reward_fn=my_reward)
```

### Prompts — system prompt and pre-prompt

```python
LLMTrainer(df=df, config=config,
           system_prompt="You are a disciplined BTC swing trader...",  # override the actor default
           user_prompt_fn=my_user_prompt_fn)   # the "pre-prompt" builder (obs td -> str)
```

### Loss and training method

```python
LLMTrainer(df=df, config=config,
           method="qlora",         # "full" | "lora" | "qlora"
           loss="grpo",            # registry name, or a factory f(actor) -> LossModule
           loss_kwargs={...},      # forwarded to the loss constructor
           num_generations=8)      # GRPO group size (K completions per bar)
```

## Timeframe — train on daily (or coarser) bars

An LLM reasons over higher-level market structure; a decision over 1-minute or 1-hour noise gives it
almost nothing to latch onto, and GRPO groups collapse to "all hold" (no within-group reward
variance → no gradient). Set `time_frames`/`execute_on` to `"1Day"` (the sampler resamples your
source data up) with a window of ~20–60 bars. The timeframe is entirely your `config` — the trainer
passes it through untouched.

## Why `OneStepTradingEnv` for training and `SequentialTradingEnv` for eval

GRPO's advantage is **group-relative**: for one state it samples K completions, gets K rewards,
and uses `advantage_i = (r_i − mean) / std`. That requires **K independent samples of the *same*
state**, each scored by one scalar — a contextual bandit. `OneStepTradingEnv` is exactly that
(reset → a bar, `step(action)` → one scalar reward that rolls the position to its SL/TP/horizon).
`SequentialTradingEnv` would break the group: after the first action the K rollouts diverge into
different states, so there is no shared state to group on.

So training uses the one-step form (via the deterministic reward oracle
`OneStepTradingEnv.score(bar_index, action)`), and **evaluation** backtests the trained actor on
`SequentialTradingEnv` — the honest measure of real sequential trading performance. A different
algorithm that needs a trajectory (PPO/PG with a value head) would use `SequentialTradingEnv`
instead — that is a separate future recipe, not a `loss` swap.

## Guided decoding (constrained `<think>`/`<answer>` output)

GRPO only learns from **within-group reward variance**. If the model's completions don't emit a
valid `<answer>N</answer>`, the parser falls back to action 0 (hold) → every group is all-hold →
zero reward variance → no gradient. Models comply with the format unreliably (Qwen2.5-0.5B ~2% of
the time), so set `constrain_actions=True` (recommended; the parameter defaults to `False`) to apply
vLLM **guided decoding**: a regex forces every completion to `<think>…</think><answer>N</answer>`
with an in-range index → parseable → the model's genuine action variety becomes real learning signal.

The regex is `<think>[^<]{40,600}</think>\s*<answer>(0|…)</answer>`. The bounded, `<`-free think body
is deliberate: a permissive `[\s\S]*?` body is **not actually enforced** by the decoding FSM (it can
absorb the closing tags as free text), which lets the model emit an empty `<think></think>`, ramble
past `max_tokens` with no answer, or add text after `</answer>`. Forbidding `<` forces the only way to
produce `<` to be `</think>`, and `{40,600}` makes a minimum of real reasoning while forcing the close
(then the answer) before the token cap. Keep `max_tokens` at roughly 2× the think token budget so the
forced answer always fits. Reasoning can't contain a literal `<` (the decoder rejects the lookahead
that would allow it) — the model uses `>` / "above" / "below" naturally. Set `constrain_actions=False`
only with a strong instruction-tuned model where you want fully free-form reasoning and accept the
occasional unparseable completion (which scores as hold).

## The stack (hybrid torchrl-native)

- Rollouts: torchrl `vLLMWrapper` over a plain `vllm.LLM` (K generations per bar), guided-decoded.
- Advantage: torchrl `MCAdvantage` (group-relative over the K-group).
- Loss: stock torchrl `GRPOLoss` over a LoRA/QLoRA `ChunkedTransformersWrapper` (the chunked
  log-prob/entropy from the Memory section, plugged into the wrapper's dist hook so the loss is
  unmodified). `masking_strategy="rlhf"` scores only the **answer/assistant tokens** the model
  generated, not the prompt.
- Weight sync (LoRA/QLoRA): the vLLM engine loads the frozen 4-bit base **once**; each step the
  trainer saves just the LoRA adapter (~tens of MB) and hot-swaps it via a fresh `LoRARequest`. The
  base never changes, so this replaces re-pushing the whole ~16 GB base every step — far less memory
  and a ~300× smaller per-step sync. (`method="full"` falls back to a merged-weight `collective_rpc`
  sync; run with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.)

> On DGX Spark (GB10/sm_121) this hybrid path is used because torchrl's `AsyncVLLM` + NCCL sync
> does not import against the vLLM build that runs on that GPU. It still uses torchrl's
> `vLLMWrapper` / `TransformersWrapper` / `GRPOLoss` / `MCAdvantage` — only the engine and the
> sync mechanism differ.

## Memory — why `num_generations` (K) is a free knob

Stock GRPO builds a full `[K × seq_len × vocab]` logits tensor to get per-token log-probs, and with
a large vocabulary (e.g. Qwen3.5's 248k) that term explodes as K grows — training used to OOM by
K≈4. `LLMTrainer` avoids it with two things, so peak memory grows only gently with K:

- **Chunked log-probs (`ChunkedTransformersWrapper`).** Instead of materializing all logits, it
  computes the log-prob and entropy from the model's hidden states, chunked over the sequence, so
  peak vocab memory is `O(logprob_chunk_size × vocab)` — independent of K and sequence length. It's
  a lazy distribution plugged into the wrapper's own dist hook, so stock `GRPOLoss` is used
  unchanged. Tune `logprob_chunk_size` (default 1024): smaller = less memory, slightly slower.
- **Gradient checkpointing that is actually active.** The trainer enables checkpointing on the
  *final* PEFT model and calls `train()` so every layer's checkpoint gate is live (enabling it before
  the PEFT wrap, or leaving the model with children in eval, silently makes it inert — the flags lie).

Measured on a single GB10 (Qwen3.5-4B QLoRA, seq 512): K=2 → 9.4 GB, K=8 → 22 GB, K=16 → 38 GB,
K=24 → 55 GB — ~2 GB/K, versus ~12 GB/K with the naive path. So raise K for a stronger group signal
rather than fighting OOMs. If you still need to trim: lower `logprob_chunk_size`, shorten the prompt
(fewer/smaller feature columns, smaller window), lower `max_tokens`, and keep `gpu_memory_utilization`
modest (the 4-bit base needs little).

## Logging & observability

With `use_wandb=True` (default) each step logs `reward`, `advantage` (mean |group-relative
advantage| — ~0 when the group agrees, positive when it disagrees), `loss`, `step_time_s`,
`rollout_time_s`, `tokens_per_s` (generation throughput), and `gen_tokens`. Every
`log_completions_every` steps (default 5) it also appends `n_completions_log` (default 2) sample
completions to a growing wandb **table** — `step` / `completion` / parsed `action` / `reward` — so
you can watch the actual `<think>…</think><answer>N</answer>` text evolve as training progresses (the
best intuition for whether the model is learning the format and taking sensible actions). Everything
is also printed to stdout, and logging never blocks training (a missing wandb login degrades to
stdout only).

## Requirements

`pip install -e ".[llm]"` and a CUDA GPU. See `examples/llm/train/grpo_finetune.py` for a full
runnable example.
