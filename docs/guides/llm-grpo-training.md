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
    num_generations=2,       # K completions per bar (the GRPO group); higher K = more memory
    constrain_actions=True,  # guided decoding -> every completion is a parseable action
    reward_fn=None,          # default: OneStepTradingEnv score; override to customize
    system_prompt=None,      # default: the actor's system prompt; override to customize
    user_prompt_fn=None,     # default: the actor's user prompt; the "pre-prompt" override
    loss="grpo",             # registry name or a factory f(actor) -> LossModule
    use_wandb=True,          # logs loss / mean_reward per step
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

## Guided decoding (why a bigger model + constrained output)

GRPO only learns from **within-group reward variance**. If the model's completions don't emit a
valid `<answer>N</answer>`, the parser falls back to action 0 (hold) → every group is all-hold →
zero reward variance → no gradient. Small models comply poorly with the format (Qwen2.5-0.5B ~2% of
the time), so `constrain_actions=True` (the default) applies vLLM **guided decoding**: a regex forces
every completion to a valid action index → 100% parseable → the model's genuine action variety
becomes real learning signal. The default model is a capable `unsloth/Qwen3-8B-Base-bnb-4bit`; set
`constrain_actions=False` only if you use a strong instruction-tuned model and want free-form
reasoning tokens.

## The stack (hybrid torchrl-native)

- Rollouts: torchrl `vLLMWrapper` over a plain `vllm.LLM` (K generations per bar), guided-decoded.
- Advantage: torchrl `MCAdvantage` (group-relative over the K-group).
- Loss: torchrl `GRPOLoss` over a LoRA/QLoRA `TransformersWrapper`.
- Weight sync (LoRA/QLoRA): the vLLM engine loads the frozen 4-bit base **once**; each step the
  trainer saves just the LoRA adapter (~tens of MB) and hot-swaps it via a fresh `LoRARequest`. The
  base never changes, so this replaces re-pushing the whole ~16 GB base every step — far less memory
  and a ~300× smaller per-step sync. (`method="full"` falls back to a merged-weight `collective_rpc`
  sync; run with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.)

> On DGX Spark (GB10/sm_121) this hybrid path is used because torchrl's `AsyncVLLM` + NCCL sync
> does not import against the vLLM build that runs on that GPU. It still uses torchrl's
> `vLLMWrapper` / `TransformersWrapper` / `GRPOLoss` / `MCAdvantage` — only the engine and the
> sync mechanism differ.

## Memory

The GRPO loss materializes a `[K × seq_len × vocab]` logits tensor for the backward pass, and with
Qwen's ~152k vocab that dominates memory (independent of model size). If you OOM: lower
`num_generations` (K), shorten the prompt (fewer/smaller feature columns, smaller window), and keep
`gpu_memory_utilization` modest (the 4-bit base needs little). QLoRA runs gradient checkpointing
automatically.

## Requirements

`pip install -e ".[llm]"` and a CUDA GPU. See `examples/llm/train/grpo_finetune.py` for a full
runnable example.
