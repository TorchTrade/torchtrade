# Fine-tuning an LLM trading actor with GRPO (`LLMTrainer`)

`LLMTrainer` fine-tunes a local LLM to make better `<answer>N</answer>` trading decisions with
**GRPO** (group-relative policy optimization) on your historical data. The trading environment
provides the reward; you pass a dataset and pick what to train.

```python
from torchtrade.envs.offline import OneStepTradingEnvConfig
from torchtrade.train import LLMTrainer

config = OneStepTradingEnvConfig(symbol="BTC/USD", time_frames=["1Hour"], window_sizes=[48],
                                 execute_on="1Hour", action_levels=[-1, 0, 1], random_start=False)

adapter = LLMTrainer(
    df=df, config=config, feature_preprocessing_fn=my_features,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    method="qlora",          # "full" | "lora" | "qlora"
    num_generations=8,       # K completions per bar (the GRPO group)
    reward_fn=None,          # default: OneStepTradingEnv score; override to customize
    system_prompt=None,      # default: the actor's system prompt; override to customize
    user_prompt_fn=None,     # default: the actor's user prompt; the "pre-prompt" override
    loss="grpo",             # registry name or a factory f(actor) -> LossModule
    use_wandb=True,          # logs loss / mean_reward per step
).train()
# -> load `adapter` into LocalLLMActor for eval / live inference.
```

Everything customizable has a default: **reward, system prompt, user prompt, and loss** all
follow the same "pass to override, omit for our default" pattern.

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

## The stack (hybrid torchrl-native)

- Rollouts: torchrl `vLLMWrapper` over a plain `vllm.LLM` (K generations per bar).
- Advantage: torchrl `MCAdvantage` (group-relative over the K-group).
- Loss: torchrl `GRPOLoss` over a LoRA/QLoRA `TransformersWrapper`.
- Weight sync: after each step the merged LoRA weights are pushed into the vLLM engine via
  `collective_rpc` (run with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`).

> On DGX Spark (GB10/sm_121) this hybrid path is used because torchrl's `AsyncVLLM` + NCCL sync
> does not import against the vLLM build that runs on that GPU. It still uses torchrl's
> `vLLMWrapper` / `TransformersWrapper` / `GRPOLoss` / `MCAdvantage` — only the engine and the
> sync mechanism differ.

## Requirements

`pip install -e ".[llm]"` and a CUDA GPU. See `examples/llm/train/grpo_finetune.py` for a full
runnable example.
