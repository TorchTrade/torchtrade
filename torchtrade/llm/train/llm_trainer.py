"""LLMTrainer — GRPO fine-tuning of a local LLM trading actor on historical data.

Easy facade over the (validated-on-DGX-Spark) hybrid torchrl GRPO recipe: rollouts via
torchrl vLLMWrapper over a plain vLLM engine, group-relative advantage via MCAdvantage,
token loss via torchrl GRPOLoss over a LoRA/QLoRA TransformersWrapper, and per-step LoRA
adapter hot-swap into the vLLM engine. The trading env's OneStepTradingEnv is the reward
oracle (why OneStep for training / Sequential for eval: GRPO needs K samples of the SAME
bar -> a contextual bandit; see the training guide).

    LLMTrainer(df, config, model="unsloth/Qwen3-8B-Base-bnb-4bit", method="qlora",
               reward_fn=None, system_prompt=None, user_prompt_fn=None,
               loss="grpo", num_generations=2).train()
"""
from __future__ import annotations

import os
import shutil
import time

import torch

from torchtrade.actor.base_llm_actor import BaseLLMActor
from torchtrade.envs.offline import OneStepTradingEnv
from torchtrade.llm.train.losses import resolve_loss, validate_num_generations
from torchtrade.llm.train.models import (
    build_inference_policy,
    build_train_policy,
    save_lora_adapter,
    sync_weights_to_vllm,
)
from torchtrade.llm.train.trading_env import TradingRewardParser, make_trading_env


class _PromptBuilder(BaseLLMActor):
    """Minimal concrete BaseLLMActor used only to reuse the inference prompt builders
    (so training prompts == inference prompts). Never generates."""

    def generate_batch(self, system_prompt, user_prompts):  # pragma: no cover - not used
        raise NotImplementedError("_PromptBuilder is prompt-only")


class LLMTrainer:
    def __init__(self, df, config, model="unsloth/Qwen3-8B-Base-bnb-4bit", method="qlora",
                 reward_fn=None, system_prompt=None, user_prompt_fn=None,
                 feature_preprocessing_fn=None, feature_keys=None,
                 loss="grpo", loss_kwargs=None, num_generations=4, lr=1e-5,
                 max_steps=50, max_tokens=1024, max_model_len=4096, gpu_memory_utilization=0.5,
                 constrain_actions=False, enforce_eager=False, logprob_chunk_size=1024,
                 output_dir="./llm_grpo_out", use_wandb=True, wandb_project="torchtrade-grpo",
                 log_completions_every=5, n_completions_log=2):
        validate_num_generations(num_generations)
        if method not in ("full", "lora", "qlora"):
            raise ValueError(f"method must be 'full'|'lora'|'qlora', got {method!r}")

        self.df, self.config, self.model = df, config, model
        self.method, self.reward_fn = method, reward_fn
        self.system_prompt, self.user_prompt_fn = system_prompt, user_prompt_fn
        self.feature_preprocessing_fn = feature_preprocessing_fn
        self.feature_keys = feature_keys
        self.loss, self.loss_kwargs = loss, loss_kwargs
        self.K = num_generations
        self.lr, self.max_steps, self.max_tokens = lr, max_steps, max_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.constrain_actions = constrain_actions
        self.enforce_eager = enforce_eager
        self.logprob_chunk_size = logprob_chunk_size
        self.output_dir = output_dir
        self.use_wandb, self.wandb_project = use_wandb, wandb_project
        self.log_completions_every = log_completions_every
        self.n_completions_log = n_completions_log

    @staticmethod
    def _action_descriptions(env):
        """Human-readable description per action index, from the env's REAL action space.

        OneStepTradingEnv is SLTP-based: action 0 = hold, 1..n-1 = open a position with a
        (side, stop-loss, take-profit) bracket (env._action_tuple). `env.action_levels` is NOT
        the action space for this env (it is forced to [0.0]) — describe the SLTP brackets so
        the prompt matches what the model can actually choose."""
        tuples = getattr(env, "_action_tuple", None)
        if tuples is None:
            return None  # non-SLTP env: let BaseLLMActor describe via action_levels
        descs = []
        for i, (side, sl, tp) in enumerate(tuples):
            if side is None:
                descs.append(f"Action {i} -> hold / no position")
            elif sl is None or tp is None:  # e.g. ("close", None, None) when include_close_action=True
                descs.append(f"Action {i} -> {side} current position")
            else:
                descs.append(f"Action {i} -> open {side}: stop-loss {sl:+.1%}, take-profit {tp:+.1%}")
        return descs

    @staticmethod
    def _build_action_regex(num_actions):
        r"""Guided-decoding regex: reason inside `<think>...</think>`, then a valid
        `<answer>N</answer>` (N in [0, num_actions)), then STOP.

        The think body is `[^<]{40,600}` — bounded and delimiter-free — NOT `[\s\S]*?`. That earlier
        form matched ANY character (incl. `<`, `/think`, `answer`), so in xgrammar's DFA the think
        body could swallow the closing tags as ordinary text: the model was never forced to close or
        answer. Measured symptoms: empty `<think></think>`, rambling to the token cap with NO answer,
        and free text AFTER `</answer>` (a still-live "in think body" DFA path). `[^<]` forbids `<`
        in the body so the only way to produce `<` is to start `</think>`; `{40,600}` makes min-40
        kill empty think and max-600 force `</think>` (then the answer) well before a typical
        max_tokens cap. Reasoning can't contain a literal `<` (xgrammar rejects the lookahead that
        would allow it) — the model uses `>`/"above"/"below" fine.

        The inter-tag gap is `\s{0,2}`, NOT `\s*`. An unbounded `\s*` is an escape hatch: a
        reasoning-hungry model (any *thinking* model — Qwen3.5, Qwen3-*-Thinking), once the char
        bound forces it to close `<think>` while it still "wants" to reason, dumps whitespace into
        the gap until the token cap and never emits `<answer>`. Measured on Qwen3-4B-Thinking: `\s*`
        gave 16-35% no-answer FLAT across every (think-bound x max_tokens) cell (raising max_tokens
        did NOT help — it just fed the dump); `\s{0,2}` took it to 0% (the base Qwen3-8B, an English
        reasoner, was already 0% either way). 0-2 whitespace covers the natural `</think>\n<answer>`
        without leaving room to escape."""
        indices = "|".join(str(i) for i in range(num_actions))
        return r"<think>[^<]{40,600}</think>\s{0,2}<answer>(" + indices + r")</answer>"

    def _build_prompt_actor(self, env):
        num_actions = env.action_spec.n
        return _PromptBuilder(
            market_data_keys=env.market_data_keys,
            account_state_labels=env.account_state,
            action_levels=list(range(num_actions)),  # index space; descriptions carry meaning
            action_descriptions=self._action_descriptions(env),
            symbol=self.config.symbol,
            execute_on=self.config.execute_on,
            feature_keys=self.feature_keys,
            system_prompt=self.system_prompt,
            user_prompt_fn=self.user_prompt_fn,
        )

    def _render_group_prompts(self, score_env, pb):
        """Pick `max_steps` bars, render each bar's prompt, and repeat it K times so each
        consecutive K-block is one bar's GRPO group."""
        prompts, bar_indices = [], []
        sysp = pb._resolve_system_prompt()
        max_bar = score_env.sampler._max_organic_start_idx() + 1
        step_bars = torch.linspace(1, max_bar, self.max_steps).long().tolist()
        for b in step_bars:
            td = score_env.obs_at(int(b))
            user = pb._build_user_prompt(td)
            prompts += [user] * self.K
            bar_indices += [int(b)] * self.K
        return sysp, prompts, bar_indices

    def train(self):
        from transformers import AutoTokenizer
        from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
        from torchrl.objectives.llm import MCAdvantage

        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        device = "cuda"

        os.makedirs(self.output_dir, exist_ok=True)
        # reward oracle (never draws random bars during scoring — score()/obs_at seek)
        score_env = OneStepTradingEnv(df=self.df, config=self.config,
                                      feature_preprocessing_fn=self.feature_preprocessing_fn)
        num_actions = score_env.action_spec.n
        pb = self._build_prompt_actor(score_env)

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.padding_side = "left"

        sysp, prompts, bar_indices = self._render_group_prompts(score_env, pb)
        env = make_trading_env(score_env, tokenizer, prompts, bar_indices, sysp, num_actions,
                               self.K, reward_fn=self.reward_fn)

        # Guided decoding: constrain every completion to a valid <answer>N</answer> so it always
        # parses to a real action (base/small models emit the format unreliably otherwise), which
        # is what feeds GRPO within-group reward variance.
        action_regex = self._build_action_regex(num_actions) if self.constrain_actions else None
        # LoRA/QLoRA: vLLM loads the frozen base once with LoRA support, and each step we
        # hot-swap only the trained adapter (cheap); "full" re-syncs the whole model.
        is_peft = self.method in ("lora", "qlora")
        engine, infer = build_inference_policy(self.model, tokenizer,
                                               gpu_memory_utilization=self.gpu_memory_utilization,
                                               max_model_len=self.max_model_len,
                                               max_tokens=self.max_tokens, action_regex=action_regex,
                                               enable_lora=is_peft, enforce_eager=self.enforce_eager)
        hf, train_policy = build_train_policy(self.model, tokenizer, method=self.method,
                                              device=device,
                                              logprob_chunk_size=self.logprob_chunk_size)

        # Train the ANSWER tokens only: masking_strategy="rlhf" scores the assistant/completion
        # tokens (the <think>.../<answer>N</answer> the model generated), not the prompt. Stock
        # GRPOLoss defaults to "sft"; override unless the user set it explicitly in loss_kwargs.
        loss_kwargs = dict(self.loss_kwargs or {})
        if self.loss == "grpo":
            loss_kwargs.setdefault("masking_strategy", "rlhf")
        loss_fn = resolve_loss(self.loss, train_policy, loss_kwargs)
        # Buffer holds exactly one group (K completions of one bar), matching torchrl's canonical
        # grpo-sync recipe: sample(K) then returns the just-collected group with no cross-round
        # mixing — a larger buffer + SamplerWithoutReplacement would blend up to several prior
        # (staler) groups into each minibatch.
        rb = ReplayBuffer(storage=LazyStackStorage(self.K),
                          sampler=SamplerWithoutReplacement(),
                          transform=MCAdvantage(grpo_size=self.K))
        opt = torch.optim.Adam([p for p in hf.parameters() if p.requires_grad], lr=self.lr)

        logger = None
        if self.use_wandb:
            try:  # default-on: never let logging setup (e.g. missing wandb auth) kill training
                import wandb
                logger = wandb.init(project=self.wandb_project, config={
                    "model": self.model, "method": self.method, "num_generations": self.K,
                    "lr": self.lr, "max_steps": self.max_steps})
            except Exception as e:
                print(f"[LLMTrainer] wandb disabled ({e}); logging to stdout only", flush=True)

        # Sample completions logged to a growing wandb.Table so you can watch WHAT the policy
        # generates as training progresses (the best intuition for whether it is learning the
        # <think>/<answer> format + sensible actions). Accumulated + re-logged each log step;
        # capped so the table stays light.
        from torchtrade.actor.parsers import extract_action
        completion_cols = ["step", "completion", "action", "reward"]
        completion_rows = []
        max_table_rows = 100  # keep the re-logged wandb table light over long runs

        for step in range(self.max_steps):
            t0 = time.time()
            data = env.rollout(1, infer)
            rollout_dt = time.time() - t0  # generation is the throughput number that matters
            rewards = data.get(("next", "reward")).flatten().tolist()
            # capture a few of this step's completions (the last turn's content) for the wandb table
            log_completions = (logger is not None and self.log_completions_every
                               and step % self.log_completions_every == 0)
            if log_completions:
                # reshape(-1) drops the rollout's time dim so each history is a single scalar
                # conversation (matching the (K,)-batched td the reward parser reads); on the raw
                # (K,1) data, h[-1] would grab the whole conversation, not the last turn.
                histories = list(data.reshape(-1).get(("history", "full")))
                for h, r in list(zip(histories, rewards))[:self.n_completions_log]:
                    text = str(TradingRewardParser._response_text(h))
                    completion_rows.append([step, text, extract_action(text, num_actions), r])
            # MCAdvantage computes the group-relative advantage at extend() time (grouped by the
            # shared prompt), so it is baked in before sampling; with a K-sized buffer, sample(K)
            # returns exactly this step's group.
            rb.extend(data.reshape(-1))
            batch = rb.sample(self.K).to(device)
            # generation throughput: assistant mask marks the model-generated (completion) tokens
            # across all K completions; tokens/s = those tokens / the rollout wall-time.
            gen_tokens = int(batch.get(("masks", "all_assistant_mask"), as_padded_tensor=True,
                                       padding_side="left", padding_value=0).sum())
            tok_s = gen_tokens / rollout_dt if rollout_dt > 0 else 0.0
            loss = loss_fn(batch)
            # Sum ALL loss_* terms — GRPOLoss writes loss_objective, loss_entropy (entropy bonus),
            # and any KL penalty as SEPARATE keys; cherry-picking loss_objective would silently
            # drop the entropy/KL regularization from backward. Fall back to a plain-tensor loss.
            loss_keys = [k for k in loss.keys() if isinstance(k, str) and k.startswith("loss_")]
            loss_val = sum(loss.get(k).mean() for k in loss_keys) if loss_keys \
                else loss.mean(reduce=True)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            if is_peft:
                adapters_dir = os.path.join(self.output_dir, "adapters")
                infer._lora_request = save_lora_adapter(hf, adapters_dir, step)
                # bound disk over long runs: vLLM only needs the current adapter (loaded lazily on
                # the next rollout), so drop stale ones — otherwise ~tens of MB/step accumulates.
                stale = os.path.join(adapters_dir, f"step_{step - 3}")
                if os.path.isdir(stale):
                    shutil.rmtree(stale, ignore_errors=True)
                synced = f"adapter@{infer._lora_request.lora_int_id}"
            else:
                synced = sync_weights_to_vllm(engine, hf, path=os.path.join(self.output_dir, "_full.pt"))
            mean_r = sum(rewards) / len(rewards)
            # advantage = the group-relative signal MCAdvantage baked in; log its magnitude
            # (mean |adv|) — ~0 on a degenerate all-agree group, positive when the group disagrees.
            adv = batch.get("advantage", None)
            adv_mag = float(adv.float().abs().mean()) if adv is not None else float("nan")
            dt = time.time() - t0
            print(f"[LLMTrainer] step {step}: reward={mean_r:.4f} advantage={adv_mag:.4f} "
                  f"loss={float(loss_val):.4f} synced={synced} dt={dt:.1f}s "
                  f"rollout={rollout_dt:.1f}s tok/s={tok_s:.1f} gen_tokens={gen_tokens}", flush=True)
            if logger is not None:
                metrics = {"step": step, "reward": mean_r, "advantage": adv_mag,
                           "loss": float(loss_val), "step_time_s": dt,
                           "rollout_time_s": rollout_dt, "tokens_per_s": tok_s,
                           "gen_tokens": gen_tokens}
                if log_completions:
                    import wandb  # Tables are immutable once logged: re-build from accumulated rows
                    metrics["completions"] = wandb.Table(columns=completion_cols,
                                                         data=completion_rows[-max_table_rows:])
                logger.log(metrics)

        adapter_dir = os.path.join(self.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        hf.save_pretrained(adapter_dir)
        if logger is not None:
            logger.finish()
        print(f"[LLMTrainer] done — adapter saved to {adapter_dir}", flush=True)
        return adapter_dir
