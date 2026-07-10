"""LLMTrainer — GRPO fine-tuning of a local LLM trading actor on historical data.

Easy facade over the (validated-on-DGX-Spark) hybrid torchrl GRPO recipe: rollouts via
torchrl vLLMWrapper over a plain vLLM engine, group-relative advantage via MCAdvantage,
token loss via torchrl GRPOLoss over a LoRA/QLoRA TransformersWrapper, and merged-weight
sync back to the vLLM engine each step. The trading env's OneStepTradingEnv is the reward
oracle (why OneStep for training / Sequential for eval: GRPO needs K samples of the SAME
bar -> a contextual bandit; see the training guide).

    LLMTrainer(df, config, model="Qwen/Qwen2.5-0.5B-Instruct", method="qlora",
               reward_fn=None, system_prompt=None, user_prompt_fn=None,
               loss="grpo", num_generations=8).train()
"""
from __future__ import annotations

import os

import torch

from torchtrade.actor.base_llm_actor import BaseLLMActor
from torchtrade.envs.offline import OneStepTradingEnv
from torchtrade.train.losses import resolve_loss, validate_num_generations
from torchtrade.train.models import build_inference_policy, build_train_policy, sync_weights_to_vllm
from torchtrade.train.trading_env import make_trading_env


class _PromptBuilder(BaseLLMActor):
    """Minimal concrete BaseLLMActor used only to reuse the inference prompt builders
    (so training prompts == inference prompts). Never generates."""

    def generate_batch(self, system_prompt, user_prompts):  # pragma: no cover - not used
        raise NotImplementedError("_PromptBuilder is prompt-only")


class LLMTrainer:
    def __init__(self, df, config, model="Qwen/Qwen2.5-0.5B-Instruct", method="qlora",
                 reward_fn=None, system_prompt=None, user_prompt_fn=None,
                 feature_preprocessing_fn=None, feature_keys=None,
                 loss="grpo", loss_kwargs=None, num_generations=8, lr=1e-5,
                 max_steps=50, max_tokens=256, gpu_memory_utilization=0.3,
                 output_dir="./llm_grpo_out", use_wandb=False, wandb_project="torchtrade-grpo"):
        validate_num_generations(num_generations)
        # build_peft_config validates `method` early (raises on unknown method)
        from torchtrade.train.peft_config import build_peft_config
        build_peft_config(method)

        self.df, self.config, self.model = df, config, model
        self.method, self.reward_fn = method, reward_fn
        self.system_prompt, self.user_prompt_fn = system_prompt, user_prompt_fn
        self.feature_preprocessing_fn = feature_preprocessing_fn
        self.feature_keys = feature_keys
        self.loss, self.loss_kwargs = loss, loss_kwargs
        self.K = num_generations
        self.lr, self.max_steps, self.max_tokens = lr, max_steps, max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.output_dir = output_dir
        self.use_wandb, self.wandb_project = use_wandb, wandb_project

    def _build_prompt_actor(self, env):
        return _PromptBuilder(
            market_data_keys=env.market_data_keys,
            account_state_labels=env.account_state,
            action_levels=env.action_levels,
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

        # reward oracle (never draws random bars during scoring — score()/obs_at seek)
        os.makedirs(self.output_dir, exist_ok=True)
        score_env = OneStepTradingEnv(df=self.df, config=self.config,
                                      feature_preprocessing_fn=self.feature_preprocessing_fn)
        num_actions = len(score_env.action_levels)
        pb = self._build_prompt_actor(score_env)

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.padding_side = "left"

        sysp, prompts, bar_indices = self._render_group_prompts(score_env, pb)
        env = make_trading_env(score_env, tokenizer, prompts, bar_indices, sysp, num_actions,
                               self.K, reward_fn=self.reward_fn)

        engine, infer = build_inference_policy(self.model, tokenizer,
                                               gpu_memory_utilization=self.gpu_memory_utilization,
                                               max_tokens=self.max_tokens)
        hf, train_policy = build_train_policy(self.model, tokenizer, method=self.method, device=device)

        loss_fn = resolve_loss(self.loss, train_policy, self.loss_kwargs)
        loss_fn.set_keys(sample_log_prob=("log_probs", "full"))
        rb = ReplayBuffer(storage=LazyStackStorage(self.K * 4),
                          sampler=SamplerWithoutReplacement(),
                          transform=MCAdvantage(grpo_size=self.K))
        opt = torch.optim.Adam([p for p in hf.parameters() if p.requires_grad], lr=self.lr)

        logger = None
        if self.use_wandb:
            import wandb
            logger = wandb.init(project=self.wandb_project, config={
                "model": self.model, "method": self.method, "num_generations": self.K,
                "lr": self.lr, "max_steps": self.max_steps})

        for step in range(self.max_steps):
            data = env.rollout(1, infer)
            rewards = data.get(("next", "reward")).flatten().tolist()
            # MCAdvantage computes the group-relative advantage at extend() time (grouped by
            # the shared prompt), so it is baked in before sampling; rb.sample(K) is just a
            # minibatch draw (K here == group size only by convenience).
            rb.extend(data.reshape(-1))
            batch = rb.sample(self.K).to(device)
            loss = loss_fn(batch)
            loss_val = loss.get("loss_objective").mean() if "loss_objective" in loss.keys() \
                else loss.mean(reduce=True)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            n = sync_weights_to_vllm(engine, hf, path=os.path.join(self.output_dir, "_merged.pt"))
            mean_r = sum(rewards) / len(rewards)
            print(f"[LLMTrainer] step {step}: mean_reward={mean_r:.4f} loss={float(loss_val):.4f} synced={n}",
                  flush=True)
            if logger is not None:
                logger.log({"loss": float(loss_val), "mean_reward": mean_r, "step": step})

        adapter_dir = os.path.join(self.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        hf.save_pretrained(adapter_dir)
        if logger is not None:
            logger.finish()
        print(f"[LLMTrainer] done — adapter saved to {adapter_dir}", flush=True)
        return adapter_dir
