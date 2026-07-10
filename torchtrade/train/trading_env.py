"""Trading GRPO environment: a torchrl ChatEnv fed by trading-bar prompts, with a reward
Transform that scores the LLM's <answer> via OneStepTradingEnv.score.

This is the torchrl-side of the LLMTrainer: it turns each bar into a chat prompt (the LLM's
observation), lets the policy generate, then scores the completion's parsed action against
that bar. Validated end-to-end on the DGX Spark (see the LLMTrainer for the training loop).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import Composite, Unbounded
from torchrl.envs import StepCounter, Transform

from torchtrade.train.reward import TradingReward


class TradingRewardParser(Transform):
    """Reward Transform: parse the assistant completion's <answer>N</answer> and score it at
    its bar via `TradingReward` (the OneStepTradingEnv.score oracle, or a custom `reward_fn`).

    Reads the completion from the INPUT tensordict's ("history", "full") (the last turn's
    content) — in a one-step env the completion lives there, not under "next".
    """

    def __init__(self, score_env, num_actions: int, reward_fn=None):
        super().__init__(in_keys=[("history", "full"), "bar_index"], out_keys=["reward"])
        self.reward = TradingReward(score_env, num_actions, reward_fn)

    @staticmethod
    def _response_text(history_item) -> str:
        try:
            return history_item[-1].content  # last turn = assistant response
        except (AttributeError, IndexError, TypeError):
            return str(history_item)

    def _step(self, tensordict, next_tensordict):
        history = tensordict.get(("history", "full"))
        bars = tensordict.get("bar_index")
        bars = bars.tolist() if hasattr(bars, "tolist") else [bars]
        rewards = [self.reward(self._response_text(h), int(b)) for h, b in zip(list(history), bars)]
        next_tensordict.set("reward", torch.tensor(rewards).reshape(next_tensordict.batch_size + (1, 1)))
        return next_tensordict

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        reward_spec.update(Composite(reward=Unbounded(reward_spec.shape + (1, 1))))
        return reward_spec


def make_trading_env(score_env, tokenizer, prompts, bar_indices, system_prompt, num_actions,
                     group_size, reward_fn=None):
    """Build the trading ChatEnv: a dataloader of (prompt, bar_index) rows -> ChatEnv (history
    mode) -> StepCounter(1) -> TradingRewardParser. `group_size` prompts are consumed per batch
    to form the GRPO group (K completions of the same bar when `prompts` repeats a bar K times).
    """
    from torchrl.envs.llm import ChatEnv  # lazy: pulls the torchrl LLM/vLLM stack

    rows = [{"query": q, "bar_index": int(b)} for q, b in zip(prompts, bar_indices)]

    def collate(batch):
        return TensorDict(
            {"query": [r["query"] for r in batch],
             "bar_index": torch.tensor([r["bar_index"] for r in batch])},
            batch_size=[len(batch)],
        )

    dataloader = DataLoader(rows, batch_size=group_size, collate_fn=collate, shuffle=False)
    env = ChatEnv.from_dataloader(
        dataloader=dataloader, data_key="query", input_mode="history",
        system_prompt=system_prompt, tokenizer=tokenizer, batch_size=(group_size,),
    )
    return env.append_transform(StepCounter(max_steps=1)).append_transform(
        TradingRewardParser(score_env, num_actions, reward_fn)
    )
