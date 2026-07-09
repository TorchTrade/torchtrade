"""Tests for BaseLLMActor's batched interface (stub backend, no model/GPU)."""

import pytest
import torch
from tensordict import TensorDict

from torchtrade.actor.base_llm_actor import BaseLLMActor

MARKET_KEY = "market_data_1Hour_48"
ACCOUNT_LABELS = ["exposure_pct", "position_direction", "unrealized_pnlpct",
                  "holding_time", "leverage", "distance_to_liquidation"]
ACTION_LEVELS = [-1.0, 0.0, 1.0]


class StubActor(BaseLLMActor):
    """Records the prompts it is asked to generate for, returns canned answers.

    `answers` is a list of ints; response i is f"<answer>{answers[i]}</answer>".
    If `answers` is shorter than the batch, it cycles.
    """

    def __init__(self, answers, **kwargs):
        super().__init__(**kwargs)
        self._answers = answers
        self.last_system_prompt = None
        self.last_user_prompts = None

    def generate_batch(self, system_prompt, user_prompts):
        self.last_system_prompt = system_prompt
        self.last_user_prompts = list(user_prompts)
        return [f"<answer>{self._answers[i % len(self._answers)]}</answer>"
                for i in range(len(user_prompts))]


def _actor(answers):
    return StubActor(
        answers=answers,
        market_data_keys=[MARKET_KEY],
        account_state_labels=ACCOUNT_LABELS,
        action_levels=ACTION_LEVELS,
        symbol="BTC/USD",
        execute_on="1Hour",
    )


def _unbatched_td():
    return TensorDict({
        MARKET_KEY: torch.randn(1, 48, 5),
        "account_state": torch.tensor([[0.5, 1.0, 0.02, 5.0, 1.0, 1.0]]),
    }, batch_size=[])


def _batched_td(n):
    return TensorDict({
        MARKET_KEY: torch.randn(n, 48, 5),
        "account_state": torch.randn(n, 6),
    }, batch_size=[n])


def test_unbatched_forward_matches_single_path():
    """batch_size=[] -> scalar long action + plain-string thinking (backward compat)."""
    actor = _actor(answers=[2])
    td = actor.forward(_unbatched_td())
    assert td["action"].item() == 2
    assert td["action"].dtype == torch.long
    assert td["action"].shape == torch.Size([])
    assert isinstance(td["thinking"], str)
    assert "<answer>2</answer>" in td["thinking"]
    assert isinstance(td["system_prompt"], str)
    assert isinstance(td["user_prompt"], str)
    assert len(actor.last_user_prompts) == 1


@pytest.mark.parametrize("n", [1, 3])
def test_batched_forward_builds_n_prompts_and_actions(n):
    """batch_size=[n] -> n prompts built, action tensor shape [n] with correct values."""
    actor = _actor(answers=[0, 1, 2])
    td = actor.forward(_batched_td(n))
    assert td["action"].shape == torch.Size([n])
    assert td["action"].dtype == torch.long
    assert td["action"].tolist() == [0, 1, 2][:n]  # answers cycle [0,1,2]
    assert len(actor.last_user_prompts) == n
    # each sub-prompt was built independently (contains the account-state header)
    for p in actor.last_user_prompts:
        assert "exposure_pct" in p


def test_batched_malformed_response_falls_back_per_element():
    """A malformed response in the batch -> that element's action is 0, others unaffected."""
    class HalfBadActor(StubActor):
        def generate_batch(self, system_prompt, user_prompts):
            self.last_user_prompts = list(user_prompts)
            # element 0 valid (1), element 1 malformed, element 2 valid (2)
            return ["<answer>1</answer>", "no tag here", "<answer>2</answer>"]
    actor = HalfBadActor(answers=[0], market_data_keys=[MARKET_KEY],
                         account_state_labels=ACCOUNT_LABELS, action_levels=ACTION_LEVELS)
    td = actor.forward(_batched_td(3))
    assert td["action"].tolist() == [1, 0, 2]


def test_generate_convenience_wraps_generate_batch():
    """generate(system, user) returns the single string from generate_batch."""
    actor = _actor(answers=[1])
    out = actor.generate("sys", "usr")
    assert out == "<answer>1</answer>"
    assert isinstance(out, str)
