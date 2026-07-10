"""Non-GPU tests for the LLMTrainer facade construction + the reward-parser helper.
(The full training loop is GPU-only and is validated on the DGX Spark.)"""
import pytest

from torchtrade.train import LLMTrainer
from torchtrade.train.trading_env import TradingRewardParser


def test_rejects_num_generations_below_two():
    with pytest.raises(ValueError, match="num_generations"):
        LLMTrainer(df=None, config=None, num_generations=1)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        LLMTrainer(df=None, config=None, method="bogus")


class _Turn:
    def __init__(self, content):
        self.content = content


def test_reward_parser_extracts_last_turn_content():
    p = TradingRewardParser(score_env=None, num_actions=3)
    assert p._response_text([_Turn("sys"), _Turn("<answer>2</answer>")]) == "<answer>2</answer>"


def test_reward_parser_threads_custom_reward_fn():
    """A custom reward_fn passed to TradingRewardParser is used instead of env.score
    (the override was previously stored-and-ignored)."""
    seen = {}

    def rf(action, bar_index, env):
        seen.update(action=action, bar_index=bar_index)
        return 42.0

    parser = TradingRewardParser(score_env=object(), num_actions=3, reward_fn=rf)
    assert parser.reward("<answer>2</answer>", bar_index=5) == 42.0
    assert seen == {"action": 2, "bar_index": 5}


class _FakeTD:
    """Minimal tensordict stand-in: avoids TensorDict's non-tensor stacking mangling the
    mock History turns, so _step's own zip/score/reshape logic is what's under test."""
    def __init__(self, data, batch_size):
        self._d, self.batch_size = data, batch_size
    def get(self, key):
        return self._d[key]
    def set(self, key, value):
        self._d[key] = value
        return self
    def __getitem__(self, key):
        return self._d[key]


def test_reward_parser_step_scores_each_group_item():
    """_step reads ("history","full") + bar_index, scores each completion, and writes one
    reward per group item with shape batch+(1,1). Guards the zip/reshape glue (GPU-only path
    otherwise)."""
    import torch

    class _FakeScoreEnv:
        def score(self, bar_index, action):
            return float(action + bar_index)

    parser = TradingRewardParser(_FakeScoreEnv(), num_actions=3)
    td = _FakeTD({("history", "full"): [[_Turn("<answer>2</answer>")], [_Turn("<answer>1</answer>")]],
                  "bar_index": torch.tensor([5, 7])}, torch.Size([2]))
    next_td = _FakeTD({}, torch.Size([2]))

    out = parser._step(td, next_td)
    assert out["reward"].shape == (2, 1, 1)
    assert out["reward"].flatten().tolist() == [7.0, 8.0]  # score(5,2)=7, score(7,1)=8
