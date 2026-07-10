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
