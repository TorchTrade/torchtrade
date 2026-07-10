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
