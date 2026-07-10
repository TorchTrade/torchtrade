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


class _FakeSLTPEnv:
    _action_tuple = [(None, None, None), ("long", -0.02, 0.03),
                     ("short", -0.02, 0.03), ("close", None, None)]


def test_action_descriptions_covers_every_side():
    """Regression: ('close', None, None) (include_close_action=True) must not crash on the
    {None:+.1%} format; hold/long/short/close all get a sensible description."""
    d = LLMTrainer._action_descriptions(_FakeSLTPEnv())
    assert d[0] == "Action 0 -> hold / no position"
    assert d[1] == "Action 1 -> open long: stop-loss -2.0%, take-profit +3.0%"
    assert d[2].startswith("Action 2 -> open short")
    assert d[3] == "Action 3 -> close current position"


def test_build_action_regex_constrains_to_valid_indices():
    import re
    rx = LLMTrainer._build_action_regex(3)
    assert rx == r"<answer>(0|1|2)</answer>"
    assert re.fullmatch(rx, "<answer>2</answer>") and not re.fullmatch(rx, "<answer>3</answer>")


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


def test_render_group_prompts_builds_contiguous_k_blocks(sample_ohlcv_df):
    """_render_group_prompts must emit `max_steps` bars, each repeated K times as a CONTIGUOUS
    block (make_trading_env's shuffle=False dataloader relies on this so one rollout == one
    bar's GRPO group). Guards the group-structure invariant off the GPU."""
    from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig
    from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
    from tests.conftest import simple_feature_fn

    cfg = OneStepTradingEnvConfig(
        initial_cash=1000, execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)], window_sizes=[10],
        seed=42, random_start=False, action_levels=[-1, 0, 1])
    trainer = LLMTrainer(df=sample_ohlcv_df, config=cfg, feature_preprocessing_fn=simple_feature_fn,
                         feature_keys=["close", "volume"], num_generations=3, max_steps=4)
    env = OneStepTradingEnv(df=sample_ohlcv_df, config=cfg, feature_preprocessing_fn=simple_feature_fn)
    pb = trainer._build_prompt_actor(env)

    sysp, prompts, bars = trainer._render_group_prompts(env, pb)
    assert isinstance(sysp, str) and len(prompts) == 4 * 3 and len(bars) == 4 * 3
    for i in range(0, 12, 3):                    # each K-block is one bar
        assert bars[i] == bars[i + 1] == bars[i + 2]
        assert prompts[i] == prompts[i + 1] == prompts[i + 2]
    assert len({bars[i] for i in range(0, 12, 3)}) == 4   # 4 distinct bars
    assert all(b >= 1 for b in bars)             # valid bar_index range
