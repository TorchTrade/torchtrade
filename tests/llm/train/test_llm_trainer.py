"""Non-GPU tests for the LLMTrainer facade construction + the reward-parser helper.
(The full training loop is GPU-only and is validated on the DGX Spark.)"""
import re

import pytest

from torchtrade.llm.train import LLMTrainer
from torchtrade.llm.train.trading_env import TradingRewardParser

_THINK = "the daily trend is up, price is above the SMA and momentum looks constructive"  # >=40 chars


def test_rejects_num_generations_below_two():
    with pytest.raises(ValueError, match="num_generations"):
        LLMTrainer(df=None, config=None, num_generations=1)


def test_sao_allows_num_generations_one():
    """SAO is single-rollout — its baseline is the critic, so a group of 1 is legal. The
    num_generations>=2 guard must NOT fire for loss='sao' (it must for grpo, above)."""
    LLMTrainer(df=None, config=None, loss="sao", num_generations=1)  # must not raise


def test_rejects_critic_updates_below_one():
    with pytest.raises(ValueError, match="critic_updates"):
        LLMTrainer(df=None, config=None, loss="sao", num_generations=1, critic_updates=0)


def test_base_load_kwargs_uses_transformers_compatible_dtype():
    """Regression: transformers>=4.30 (the [llm] floor) accepts torch_dtype, not dtype (added
    ~4.56), so build_train_policy must pass torch_dtype to from_pretrained."""
    from torchtrade.llm.train.models import _base_load_kwargs
    kw = _base_load_kwargs()
    assert "torch_dtype" in kw and "dtype" not in kw


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


def _fmt(think, ans=1):
    return f"<think>{think}</think><answer>{ans}</answer>"


@pytest.mark.parametrize("n,good_idx,bad_idx", [
    (3, 2, 3),      # single-digit: 2 in range, 3 out
    (11, 10, 11),   # multi-digit: 10 in range, 11 out — guards against a naive [0-9] rewrite
], ids=["single-digit", "multi-digit"])
def test_build_action_regex_index_range(n, good_idx, bad_idx):
    """The <answer> index must be a valid action index (0..n-1), including multi-digit ones."""
    rx = LLMTrainer._build_action_regex(n)
    assert re.fullmatch(rx, _fmt(_THINK, good_idx))
    assert not re.fullmatch(rx, _fmt(_THINK, bad_idx))


def test_build_action_regex_structure_is_enforceable():
    """The regex must be guided-decoding-enforceable: reasoning REQUIRED and non-empty, nothing after
    </answer>, the {40,600} think bounds are load-bearing, and the inter-tag gap is BOUNDED. Regression
    for the old [\\s\\S]*? regex (empty think / no answer / trailing text) AND the unbounded \\s* gap
    that let thinking models dump whitespace to the token cap instead of answering (16-35% no-answer)."""
    rx = LLMTrainer._build_action_regex(3)
    assert not re.fullmatch(rx, "<answer>2</answer>")            # missing <think>: reasoning enforced
    assert not re.fullmatch(rx, f"<think>{_THINK}</think>")      # missing answer
    assert not re.fullmatch(rx, "<think></think><answer>1</answer>")     # empty think rejected
    assert not re.fullmatch(rx, _fmt(_THINK) + " extra")                 # no trailing text
    assert not re.fullmatch(rx, _fmt(f"has a < bracket {_THINK}"))       # '<' banned in think body
    # inter-tag gap is bounded to \\s{0,2}: 0-2 whitespace OK, 3+ rejected (kills the whitespace-dump
    # escape hatch). Pinning the upper bound fails a regression back to \\s* / \\s+.
    for gap, ok in [("", True), ("\n", True), ("\n\n", True), ("\n\n\n", False), ("   ", False)]:
        assert bool(re.fullmatch(rx, f"<think>{_THINK}</think>{gap}<answer>2</answer>")) is ok, repr(gap)
    # pin BOTH think-length bounds so a widening to {40,} or {1,600} fails here:
    for length, ok in [(39, False), (40, True), (600, True), (601, False)]:
        assert bool(re.fullmatch(rx, _fmt("a" * length))) is ok, length


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

    # SAO (distinct=True): max_steps*K DISTINCT bars, each rendered ONCE — not the K-repeat
    # blocks. `len(set) > max_steps` is the tell it isn't the GRPO grouping (which gives exactly 4).
    _, sao_prompts, sao_bars = trainer._render_group_prompts(env, pb, distinct=True)
    assert len(sao_prompts) == 4 * 3 and len(sao_bars) == 4 * 3
    assert len(set(sao_bars)) > 4
