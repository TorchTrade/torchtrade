"""score(bar_index, action) must equal the reward the normal reset+step path
produces for that action at that bar — the sacred-offline-env correctness guard."""
import pytest
import torch

from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn


def _make_onestep_env(df):
    """Build a synthetic OneStepTradingEnv, mirroring the onestep_config_spot
    fixture in tests/envs/offline/test_onestep.py."""
    config = OneStepTradingEnvConfig(
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        random_start=False,  # forced to True internally by OneStepTradingEnvConfig
    )
    return OneStepTradingEnv(df=df, config=config, feature_preprocessing_fn=simple_feature_fn)


@pytest.mark.parametrize("action", [0, 1, 2])
def test_score_matches_reset_step_reward(sample_ohlcv_df, action):
    env = _make_onestep_env(sample_ohlcv_df)
    td = env.reset()
    bar_index = env._reset_idx  # the bar the normal reset landed on
    td["action"] = torch.tensor(action)
    out = env.step(td)
    expected = out["next", "reward"].item()
    env.close()

    env2 = _make_onestep_env(sample_ohlcv_df)
    result = env2.score(bar_index, action)
    env2.close()

    assert result == pytest.approx(expected)


class _ZeroStartRNG:
    """Test double for sampler.np_rng that always draws start_idx=0.

    Used to independently reproduce, via the UNMODIFIED organic random-start
    branch of MarketDataObservationSampler.reset() (not via seek()/score()),
    the reward for landing on the very first bar of data (sampler index 0).
    This gives a ground truth for bar_index=1 that does not depend on the
    seek() mechanism under test.
    """

    def integers(self, *args, **kwargs):
        return 0


@pytest.mark.parametrize("action", [0, 1, 2])
def test_score_bar_index_one_matches_first_bar(sample_ohlcv_df, action):
    """bar_index is defined in post-increment `_reset_idx` terms (see score()'s
    docstring), so the SMALLEST valid value is 1 -- it represents landing on the
    very first bar of data (sampler start index 0). bar_index=1 is exercised here
    (not literal 0) because 0 is provably unreachable by any organic reset: an
    organic start-index draw is always >= 0, so `_reset_idx = start_idx + 1` is
    always >= 1. This test guards the off-by-one boundary that
    `test_score_bar_index_zero_raises` (below) shows is otherwise silently wrong.
    """
    env = _make_onestep_env(sample_ohlcv_df)
    env.sampler.np_rng = _ZeroStartRNG()  # force organic start_idx=0, independent of seek()
    td = env.reset()
    assert env._reset_idx == 1
    td["action"] = torch.tensor(action)
    out = env.step(td)
    expected = out["next", "reward"].item()
    env.close()

    env2 = _make_onestep_env(sample_ohlcv_df)
    result = env2.score(1, action)
    env2.close()

    assert result == pytest.approx(expected)


def test_score_bar_index_zero_raises_value_error(sample_ohlcv_df):
    """bar_index=0 is never organically reachable (see test above) -- it maps to
    sampler.seek(-1), which must raise ValueError instead of the pre-fix behavior
    of silently wrapping around to `_exec_times_arr[-1]` (the LAST bar of data)."""
    env = _make_onestep_env(sample_ohlcv_df)
    with pytest.raises(ValueError):
        env.score(0, action=0)
    env.close()


def test_score_bar_index_out_of_range_high_raises_value_error(sample_ohlcv_df):
    """bar_index beyond the sampler's data must raise, not silently seek OOB."""
    env = _make_onestep_env(sample_ohlcv_df)
    total_len = len(env.sampler._exec_times_arr)
    with pytest.raises(ValueError):
        env.score(total_len + 1, action=0)
    env.close()


def test_score_bar_index_max_valid_succeeds(sample_ohlcv_df):
    """The largest bar_index an organic reset could ever produce (== total_len)
    must be accepted, not off-by-one rejected by the new bounds check."""
    env = _make_onestep_env(sample_ohlcv_df)
    total_len = len(env.sampler._exec_times_arr)
    result = env.score(total_len, action=0)
    env.close()
    assert isinstance(result, float)


@pytest.mark.parametrize("action", [0, 1, 2])
def test_obs_at_matches_reset_obs(sample_ohlcv_df, action):
    """obs_at(bar_index) returns the same market observation a reset landing on that bar
    yields — the deterministic per-bar prompt source used by the GRPO trainer."""
    env = _make_onestep_env(sample_ohlcv_df)
    td = env.reset()
    bar_index = env._reset_idx
    env.close()

    env2 = _make_onestep_env(sample_ohlcv_df)
    obs = env2.obs_at(bar_index)
    env2.close()

    mkey = env.market_data_keys[0]
    assert torch.allclose(td[mkey].float(), obs[mkey].float(), atol=1e-4)
