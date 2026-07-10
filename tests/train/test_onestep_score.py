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
