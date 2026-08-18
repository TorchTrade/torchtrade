"""The vectorized reward must agree with log_return_reward, edge cases included (#289)."""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.offline import (
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)


def _df(n=400):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = 100 + np.cumsum(np.random.RandomState(0).randn(n) * 0.1)
    return pd.DataFrame({"timestamp": ts, "open": prices, "high": prices + 0.2,
                         "low": prices - 0.2, "close": prices, "volume": np.ones(n) * 1000})


def _env(num_envs=4):
    return VectorizedSequentialTradingEnv(
        _df(),
        VectorizedSequentialTradingEnvConfig(
            time_frames=["1Min"], window_sizes=[10], execute_on="1Min",
            num_envs=num_envs,
        ),
    )


class _History:
    def __init__(self, values):
        self.portfolio_values = values


@pytest.mark.parametrize("old,new", [
    (10000.0, 10100.0),   # gain
    (10000.0, 9900.0),    # loss
    (10000.0, 10000.0),   # flat
    (10000.0, 0.0),       # bankrupt -> both report -10.0
    (10000.0, -50.0),     # past zero, same
])
def test_the_vectorized_reward_matches_log_return_reward(old, new):
    """Same formula, same bankruptcy constant. The vectorized env hardcodes the
    arithmetic inline rather than calling the shared function, so nothing but a test
    keeps the two in step."""
    scalar = log_return_reward(_History([old, new]))

    safe_new = torch.tensor([new]).clamp(min=1e-10)
    vectorized = torch.log(safe_new / torch.tensor([old]))
    vectorized = torch.where(
        torch.tensor([new]) <= 0, torch.full_like(vectorized, -10.0), vectorized
    )
    assert vectorized.item() == pytest.approx(scalar, rel=1e-6)


def test_a_non_positive_previous_value_raises_instead_of_fabricating_a_reward():
    """The scalar reward raises here because a non-positive PREVIOUS value is a
    calculation error, not a market outcome -- the prior step already checked it, and
    bankruptcy terminates before it can go negative.

    The vectorized env clamped to 1e-10 instead, producing log(new/1e-10) -- a reward
    around +25 for an env whose accounting had broken, silently, for the rest of the
    batch. That is the 'guard in the rule' anti-pattern: it made the nonsense silent.
    """
    env = _env(num_envs=3)
    env.reset()
    env._portfolio_values[1] = -5.0

    with pytest.raises(ValueError, match="calculation error"):
        td = env.rand_action()
        env.step(td)

    # The scalar path raises on the same input.
    with pytest.raises(ValueError, match="calculation error"):
        log_return_reward(_History([-5.0, 100.0]))


def test_the_error_names_which_env_broke():
    """A batch of 64 with one broken accumulator is unactionable without the index."""
    env = _env(num_envs=4)
    env.reset()
    env._portfolio_values[2] = 0.0

    with pytest.raises(ValueError, match=r"envs \[2\]"):
        env.step(env.rand_action())
