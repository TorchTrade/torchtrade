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


@pytest.mark.parametrize("leverage,threshold", [(20, 0.0), (50, 0.0), (20, 0.1)])
def test_a_wiped_lane_terminates_instead_of_stepping_into_a_broken_reward(leverage, threshold):
    """The reachable defect, and the reason the raise below is an assertion not a crash.

    `_apply_liquidation` clamps a wiped balance to exactly 0.0, and termination tested
    `new_pv < initial * threshold` -- strict. With `bankrupt_threshold=0.0` (explicitly
    permitted, and documented as "leveraged positions can still terminate on
    portfolio_value <= 0") that is `0.0 < 0.0`, False. So a dead lane never terminated,
    never reset, and kept stepping. Measured before the fix: raised at step 1 with
    `envs [11, 23, 48]` at PV 0.0.
    """
    n = 600
    prices = np.maximum(100 + np.cumsum(np.random.RandomState(0).randn(n) * 2.0), 1.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": np.ones(n) * 1000,
    })
    env = VectorizedSequentialTradingEnv(
        df,
        VectorizedSequentialTradingEnvConfig(
            time_frames=["1Min"], window_sizes=[10], execute_on="1Min", num_envs=64,
            leverage=leverage, bankrupt_threshold=threshold,
            action_levels=[-1, 0, 1], transaction_fee=0.0004,
        ),
    )
    td = env.reset()
    for _ in range(60):
        td["action"] = torch.randint(0, 3, (64,))
        td = env.step(td)["next"]           # must not raise
        if td["done"].any():
            td = env.reset(td)

    assert (env._portfolio_values > 0).all(), (
        "a lane is sitting at a non-positive portfolio value without having terminated"
    )


def test_the_reward_matches_log_return_reward_through_the_real_env():
    """Drives the env, rather than re-implementing the formula in the test.

    The first version built the "vectorized" side from `torch.tensor([...])` -- float32,
    where the env's money dtype is float64 -- so it compared the test's arithmetic to
    itself and never touched env source. Two mutations survived it, including changing
    the -10.0 bankruptcy constant.
    """
    n = 300
    prices = 100 + np.cumsum(np.random.RandomState(3).randn(n) * 0.1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices, "high": prices + 0.2, "low": prices - 0.2,
        "close": prices, "volume": np.ones(n) * 1000,
    })
    env = VectorizedSequentialTradingEnv(
        df,
        VectorizedSequentialTradingEnvConfig(
            time_frames=["1Min"], window_sizes=[10], execute_on="1Min", num_envs=8,
        ),
    )
    td = env.reset()
    for _ in range(25):
        previous = env._portfolio_values.clone()
        td["action"] = torch.randint(0, env.action_spec.n, (8,))
        td = env.step(td)["next"]
        current = env._portfolio_values

        for i in range(8):
            expected = log_return_reward(_History([previous[i].item(), current[i].item()]))
            assert td["reward"][i].item() == pytest.approx(expected, abs=1e-5), (
                f"env {i}: {previous[i].item()} -> {current[i].item()} gave "
                f"{td['reward'][i].item()}, log_return_reward says {expected}"
            )
        if td["done"].any():
            td = env.reset(td)


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


def test_the_bankruptcy_reward_constant_matches_the_scalar():
    """-10.0 on `new_pv <= 0`, in both envs. This constant had ZERO coverage repo-wide:
    changing it to -3.0 passed the entire offline suite, including the equivalence tests,
    because none of them drives a lane to bankruptcy."""
    n = 400
    prices = np.maximum(100 + np.cumsum(np.random.RandomState(7).randn(n) * 3.0), 1.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices, "high": prices + 2, "low": prices - 2,
        "close": prices, "volume": np.ones(n) * 1000,
    })
    env = VectorizedSequentialTradingEnv(
        df,
        VectorizedSequentialTradingEnvConfig(
            time_frames=["1Min"], window_sizes=[10], execute_on="1Min", num_envs=64,
            leverage=100, bankrupt_threshold=0.0, action_levels=[-1, 0, 1],
            transaction_fee=0.0004,
        ),
    )
    td = env.reset()
    bankrupt_rewards = []
    for _ in range(80):
        previous = env._portfolio_values.clone()
        td["action"] = torch.randint(0, 3, (64,))
        td = env.step(td)["next"]
        wiped = env._portfolio_values <= 0
        if wiped.any():
            bankrupt_rewards += td["reward"][wiped].flatten().tolist()
        if td["done"].any():
            td = env.reset(td)

    assert bankrupt_rewards, "no lane reached bankruptcy; the fixture proves nothing"
    assert all(r == pytest.approx(-10.0) for r in bankrupt_rewards), (
        f"bankruptcy rewards {set(bankrupt_rewards)} -- log_return_reward returns -10.0"
    )
    assert log_return_reward(_History([100.0, 0.0])) == -10.0
