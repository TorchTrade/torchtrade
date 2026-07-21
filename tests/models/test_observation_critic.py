"""Tests for ObservationCritic — the SAO state-value baseline.

Tested against a REAL OneStepTradingEnv observation (not a hand-built mock), so a
drift in the env's obs structure surfaces here rather than only on the Spark.
"""

import pytest
import torch
from tensordict import TensorDict

from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig
from torchtrade.models.observation_critic import ObservationCritic
from tests.conftest import simple_feature_fn


@pytest.fixture
def env(sample_ohlcv_df):
    config = OneStepTradingEnvConfig(
        initial_cash=1000,
        time_frames=["1Min"],
        window_sizes=[10],
        execute_on="1Min",
        seed=42,
        random_start=False,
    )
    e = OneStepTradingEnv(df=sample_ohlcv_df, config=config, feature_preprocessing_fn=simple_feature_fn)
    yield e
    e.close()


@pytest.mark.parametrize("bars,expected_shape", [
    (5, (1,)),                  # single real observation, batch_size []
    ((3, 5, 7, 9), (4, 1)),     # stacked batch of B bars (how the trainer feeds it)
], ids=["single", "stacked-batch"])
def test_value_of_a_real_observation(env, bars, expected_shape):
    """A real observation (single or stacked batch) -> finite V(s) of the expected shape."""
    critic = ObservationCritic(env.market_data_keys)
    obs = env.obs_at(bars) if isinstance(bars, int) else torch.stack([env.obs_at(b) for b in bars])
    value = critic(obs)
    assert value.shape == expected_shape
    assert torch.isfinite(value).all()


def test_critic_learns_a_constant_target(env):
    """MSE-regressing V toward a fixed target must reduce the loss — the training
    signal the SAO trainer relies on to build a useful baseline."""
    critic = ObservationCritic(env.market_data_keys, hidden_size=32)
    batch = torch.stack([env.obs_at(b) for b in (3, 5, 7, 9)])
    target = torch.tensor([[0.5], [-0.5], [0.2], [-0.1]])
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)

    critic(batch)  # materialize LazyLinear before capturing loss_before
    loss_before = torch.nn.functional.mse_loss(critic(batch), target).item()
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.mse_loss(critic(batch), target).backward()
        opt.step()
    loss_after = torch.nn.functional.mse_loss(critic(batch), target).item()
    assert loss_after < loss_before * 0.5


def test_ignores_extra_keys():
    """An extra td key must NOT change V (reads only the declared market_data + account_state).
    Asserts value-invariance, not just shape: LazyLinear absorbs any input width, so a mutation
    that concatenated every td key would still yield shape (2,1) — only the value check catches it."""
    critic = ObservationCritic(["market_data_1Min"])
    base = TensorDict(
        {"market_data_1Min": torch.randn(2, 10, 4), "account_state": torch.randn(2, 6)},
        batch_size=[2],
    )
    with_extra = base.clone()
    with_extra["reward"] = torch.randn(2, 1)  # extraneous key that must be ignored
    with torch.no_grad():
        critic(base)  # materialize LazyLinear on the declared-keys width first
        assert torch.allclose(critic(base), critic(with_extra))


@pytest.mark.parametrize("perturb_key", ["market_data_a", "market_data_b", "account_state"])
def test_every_declared_input_affects_the_value(perturb_key):
    """EVERY declared input — each market_data key AND account_state — must influence V. Guards
    the concat path against a dropped input: a hardcoded keys[0] OR a dropped account_state still
    passes shape checks (LazyLinear masks the narrower input), so only value-sensitivity per input
    catches it. account_state carries position/exposure/pnl — the state a trading baseline most needs."""
    critic = ObservationCritic(["market_data_a", "market_data_b"])
    td = TensorDict(
        {"market_data_a": torch.randn(2, 10, 4), "market_data_b": torch.randn(2, 8, 3),  # widths 40 vs 24
         "account_state": torch.randn(2, 6)},
        batch_size=[2],
    )
    with torch.no_grad():
        v0 = critic(td)  # materializes LazyLinear
        perturbed = td.clone()
        perturbed[perturb_key] = torch.randn_like(td[perturb_key])  # change ONLY this input
        assert not torch.allclose(v0, critic(perturbed)), f"V ignores {perturb_key} — an input was dropped"
