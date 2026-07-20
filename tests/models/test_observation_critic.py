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


def test_value_of_a_real_single_observation(env):
    """A single env observation -> scalar V(s) of shape (1,)."""
    critic = ObservationCritic(env.market_data_keys)
    obs = env.obs_at(5)  # real reset td at bar 5, batch_size []
    value = critic(obs)
    assert value.shape == (1,)
    assert torch.isfinite(value).all()


def test_value_of_a_stacked_batch(env):
    """A batch of B bars (how the trainer feeds it) -> V(s) of shape (B, 1)."""
    critic = ObservationCritic(env.market_data_keys)
    batch = torch.stack([env.obs_at(b) for b in (3, 5, 7, 9)])  # batch_size [4]
    value = critic(batch)
    assert value.shape == (4, 1)
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
    """Reads only market_data + account_state; other keys in the td are ignored."""
    critic = ObservationCritic(["market_data_1Min"])
    td = TensorDict(
        {
            "market_data_1Min": torch.randn(2, 10, 4),
            "account_state": torch.randn(2, 6),
            "reward": torch.randn(2, 1),  # extraneous, must be ignored
        },
        batch_size=[2],
    )
    assert critic(td).shape == (2, 1)


def test_every_market_data_key_affects_the_value():
    """Multi-timeframe (differing flattened widths): EVERY market_data key must influence V.
    Guards the concat path against a dropped key — a regression that hardcoded keys[0] would
    still pass shape checks (LazyLinear masks the narrower input), so only value-sensitivity
    catches it."""
    critic = ObservationCritic(["market_data_a", "market_data_b"])
    td = TensorDict(
        {"market_data_a": torch.randn(2, 10, 4), "market_data_b": torch.randn(2, 8, 3),  # widths 40 vs 24
         "account_state": torch.randn(2, 6)},
        batch_size=[2],
    )
    v0 = critic(td)
    perturbed = td.clone()
    perturbed["market_data_b"] = torch.randn(2, 8, 3)  # change ONLY the second key
    assert not torch.allclose(v0, critic(perturbed)), "V ignores market_data_b — a key was dropped"
