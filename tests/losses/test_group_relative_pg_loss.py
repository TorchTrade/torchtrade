"""Tests for GroupRelativePGLoss module."""

import functools

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.envs import (
    Compose,
    EnvCreator,
    FlattenObservation,
    InitTracker,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import ProbabilisticActor

from torchtrade.envs.offline.onestep import OneStepTradingEnv, OneStepTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.losses import GroupRelativePGLoss

from tests.conftest import simple_feature_fn


class TestGroupRelativePGLoss:
    """Test suite for GroupRelativePGLoss."""

    OBS_DIM = 4
    ACTION_DIM = 3
    BATCH_SIZE = 10

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        """Set random seed for reproducibility before each test."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    @pytest.fixture
    def actor_network(self):
        """Create a simple actor network for testing."""
        # Create a simple policy network
        policy_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(self.OBS_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, self.ACTION_DIM),
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        )

        # Wrap in probabilistic module
        actor = ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        return ProbabilisticTensorDictSequential(policy_net, actor)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return TensorDict(
            {
                "observation": torch.randn(self.BATCH_SIZE, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (self.BATCH_SIZE,)),
                "action_log_prob": torch.randn(self.BATCH_SIZE),
                "next": {
                    "reward": torch.randn(self.BATCH_SIZE, 1),
                    "done": torch.zeros(self.BATCH_SIZE, 1, dtype=torch.bool),
                    "terminated": torch.zeros(self.BATCH_SIZE, 1, dtype=torch.bool),
                },
            },
            batch_size=[self.BATCH_SIZE],
        )

    def test_loss_initialization(self, actor_network):
        """Test that loss module initializes correctly."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        assert loss is not None
        assert loss.actor_network is not None
        assert loss.epsilon_low == 0.2
        assert loss.epsilon_high == 0.2
        assert loss.entropy_bonus is True
        assert loss.reduction == "mean"

    def test_loss_initialization_custom_params(self, actor_network):
        """Test loss initialization with custom parameters."""
        loss = GroupRelativePGLoss(
            actor_network=actor_network,
            epsilon_low=0.1,
            epsilon_high=0.3,
            entropy_bonus=False,
            entropy_coeff=0.02,
            reduction="sum",
        )
        assert loss.epsilon_low == pytest.approx(0.1)
        assert loss.epsilon_high == pytest.approx(0.3)
        assert loss.entropy_bonus is False
        assert loss.entropy_coeff.item() == pytest.approx(0.02)
        assert loss.reduction == "sum"

    def test_loss_initialization_missing_actor(self):
        """Test that loss raises error when actor_network is missing."""
        with pytest.raises(TypeError, match="Missing positional arguments actor_network"):
            GroupRelativePGLoss(actor_network=None)

    def test_loss_forward_pass(self, actor_network, sample_data):
        """Test forward pass produces expected outputs."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert output["loss_objective"].requires_grad
        assert output["loss_objective"].shape == torch.Size([])

    def test_loss_with_entropy_bonus(self, actor_network, sample_data):
        """Test that entropy bonus is computed when enabled."""
        loss = GroupRelativePGLoss(actor_network=actor_network, entropy_bonus=True)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" in output.keys()
        assert "loss_entropy" in output.keys()
        assert output["entropy"].shape == torch.Size([])
        assert output["loss_entropy"].requires_grad

    def test_loss_without_entropy_bonus(self, actor_network, sample_data):
        """Test that entropy is not computed when disabled."""
        loss = GroupRelativePGLoss(actor_network=actor_network, entropy_bonus=False)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" not in output.keys()
        assert "loss_entropy" not in output.keys()

    def test_loss_backward_pass(self, actor_network, sample_data):
        """Test that gradients flow correctly through the loss."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        # Check that we can compute gradients
        total_loss = output["loss_objective"]
        if "loss_entropy" in output.keys():
            total_loss = total_loss + output["loss_entropy"]

        total_loss.backward()

        # Verify that actor network has gradients
        for param in loss.actor_network_params.values(True, True):
            assert param.grad is not None

    def test_loss_reduction_modes(self, actor_network, sample_data):
        """Test different reduction modes."""
        # Mean reduction
        loss_mean = GroupRelativePGLoss(actor_network=actor_network, reduction="mean")
        output_mean = loss_mean(sample_data)
        assert output_mean["loss_objective"].shape == torch.Size([])

        # Sum reduction
        loss_sum = GroupRelativePGLoss(actor_network=actor_network, reduction="sum")
        output_sum = loss_sum(sample_data)
        assert output_sum["loss_objective"].shape == torch.Size([])

    @pytest.mark.parametrize("batch_size", [1, 5, 32])
    def test_loss_with_different_batch_sizes(self, actor_network, batch_size):
        """Test loss computation with different batch sizes."""
        loss = GroupRelativePGLoss(actor_network=actor_network)

        data = TensorDict(
            {
                "observation": torch.randn(batch_size, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (batch_size,)),
                "action_log_prob": torch.randn(batch_size),
                "next": {
                    "reward": torch.randn(batch_size, 1),
                    "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                },
            },
            batch_size=[batch_size],
        )
        output = loss(data)
        assert output["loss_objective"].requires_grad

    def test_loss_clipping_behavior(self, actor_network, sample_data):
        """Test that loss clipping works as expected."""
        loss = GroupRelativePGLoss(
            actor_network=actor_network,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        output = loss(sample_data)

        # Loss should be finite
        assert torch.isfinite(output["loss_objective"])

    def test_loss_advantage_computation(self, actor_network, sample_data):
        """Test that advantage is computed and logged correctly."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        # Advantage should be in the output for logging
        assert "advantage" in output.keys()
        assert output["advantage"].shape == torch.Size([])
        assert torch.isfinite(output["advantage"])

    def test_loss_kl_approximation(self, actor_network, sample_data):
        """Test that KL approximation is computed and logged."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        # KL approximation should be in the output
        assert "kl_approx" in output.keys()
        assert output["kl_approx"].shape == torch.Size([])
        assert torch.isfinite(output["kl_approx"])

    def test_entropy_coeff_scalar(self, actor_network):
        """Test entropy coefficient as scalar value."""
        loss = GroupRelativePGLoss(actor_network=actor_network, entropy_coeff=0.05)
        assert loss.entropy_coeff.item() == pytest.approx(0.05)

    def test_entropy_coeff_mapping(self, actor_network):
        """Test entropy coefficient as mapping for composite action spaces."""
        coeff_map = {"action_head_1": 0.01, "action_head_2": 0.02}
        loss = GroupRelativePGLoss(actor_network=actor_network, entropy_coeff=coeff_map)
        assert loss._entropy_coeff_map == coeff_map

    def test_functional_mode(self, actor_network):
        """Test that functional mode works correctly."""
        loss = GroupRelativePGLoss(actor_network=actor_network, functional=True)
        assert loss.functional is True
        assert loss.actor_network_params is not None

    def test_non_functional_mode(self, actor_network):
        """Test that non-functional mode works correctly."""
        loss = GroupRelativePGLoss(actor_network=actor_network, functional=False)
        assert loss.functional is False
        assert loss.actor_network_params is None

    def test_in_keys_property(self, actor_network):
        """Test that in_keys property returns correct keys."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        in_keys = loss.in_keys
        assert "observation" in in_keys
        assert "action" in in_keys

    def test_out_keys_property(self, actor_network):
        """Test that out_keys property returns correct keys."""
        loss_with_entropy = GroupRelativePGLoss(actor_network=actor_network, entropy_bonus=True)
        out_keys = loss_with_entropy.out_keys
        assert "loss_objective" in out_keys
        assert "entropy" in out_keys
        assert "loss_entropy" in out_keys

        loss_no_entropy = GroupRelativePGLoss(actor_network=actor_network, entropy_bonus=False)
        out_keys = loss_no_entropy.out_keys
        assert "loss_objective" in out_keys
        assert "entropy" not in out_keys
        assert "loss_entropy" not in out_keys

    def test_loss_with_zero_rewards(self, actor_network, sample_data):
        """Test loss computation with zero rewards."""
        sample_data["next", "reward"] = torch.zeros_like(sample_data["next", "reward"])
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert output["loss_objective"].requires_grad
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_negative_rewards(self, actor_network, sample_data):
        """Test loss computation with negative rewards."""
        sample_data["next", "reward"] = -torch.abs(sample_data["next", "reward"])
        loss = GroupRelativePGLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert output["loss_objective"].requires_grad
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_large_epsilon(self, actor_network, sample_data):
        """Test loss with large clipping epsilon values."""
        loss = GroupRelativePGLoss(
            actor_network=actor_network,
            epsilon_low=1.0,
            epsilon_high=1.0,
        )
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_small_epsilon(self, actor_network, sample_data):
        """Test loss with small clipping epsilon values."""
        loss = GroupRelativePGLoss(
            actor_network=actor_network,
            epsilon_low=0.01,
            epsilon_high=0.01,
        )
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_loss_samples_mc_entropy(self, actor_network, sample_data):
        """Test Monte Carlo entropy estimation with different sample counts."""
        loss = GroupRelativePGLoss(
            actor_network=actor_network,
            samples_mc_entropy=10,
            entropy_bonus=True,
        )
        output = loss(sample_data)
        assert "entropy" in output.keys()
        assert torch.isfinite(output["entropy"])

    def test_loss_with_cuda_if_available(self, actor_network, sample_data):
        """Test loss computation on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        loss = GroupRelativePGLoss(actor_network=actor_network.to(device))
        sample_data = sample_data.to(device)

        output = loss(sample_data)
        assert output["loss_objective"].device.type == "cuda"

    def test_reset_method(self, actor_network):
        """Test that reset method exists and works."""
        loss = GroupRelativePGLoss(actor_network=actor_network)
        loss.reset()  # Should not raise any errors

    def test_set_keys_invalidates_the_cached_in_keys(self, actor_network):
        """A renamed key must appear in in_keys, and the old one must be gone.

        This is the half that an empty `_forward_value_estimator_keys` would silently break:
        set_keys() would stop raising, tensor_keys would update, but the CACHED in_keys would
        still name the old key -- so a training loop doing tensordict.select(*loss.in_keys)
        would quietly drop the renamed one.
        """
        loss = GroupRelativePGLoss(actor_network=actor_network)
        _ = loss.in_keys                      # populate the cache with the default keys

        loss.set_keys(action="custom_action")

        in_keys = [str(k) for k in loss.in_keys]
        assert "custom_action" in in_keys
        assert "action" not in in_keys, "the cached in_keys still name the old key"

    def test_constructor_takes_the_keys_from_the_actor(self):
        """__init__ configures the loss's keys from the actor's, not from the defaults.

        The actor deliberately uses NON-default key names. Comparing against
        `actor.dist_sample_keys[0]` would be a tautology -- that is "action", which is also
        what _AcceptedKeys.action defaults to, so the assertion would hold even if __init__
        never configured anything (and it did hold: deleting the whole set_keys block from the
        constructor left every test in this file green).
        """
        policy_net = TensorDictModule(
            nn.Sequential(nn.Linear(self.OBS_DIM, 64), nn.ReLU(), nn.Linear(64, self.ACTION_DIM)),
            in_keys=["observation"], out_keys=["logits"],
        )
        actor = ProbabilisticTensorDictModule(
            in_keys=["logits"], out_keys=["my_action"],
            distribution_class=Categorical, return_log_prob=True,
        )
        loss = GroupRelativePGLoss(
            actor_network=ProbabilisticTensorDictSequential(policy_net, actor)
        )

        assert loss.tensor_keys.action == "my_action"          # not the "action" default
        assert loss.tensor_keys.sample_log_prob == "my_action_log_prob"


class TestGroupRelativePGLossGroupingInvariant:
    """Regression test for the group-relative-advantage invariant.

    GroupRelativePGLoss.forward() normalizes reward across dim 0 of the
    input batch. This is only correct group-relative (GRPO-style) advantage
    normalization when dim 0 is a genuine "K samples of the same state" axis
    -- which holds because every parallel copy of the env is constructed with
    the same config seed (so each builds an identical internal RNG and samples
    the same episode-start state) and, being one-step, all copies reset in
    lockstep. These tests exercise the real env+collector pipeline (not
    synthetic random tensors) to prove that invariant actually holds, then
    verify the loss's reported advantage matches a hand-computed group-relative
    value.
    """

    @staticmethod
    def _env_maker(df, seed):
        config = OneStepTradingEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=100,
            seed=seed,
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.04],
            include_hold_action=True,
        )
        return OneStepTradingEnv(
            df, config, feature_preprocessing_fn=simple_feature_fn
        )

    @pytest.fixture
    def grouped_batch(self, large_ohlcv_df):
        """A real collected batch with a genuine GRPO group structure.

        4 SerialEnv copies of OneStepTradingEnv, all built with the same
        config seed -- so each constructs an identical internal RNG and, being
        one-step (always done=True), resets in lockstep. At every collected
        time-step all 4 copies therefore sample the identical underlying market
        state (dim 0 = group axis), while different time-steps sample different
        states (dim 1 = distinct groups). (set_seed(static_seed=True) below
        mirrors the production make_environment call but is not what creates the
        groups -- the sampler draws reset positions from a local RNG seeded at
        construction, unaffected by set_seed.)
        """
        n_group = 4
        maker = functools.partial(self._env_maker, large_ohlcv_df, 0)
        serial_env = SerialEnv(n_group, EnvCreator(maker))
        serial_env.set_seed(0, static_seed=True)

        env = TransformedEnv(
            serial_env,
            Compose(
                InitTracker(),
                StepCounter(max_steps=large_ohlcv_df.shape[0]),
                FlattenObservation(in_keys=["market_data_1Minute_10"], first_dim=-2, last_dim=-1),
            ),
        )

        market_key = "market_data_1Minute_10"
        market_dim = env.observation_spec[market_key].shape[-1]
        account_dim = env.observation_spec["account_state"].shape[-1]

        class _TinyPolicyNet(nn.Module):
            def __init__(self, in_dim, n_actions):
                super().__init__()
                self.net = nn.Linear(in_dim, n_actions)

            def forward(self, market, account):
                return self.net(torch.cat([market, account], dim=-1))

        policy_net = TensorDictModule(
            _TinyPolicyNet(market_dim + account_dim, env.action_spec.n),
            in_keys=[market_key, "account_state"],
            out_keys=["logits"],
        )
        actor = ProbabilisticActor(
            policy_net,
            in_keys=["logits"],
            spec=env.full_action_spec_unbatched,
            distribution_class=Categorical,
            return_log_prob=True,
        )

        t_steps = 3
        collector = Collector(
            env, actor, frames_per_batch=n_group * t_steps, total_frames=n_group * t_steps, device="cpu"
        )
        data = next(iter(collector))
        collector.shutdown()
        return data, actor, n_group, t_steps

    def test_reset_index_constant_within_group(self, grouped_batch):
        """Every parallel-env slot must see the identical state within a time-step column."""
        data, _actor, n_group, t_steps = grouped_batch
        assert data.batch_size == torch.Size([n_group, t_steps])

        reset_idx = data["reset_index"]
        for t in range(t_steps):
            column = reset_idx[:, t]
            assert (column == column[0]).all(), (
                f"time-step {t}: expected all {n_group} parallel envs to share the same "
                f"reset_index (proving a valid GRPO group), got {column.tolist()}"
            )

    def test_advantage_normalized_over_group_axis_not_time_axis(self, grouped_batch):
        """The loss must normalize advantage over dim 0 (the group axis), not dim 1.

        Uses reduction="none" so loss_objective is per-element. With a freshly
        collected on-policy batch the importance ratio is 1, so
        loss_objective == -advantage element-wise. We inject a synthetic reward
        crafted so that per-group (dim 0) and per-time-step (dim 1) normalization
        give different tensors, then assert the loss matches the dim-0 version and
        NOT the dim-1 version. A future edit swapping .mean(0)/.std(0) for the
        wrong axis (the exact regression this class guards against) would flip
        which assertion passes -- a scalar-mean check could not catch that,
        because mean-centering makes the total mean ~0 regardless of axis.
        """
        data, actor, n_group, t_steps = grouped_batch

        # Row 0 is constant so dim-1 normalization differs sharply from dim-0;
        # every column (dim 0) still has real spread so dim-0 advantage is
        # well-defined and finite.
        synthetic_reward = torch.tensor(
            [[5.0, 5.0, 5.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 1.0, 2.0]]
        ).unsqueeze(-1)
        assert synthetic_reward.shape == torch.Size([n_group, t_steps, 1])
        data["next", "reward"] = synthetic_reward

        adv_group_axis = (
            synthetic_reward - synthetic_reward.mean(0, keepdim=True)
        ) / (synthetic_reward.std(0, keepdim=True) + 1e-8)
        adv_time_axis = (
            synthetic_reward - synthetic_reward.mean(1, keepdim=True)
        ) / (synthetic_reward.std(1, keepdim=True) + 1e-8)

        # The two axes must give genuinely different advantages, otherwise the
        # assertions below could not distinguish a correct loss from a broken one.
        assert not torch.allclose(adv_group_axis, adv_time_axis, atol=1e-4)

        loss = GroupRelativePGLoss(
            actor_network=actor, reduction="none", entropy_bonus=False
        )
        output = loss(data)

        # Fresh on-policy batch => importance ratio == 1 => per-element
        # loss_objective == -advantage.
        per_element_loss = output["loss_objective"].squeeze()
        assert torch.allclose(per_element_loss, (-adv_group_axis).squeeze(), atol=1e-4)
        assert not torch.allclose(per_element_loss, (-adv_time_axis).squeeze(), atol=1e-4)
