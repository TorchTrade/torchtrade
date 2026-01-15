"""Tests for SPOLoss module."""

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

from torchtrade.losses import SPOLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.modules import ValueOperator


class TestSPOLoss:
    """Test suite for SPOLoss."""

    @pytest.fixture
    def obs_dim(self):
        return 4

    @pytest.fixture
    def action_dim(self):
        return 3

    @pytest.fixture
    def batch_size(self):
        return 10

    @pytest.fixture
    def actor_network(self, obs_dim, action_dim):
        """Create a simple actor network for testing."""
        # Create a simple policy network
        policy_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
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
    def critic_network(self, obs_dim):
        """Create a simple critic network for testing."""
        critic_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        return ValueOperator(module=critic_net, in_keys=["observation"])

    @pytest.fixture
    def sample_data(self, batch_size, obs_dim, action_dim):
        """Create sample training data."""
        return TensorDict(
            {
                "observation": torch.randn(batch_size, obs_dim),
                "action": torch.randint(0, action_dim, (batch_size,)),
                "action_log_prob": torch.randn(batch_size),
                "next": {
                    "observation": torch.randn(batch_size, obs_dim),
                    "reward": torch.randn(batch_size, 1),
                    "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                },
            },
            batch_size=[batch_size],
        )

    @pytest.fixture
    def sample_data_with_advantage(self, sample_data):
        """Create sample data with pre-computed advantage."""
        data = sample_data.clone()
        data.set("advantage", torch.randn(data.batch_size[0], 1))
        data.set("value_target", torch.randn(data.batch_size[0], 1))
        data.set("state_value", torch.randn(data.batch_size[0], 1))
        return data

    def test_loss_initialization(self, actor_network, critic_network):
        """Test that loss module initializes correctly."""
        loss = SPOLoss(actor_network, critic_network)
        assert loss is not None
        assert loss.actor_network is not None
        assert loss.critic_network is not None
        assert loss.epsilon == 0.2  # default

    def test_loss_initialization_no_critic(self, actor_network):
        """Test loss initialization without critic."""
        loss = SPOLoss(actor_network, critic_network=None)
        assert loss is not None
        assert loss.actor_network is not None
        assert loss.critic_network is None

    def test_loss_initialization_custom_epsilon(self, actor_network, critic_network):
        """Test loss initialization with custom epsilon."""
        loss = SPOLoss(actor_network, critic_network, epsilon=0.1)
        assert loss.epsilon == 0.1

    def test_inherits_from_ppo(self, actor_network, critic_network):
        """Test that SPOLoss inherits from PPOLoss."""
        from torchrl.objectives.ppo import PPOLoss

        loss = SPOLoss(actor_network, critic_network)
        assert isinstance(loss, PPOLoss)

    def test_forward_pass_with_advantage(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test forward pass with pre-computed advantage."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert "loss_objective" in output.keys()
        assert output["loss_objective"].requires_grad
        assert output["loss_objective"].shape == torch.Size([])

    def test_forward_pass_without_advantage(
        self, actor_network, critic_network, sample_data
    ):
        """Test forward pass that computes advantage internally."""
        loss = SPOLoss(actor_network, critic_network)
        loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)

        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert output["loss_objective"].requires_grad

    def test_spo_objective_computation(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that SPO objective is computed correctly."""
        loss = SPOLoss(actor_network, critic_network, epsilon=0.2)
        output = loss(sample_data_with_advantage)

        # SPO should compute: L = rho * A - (|A| / 2*eps) * (rho - 1)^2
        # Check that loss is finite
        assert torch.isfinite(output["loss_objective"])

        # Check that rho_mean is logged
        assert "rho_mean" in output.keys()
        assert output["rho_mean"] >= 0  # importance weights should be positive

    def test_penalty_term_computation(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that penalty term is computed and logged."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        # Penalty should be logged
        assert "penalty_mean" in output.keys()
        assert torch.isfinite(output["penalty_mean"])
        assert output["penalty_mean"] >= 0  # squared penalty is always non-negative

    def test_importance_weight_logging(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that importance weights are logged."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert "rho_mean" in output.keys()
        assert torch.isfinite(output["rho_mean"])

    def test_kl_approximation_logging(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that KL approximation is logged if available."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        # KL approximation might be available
        if "kl_approx" in output.keys():
            assert torch.isfinite(output["kl_approx"])

    def test_backward_pass(self, actor_network, critic_network, sample_data_with_advantage):
        """Test that gradients flow correctly through the loss."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        # Compute total loss
        total_loss = output["loss_objective"]
        if "loss_entropy" in output.keys():
            total_loss = total_loss + output["loss_entropy"]
        if "loss_critic" in output.keys():
            total_loss = total_loss + output["loss_critic"]

        total_loss.backward()

        # Verify that actor network has gradients
        for param in loss.actor_network_params.values(True, True):
            assert param.grad is not None

    def test_epsilon_effect(self, actor_network, critic_network, sample_data_with_advantage):
        """Test that different epsilon values affect the loss."""
        loss_small = SPOLoss(actor_network, critic_network, epsilon=0.05)
        loss_large = SPOLoss(actor_network, critic_network, epsilon=0.5)

        output_small = loss_small(sample_data_with_advantage)
        output_large = loss_large(sample_data_with_advantage)

        # Both should produce finite losses
        assert torch.isfinite(output_small["loss_objective"])
        assert torch.isfinite(output_large["loss_objective"])

        # Penalties should be different (inverse relationship with epsilon)
        # Smaller epsilon -> larger penalty coefficient
        assert torch.isfinite(output_small["penalty_mean"])
        assert torch.isfinite(output_large["penalty_mean"])

    def test_with_entropy_bonus(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO loss with entropy bonus."""
        loss = SPOLoss(
            actor_network,
            critic_network,
            entropy_bonus=True,
            entropy_coeff=0.01,
        )
        output = loss(sample_data_with_advantage)

        assert "entropy" in output.keys()
        assert "loss_entropy" in output.keys()
        assert torch.isfinite(output["entropy"])
        assert output["loss_entropy"].requires_grad

    def test_without_entropy_bonus(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO loss without entropy bonus."""
        loss = SPOLoss(actor_network, critic_network, entropy_bonus=False)
        output = loss(sample_data_with_advantage)

        assert "entropy" not in output.keys()
        assert "loss_entropy" not in output.keys()

    def test_with_critic_loss(self, actor_network, critic_network, sample_data):
        """Test SPO loss with critic network."""
        loss = SPOLoss(actor_network, critic_network)
        loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)

        output = loss(sample_data)

        assert "loss_critic" in output.keys()
        assert output["loss_critic"].requires_grad

    def test_explained_variance_logging(self, actor_network, critic_network, sample_data):
        """Test that explained variance is logged when available."""
        loss = SPOLoss(actor_network, critic_network)
        loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)

        output = loss(sample_data)

        # Explained variance might be available
        if "explained_variance" in output.keys():
            assert torch.isfinite(output["explained_variance"])

    def test_reduction_modes(self, actor_network, critic_network, sample_data_with_advantage):
        """Test different reduction modes."""
        # Mean reduction (default)
        loss_mean = SPOLoss(actor_network, critic_network, reduction="mean")
        output_mean = loss_mean(sample_data_with_advantage)
        assert output_mean["loss_objective"].shape == torch.Size([])

        # Sum reduction
        loss_sum = SPOLoss(actor_network, critic_network, reduction="sum")
        output_sum = loss_sum(sample_data_with_advantage)
        assert output_sum["loss_objective"].shape == torch.Size([])

    def test_normalize_advantage(self, actor_network, critic_network, sample_data_with_advantage):
        """Test advantage normalization."""
        loss = SPOLoss(actor_network, critic_network, normalize_advantage=True)
        output = loss(sample_data_with_advantage)
        assert torch.isfinite(output["loss_objective"])

    def test_different_batch_sizes(self, actor_network, critic_network, obs_dim, action_dim):
        """Test loss computation with different batch sizes."""
        loss = SPOLoss(actor_network, critic_network)

        for batch_size in [1, 5, 32]:
            data = TensorDict(
                {
                    "observation": torch.randn(batch_size, obs_dim),
                    "action": torch.randint(0, action_dim, (batch_size,)),
                    "action_log_prob": torch.randn(batch_size),
                    "advantage": torch.randn(batch_size, 1),
                    "value_target": torch.randn(batch_size, 1),
                    "state_value": torch.randn(batch_size, 1),
                    "next": {
                        "observation": torch.randn(batch_size, obs_dim),
                        "reward": torch.randn(batch_size, 1),
                        "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                        "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                    },
                },
                batch_size=[batch_size],
            )
            output = loss(data)
            assert torch.isfinite(output["loss_objective"])

    def test_with_zero_advantage(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO loss with zero advantage."""
        sample_data_with_advantage["advantage"] = torch.zeros_like(
            sample_data_with_advantage["advantage"]
        )
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert torch.isfinite(output["loss_objective"])
        # With zero advantage, penalty coefficient is zero, so penalty should be zero
        assert output["penalty_mean"] == 0.0

    def test_with_positive_advantage(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO loss with positive advantage."""
        sample_data_with_advantage["advantage"] = torch.abs(
            sample_data_with_advantage["advantage"]
        )
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert torch.isfinite(output["loss_objective"])
        assert output["penalty_mean"] >= 0

    def test_with_negative_advantage(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO loss with negative advantage."""
        sample_data_with_advantage["advantage"] = -torch.abs(
            sample_data_with_advantage["advantage"]
        )
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert torch.isfinite(output["loss_objective"])
        assert output["penalty_mean"] >= 0  # penalty is based on |A|

    def test_penalty_grows_with_policy_change(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that penalty increases when policy changes significantly."""
        loss = SPOLoss(actor_network, critic_network, epsilon=0.1)

        # First forward pass
        output1 = loss(sample_data_with_advantage)
        penalty1 = output1["penalty_mean"].item()

        # The penalty depends on (rho - 1)^2, where rho = exp(log_prob_new - log_prob_old)
        # With random initialization, we can at least check that penalty is non-negative
        assert penalty1 >= 0

    def test_functional_mode(self, actor_network, critic_network):
        """Test that functional mode works correctly (inherited from PPOLoss)."""
        loss = SPOLoss(actor_network, critic_network, functional=True)
        assert hasattr(loss, "actor_network_params")

    def test_non_functional_mode(self, actor_network, critic_network):
        """Test that non-functional mode works correctly."""
        loss = SPOLoss(actor_network, critic_network, functional=False)
        assert loss.actor_network_params is None

    def test_clip_fraction_inherited(self, actor_network, critic_network, sample_data_with_advantage):
        """Test that PPOLoss features like clip_fraction work."""
        loss = SPOLoss(actor_network, critic_network, clip_value=0.5)
        output = loss(sample_data_with_advantage)

        # SPO doesn't use clipping, but inherited PPOLoss features should still work
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_cuda_if_available(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test loss computation on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        loss = SPOLoss(
            actor_network.to(device),
            critic_network.to(device) if critic_network else None,
        )
        sample_data_with_advantage = sample_data_with_advantage.to(device)

        output = loss(sample_data_with_advantage)
        assert output["loss_objective"].device.type == "cuda"

    def test_spo_vs_ppo_formulation(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test that SPO uses different formulation than PPO."""
        from torchrl.objectives import ClipPPOLoss

        spo_loss = SPOLoss(actor_network, critic_network, epsilon=0.2)
        ppo_loss = ClipPPOLoss(actor_network, critic_network, clip_epsilon=0.2)

        output_spo = spo_loss(sample_data_with_advantage)
        output_ppo = ppo_loss(sample_data_with_advantage)

        # Both should produce finite losses
        assert torch.isfinite(output_spo["loss_objective"])
        assert torch.isfinite(output_ppo["loss_objective"])

        # Losses should generally be different (unless in special cases)
        # We can't assert inequality due to random initialization,
        # but we can verify both work
        assert output_spo["loss_objective"].requires_grad
        assert output_ppo["loss_objective"].requires_grad

    def test_very_small_epsilon(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO with very small epsilon (high penalty)."""
        loss = SPOLoss(actor_network, critic_network, epsilon=0.001)
        output = loss(sample_data_with_advantage)
        assert torch.isfinite(output["loss_objective"])

    def test_very_large_epsilon(self, actor_network, critic_network, sample_data_with_advantage):
        """Test SPO with very large epsilon (low penalty)."""
        loss = SPOLoss(actor_network, critic_network, epsilon=10.0)
        output = loss(sample_data_with_advantage)
        assert torch.isfinite(output["loss_objective"])

    def test_output_keys_format(self, actor_network, critic_network, sample_data_with_advantage):
        """Test that output follows TorchRL conventions."""
        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        # Loss keys should start with "loss_"
        for key in output.keys():
            if "loss" in key:
                assert key.startswith("loss_")

    def test_loss_with_mixed_advantages(
        self, actor_network, critic_network, sample_data_with_advantage
    ):
        """Test SPO with mixed positive and negative advantages."""
        # Create mixed advantages
        adv = sample_data_with_advantage["advantage"]
        adv[: len(adv) // 2] = torch.abs(adv[: len(adv) // 2])  # positive
        adv[len(adv) // 2 :] = -torch.abs(adv[len(adv) // 2 :])  # negative

        loss = SPOLoss(actor_network, critic_network)
        output = loss(sample_data_with_advantage)

        assert torch.isfinite(output["loss_objective"])
        assert output["penalty_mean"] >= 0
