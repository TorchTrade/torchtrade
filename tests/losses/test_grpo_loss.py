"""Tests for GRPOLoss module."""

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

from torchtrade.losses import GRPOLoss


class TestGRPOLoss:
    """Test suite for GRPOLoss."""

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
    def sample_data(self, batch_size, obs_dim, action_dim):
        """Create sample training data."""
        return TensorDict(
            {
                "observation": torch.randn(batch_size, obs_dim),
                "action": torch.randint(0, action_dim, (batch_size,)),
                "action_log_prob": torch.randn(batch_size),
                "next": {
                    "reward": torch.randn(batch_size, 1),
                    "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                },
            },
            batch_size=[batch_size],
        )

    def test_loss_initialization(self, actor_network):
        """Test that loss module initializes correctly."""
        loss = GRPOLoss(actor_network=actor_network)
        assert loss is not None
        assert loss.actor_network is not None
        assert loss.epsilon_low == 0.2
        assert loss.epsilon_high == 0.2
        assert loss.entropy_bonus is True
        assert loss.reduction == "mean"

    def test_loss_initialization_custom_params(self, actor_network):
        """Test loss initialization with custom parameters."""
        loss = GRPOLoss(
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
            GRPOLoss(actor_network=None)

    def test_loss_forward_pass(self, actor_network, sample_data):
        """Test forward pass produces expected outputs."""
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert output["loss_objective"].requires_grad
        assert output["loss_objective"].shape == torch.Size([])

    def test_loss_with_entropy_bonus(self, actor_network, sample_data):
        """Test that entropy bonus is computed when enabled."""
        loss = GRPOLoss(actor_network=actor_network, entropy_bonus=True)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" in output.keys()
        assert "loss_entropy" in output.keys()
        assert output["entropy"].shape == torch.Size([])
        assert output["loss_entropy"].requires_grad

    def test_loss_without_entropy_bonus(self, actor_network, sample_data):
        """Test that entropy is not computed when disabled."""
        loss = GRPOLoss(actor_network=actor_network, entropy_bonus=False)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" not in output.keys()
        assert "loss_entropy" not in output.keys()

    def test_loss_backward_pass(self, actor_network, sample_data):
        """Test that gradients flow correctly through the loss."""
        loss = GRPOLoss(actor_network=actor_network)
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
        loss_mean = GRPOLoss(actor_network=actor_network, reduction="mean")
        output_mean = loss_mean(sample_data)
        assert output_mean["loss_objective"].shape == torch.Size([])

        # Sum reduction
        loss_sum = GRPOLoss(actor_network=actor_network, reduction="sum")
        output_sum = loss_sum(sample_data)
        assert output_sum["loss_objective"].shape == torch.Size([])

    def test_loss_with_different_batch_sizes(self, actor_network, obs_dim, action_dim):
        """Test loss computation with different batch sizes."""
        loss = GRPOLoss(actor_network=actor_network)

        for batch_size in [1, 5, 32]:
            data = TensorDict(
                {
                    "observation": torch.randn(batch_size, obs_dim),
                    "action": torch.randint(0, action_dim, (batch_size,)),
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
            assert "loss_objective" in output.keys()
            assert output["loss_objective"].requires_grad

    def test_loss_clipping_behavior(self, actor_network, sample_data):
        """Test that loss clipping works as expected."""
        loss = GRPOLoss(
            actor_network=actor_network,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        output = loss(sample_data)

        # Loss should be finite
        assert torch.isfinite(output["loss_objective"])

    def test_loss_advantage_computation(self, actor_network, sample_data):
        """Test that advantage is computed and logged correctly."""
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        # Advantage should be in the output for logging
        assert "advantage" in output.keys()
        assert output["advantage"].shape == torch.Size([])

    def test_loss_kl_approximation(self, actor_network, sample_data):
        """Test that KL approximation is computed and logged."""
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        # KL approximation should be in the output
        assert "kl_approx" in output.keys()
        assert output["kl_approx"].shape == torch.Size([])

    def test_entropy_coeff_scalar(self, actor_network):
        """Test entropy coefficient as scalar value."""
        loss = GRPOLoss(actor_network=actor_network, entropy_coeff=0.05)
        assert loss.entropy_coeff.item() == pytest.approx(0.05)

    def test_entropy_coeff_mapping(self, actor_network):
        """Test entropy coefficient as mapping for composite action spaces."""
        coeff_map = {"action_head_1": 0.01, "action_head_2": 0.02}
        loss = GRPOLoss(actor_network=actor_network, entropy_coeff=coeff_map)
        assert loss._entropy_coeff_map == coeff_map

    def test_deprecated_entropy_coef_warning(self, actor_network):
        """Test that deprecated entropy_coef parameter raises warning."""
        with pytest.warns(DeprecationWarning, match="entropy_coef.*deprecated"):
            loss = GRPOLoss(actor_network=actor_network, entropy_coef=0.05)
            assert loss.entropy_coeff.item() == pytest.approx(0.05)

    def test_entropy_coef_and_coeff_conflict(self, actor_network):
        """Test that using both entropy_coef and entropy_coeff raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            GRPOLoss(
                actor_network=actor_network,
                entropy_coeff=0.01,
                entropy_coef=0.02,
            )

    def test_functional_mode(self, actor_network):
        """Test that functional mode works correctly."""
        loss = GRPOLoss(actor_network=actor_network, functional=True)
        assert loss.functional is True
        assert loss.actor_network_params is not None

    def test_non_functional_mode(self, actor_network):
        """Test that non-functional mode works correctly."""
        loss = GRPOLoss(actor_network=actor_network, functional=False)
        assert loss.functional is False
        assert loss.actor_network_params is None

    def test_in_keys_property(self, actor_network):
        """Test that in_keys property returns correct keys."""
        loss = GRPOLoss(actor_network=actor_network)
        in_keys = loss.in_keys
        assert "observation" in in_keys
        assert "action" in in_keys

    def test_out_keys_property(self, actor_network):
        """Test that out_keys property returns correct keys."""
        loss_with_entropy = GRPOLoss(actor_network=actor_network, entropy_bonus=True)
        out_keys = loss_with_entropy.out_keys
        assert "loss_objective" in out_keys
        assert "entropy" in out_keys
        assert "loss_entropy" in out_keys

        loss_no_entropy = GRPOLoss(actor_network=actor_network, entropy_bonus=False)
        out_keys = loss_no_entropy.out_keys
        assert "loss_objective" in out_keys
        assert "entropy" not in out_keys
        assert "loss_entropy" not in out_keys

    def test_loss_with_zero_rewards(self, actor_network, sample_data):
        """Test loss computation with zero rewards."""
        sample_data["next", "reward"] = torch.zeros_like(sample_data["next", "reward"])
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_negative_rewards(self, actor_network, sample_data):
        """Test loss computation with negative rewards."""
        sample_data["next", "reward"] = -torch.abs(sample_data["next", "reward"])
        loss = GRPOLoss(actor_network=actor_network)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_large_epsilon(self, actor_network, sample_data):
        """Test loss with large clipping epsilon values."""
        loss = GRPOLoss(
            actor_network=actor_network,
            epsilon_low=1.0,
            epsilon_high=1.0,
        )
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_loss_with_small_epsilon(self, actor_network, sample_data):
        """Test loss with small clipping epsilon values."""
        loss = GRPOLoss(
            actor_network=actor_network,
            epsilon_low=0.01,
            epsilon_high=0.01,
        )
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_loss_samples_mc_entropy(self, actor_network, sample_data):
        """Test Monte Carlo entropy estimation with different sample counts."""
        loss = GRPOLoss(
            actor_network=actor_network,
            samples_mc_entropy=10,
            entropy_bonus=True,
        )
        output = loss(sample_data)
        assert "entropy" in output.keys()
        assert torch.isfinite(output["entropy"])

    def test_loss_log_explained_variance(self, actor_network):
        """Test log_explained_variance parameter."""
        loss = GRPOLoss(actor_network=actor_network, log_explained_variance=True)
        assert loss.log_explained_variance is True

        loss = GRPOLoss(actor_network=actor_network, log_explained_variance=False)
        assert loss.log_explained_variance is False

    def test_loss_with_cuda_if_available(self, actor_network, sample_data):
        """Test loss computation on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        loss = GRPOLoss(actor_network=actor_network.to(device))
        sample_data = sample_data.to(device)

        output = loss(sample_data)
        assert output["loss_objective"].device.type == "cuda"

    def test_reset_method(self, actor_network):
        """Test that reset method exists and works."""
        loss = GRPOLoss(actor_network=actor_network)
        loss.reset()  # Should not raise any errors
