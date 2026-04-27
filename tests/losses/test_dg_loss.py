"""Tests for DGLoss module."""

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

from torchtrade.losses import DGLoss


class TestDGLoss:

    OBS_DIM = 4
    ACTION_DIM = 3
    BATCH_SIZE = 10

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def actor_network(self):
        policy_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(self.OBS_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, self.ACTION_DIM),
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        )
        actor = ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        return ProbabilisticTensorDictSequential(policy_net, actor)

    @pytest.fixture
    def sample_data(self):
        return TensorDict(
            {
                "observation": torch.randn(self.BATCH_SIZE, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (self.BATCH_SIZE,)),
                "next": {
                    "reward": torch.randn(self.BATCH_SIZE, 1),
                },
            },
            batch_size=[self.BATCH_SIZE],
        )

    def test_missing_actor_raises(self):
        with pytest.raises(TypeError, match="Missing positional argument"):
            DGLoss(actor_network=None)

    @pytest.mark.parametrize("baseline", ["invalid", "expected"])
    def test_invalid_baseline_raises(self, actor_network, baseline):
        with pytest.raises(ValueError, match="baseline must be"):
            DGLoss(actor_network=actor_network, baseline=baseline)

    @pytest.mark.parametrize("eta", [0.0, -1.0])
    def test_invalid_eta_raises(self, actor_network, eta):
        with pytest.raises(ValueError, match="eta must be > 0"):
            DGLoss(actor_network=actor_network, eta=eta)

    def test_forward_outputs_and_gradients(self, actor_network, sample_data):
        """Forward pass produces expected keys, shapes, finite values, and flowing gradients."""
        loss = DGLoss(actor_network=actor_network)
        output = loss(sample_data)

        for key in ["loss_objective", "gate", "advantage", "entropy", "loss_entropy"]:
            assert key in output.keys(), f"missing key: {key}"
            assert output[key].shape == torch.Size([]), f"wrong shape for {key}"
            assert torch.isfinite(output[key]), f"non-finite {key}"

        assert output["loss_objective"].requires_grad
        assert output["loss_entropy"].requires_grad
        assert not output["gate"].requires_grad
        assert not output["advantage"].requires_grad

        (output["loss_objective"] + output["loss_entropy"]).backward()
        for param in loss.actor_network_params.values(True, True):
            assert param.grad is not None

    def test_no_entropy_bonus(self, actor_network, sample_data):
        """Disabling entropy removes entropy keys from output."""
        loss = DGLoss(actor_network=actor_network, entropy_bonus=False)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" not in output.keys()
        assert "loss_entropy" not in output.keys()

    def test_eta_sharpens_gate(self, actor_network, sample_data):
        """Smaller eta pushes gate further from 0.5 (sharper gating)."""
        gate_small_eta = DGLoss(actor_network=actor_network, eta=0.1)(sample_data)["gate"].item()
        gate_large_eta = DGLoss(actor_network=actor_network, eta=10.0)(sample_data)["gate"].item()

        # Both must be valid sigmoid outputs
        assert 0.0 <= gate_small_eta <= 1.0
        assert 0.0 <= gate_large_eta <= 1.0
        # Small eta = sharper = further from 0.5
        assert abs(gate_small_eta - 0.5) > abs(gate_large_eta - 0.5)

    @pytest.mark.parametrize("baseline", ["mean", "none"])
    def test_baseline_modes(self, actor_network, sample_data, baseline):
        loss = DGLoss(actor_network=actor_network, baseline=baseline)
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_batch_size_one(self, actor_network):
        """Edge case: single-element batch (mean baseline = value itself)."""
        data = TensorDict(
            {
                "observation": torch.randn(1, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (1,)),
                "next": {"reward": torch.randn(1, 1)},
            },
            batch_size=[1],
        )
        loss = DGLoss(actor_network=actor_network)
        output = loss(data)
        assert torch.isfinite(output["loss_objective"])

    def test_zero_rewards_gate_is_half(self, actor_network, sample_data):
        """Zero rewards => advantage=0 => delight=0 => gate=sigmoid(0)=0.5."""
        sample_data["next", "reward"] = torch.zeros(self.BATCH_SIZE, 1)
        loss = DGLoss(actor_network=actor_network)
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])
        assert output["gate"].item() == pytest.approx(0.5, abs=1e-5)

    def test_gate_suppresses_rare_failures_amplifies_rare_successes(self, actor_network):
        """Core DG property: positive advantage + high surprisal => gate > 0.5,
        negative advantage + high surprisal => gate < 0.5."""
        # Use a fixed policy where action=0 has high prob, action=2 has low prob
        with torch.no_grad():
            # Set logits so action 0 is very likely, action 2 is very unlikely
            for module in actor_network.modules():
                if isinstance(module, nn.Linear) and module.out_features == self.ACTION_DIM:
                    module.weight.zero_()
                    module.bias.copy_(torch.tensor([5.0, 0.0, -5.0]))

        # Rare action (action=2) with positive reward => should be amplified (gate > 0.5)
        data_rare_success = TensorDict(
            {
                "observation": torch.zeros(1, self.OBS_DIM),
                "action": torch.tensor([2]),  # rare action
                "next": {"reward": torch.tensor([[10.0]])},  # positive
            },
            batch_size=[1],
        )
        loss = DGLoss(actor_network=actor_network, baseline="none")
        gate_rare_success = loss(data_rare_success)["gate"].item()

        # Rare action (action=2) with negative reward => should be suppressed (gate < 0.5)
        data_rare_failure = TensorDict(
            {
                "observation": torch.zeros(1, self.OBS_DIM),
                "action": torch.tensor([2]),  # rare action
                "next": {"reward": torch.tensor([[-10.0]])},  # negative
            },
            batch_size=[1],
        )
        gate_rare_failure = loss(data_rare_failure)["gate"].item()

        assert gate_rare_success > 0.5, f"rare success should be amplified, got gate={gate_rare_success}"
        assert gate_rare_failure < 0.5, f"rare failure should be suppressed, got gate={gate_rare_failure}"
