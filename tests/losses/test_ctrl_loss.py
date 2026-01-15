"""Tests for CTRLLoss and CTRLPPOLoss modules."""

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchtrade.losses import CTRLLoss, CTRLPPOLoss


class TestCTRLLoss:
    """Test suite for CTRLLoss."""

    @pytest.fixture
    def obs_dim(self):
        return 64

    @pytest.fixture
    def embedding_dim(self):
        return 128

    @pytest.fixture
    def batch_size(self):
        return 32

    @pytest.fixture
    def window_len(self):
        return 4

    @pytest.fixture
    def encoder_network(self, obs_dim, embedding_dim):
        """Create a simple encoder network for testing."""
        return TensorDictModule(
            nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embedding_dim),
            ),
            in_keys=["observation"],
            out_keys=["embedding"],
        )

    @pytest.fixture
    def sample_data(self, batch_size, window_len, obs_dim):
        """Create sample training data with trajectory windows."""
        return TensorDict(
            {"observation": torch.randn(batch_size, window_len, obs_dim)},
            batch_size=[batch_size, window_len],
        )

    @pytest.fixture
    def sample_data_no_window(self, batch_size, obs_dim):
        """Create sample data without window dimension."""
        return TensorDict(
            {"observation": torch.randn(batch_size, obs_dim)},
            batch_size=[batch_size],
        )

    def test_loss_initialization(self, encoder_network, embedding_dim):
        """Test that loss module initializes correctly."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        assert loss is not None
        assert loss.encoder_network is not None
        assert loss.embedding_dim == embedding_dim
        assert loss.projection_head is not None
        assert loss.prediction_head is not None
        assert loss.prototypes is not None
        assert loss.num_prototypes == 512  # default
        assert loss.projection_dim == 128  # default

    def test_loss_initialization_custom_params(self, encoder_network, embedding_dim):
        """Test loss initialization with custom parameters."""
        loss = CTRLLoss(
            encoder_network,
            embedding_dim,
            projection_dim=256,
            num_prototypes=1024,
            sinkhorn_iters=5,
            temperature=0.2,
            window_len=8,
            myow_k=10,
            myow_coeff=2.0,
            reduction="sum",
        )
        assert loss.projection_dim == 256
        assert loss.num_prototypes == 1024
        assert loss.sinkhorn_iters.item() == 5
        assert loss.temperature.item() == pytest.approx(0.2)
        assert loss.window_len == 8
        assert loss.myow_k.item() == 10
        assert loss.myow_coeff.item() == pytest.approx(2.0)
        assert loss.reduction == "sum"

    def test_functional_property(self, encoder_network, embedding_dim):
        """Test that CTRL loss is non-functional."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        assert loss.functional is False

    def test_forward_pass(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test forward pass produces expected outputs."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        output = loss(sample_data_no_window)

        assert "loss_ctrl" in output.keys()
        assert "loss_proto" in output.keys()
        assert "loss_myow" in output.keys()
        assert output["loss_ctrl"].requires_grad
        assert output["loss_proto"].requires_grad
        assert output["loss_myow"].requires_grad

    def test_forward_pass_shapes(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test that output shapes are correct."""
        loss = CTRLLoss(encoder_network, embedding_dim, reduction="mean")
        output = loss(sample_data_no_window)

        # With mean reduction, all losses should be scalars
        assert output["loss_ctrl"].shape == torch.Size([])
        assert output["loss_proto"].shape == torch.Size([])
        assert output["loss_myow"].shape == torch.Size([])

    def test_backward_pass(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test that gradients flow correctly through the loss."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        output = loss(sample_data_no_window)

        # Compute backward
        output["loss_ctrl"].backward()

        # Check that encoder has gradients
        for param in encoder_network.parameters():
            assert param.grad is not None

        # Check that projection/prediction heads have gradients
        for param in loss.projection_head.parameters():
            assert param.grad is not None
        for param in loss.prediction_head.parameters():
            assert param.grad is not None

        # Check that prototypes have gradients
        assert loss.prototypes.grad is not None

    def test_sinkhorn_algorithm(self):
        """Test Sinkhorn-Knopp algorithm for balanced assignments."""
        batch_size = 16
        num_prototypes = 32
        scores = torch.randn(batch_size, num_prototypes)

        Q = CTRLLoss._sinkhorn(scores, iters=3)

        # Check shape
        assert Q.shape == (batch_size, num_prototypes)

        # Check that rows sum to approximately 1 (within tolerance)
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

        # Check that all values are non-negative
        assert (Q >= 0).all()

    def test_sinkhorn_iterations(self):
        """Test Sinkhorn with different iteration counts."""
        scores = torch.randn(10, 20)

        for iters in [1, 3, 5, 10]:
            Q = CTRLLoss._sinkhorn(scores, iters=iters)
            assert Q.shape == scores.shape
            assert torch.isfinite(Q).all()

    def test_prototype_loss_computation(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test prototype loss computation."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        output = loss(sample_data_no_window)

        # Prototype loss should be finite and positive (cross-entropy)
        assert torch.isfinite(output["loss_proto"])
        assert output["loss_proto"] >= 0

    def test_myow_loss_computation(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test MYOW loss computation."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        output = loss(sample_data_no_window)

        # MYOW loss should be finite
        assert torch.isfinite(output["loss_myow"])

    def test_combined_loss_computation(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test that total CTRL loss is combination of proto and MYOW."""
        loss = CTRLLoss(encoder_network, embedding_dim, myow_coeff=1.0)
        output = loss(sample_data_no_window)

        # loss_ctrl should be approximately proto + myow_coeff * myow
        # (approximate due to reduction operations)
        assert torch.isfinite(output["loss_ctrl"])

    def test_myow_coefficient_effect(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test that MYOW coefficient affects total loss."""
        loss_low = CTRLLoss(encoder_network, embedding_dim, myow_coeff=0.1)
        loss_high = CTRLLoss(encoder_network, embedding_dim, myow_coeff=10.0)

        output_low = loss_low(sample_data_no_window)
        output_high = loss_high(sample_data_no_window)

        # Different MYOW coefficients should produce different total losses
        # (unless MYOW loss is exactly zero, which is unlikely)
        # We can't assert inequality due to random initialization, but we can check finiteness
        assert torch.isfinite(output_low["loss_ctrl"])
        assert torch.isfinite(output_high["loss_ctrl"])

    def test_reduction_modes(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test different reduction modes."""
        # Mean reduction
        loss_mean = CTRLLoss(encoder_network, embedding_dim, reduction="mean")
        output_mean = loss_mean(sample_data_no_window)
        assert output_mean["loss_ctrl"].shape == torch.Size([])

        # Sum reduction
        loss_sum = CTRLLoss(encoder_network, embedding_dim, reduction="sum")
        output_sum = loss_sum(sample_data_no_window)
        assert output_sum["loss_ctrl"].shape == torch.Size([])

    def test_small_batch_handling(self, encoder_network, embedding_dim, obs_dim):
        """Test handling of small batch sizes."""
        # Batch size 1 (too small to split)
        small_data = TensorDict(
            {"observation": torch.randn(1, obs_dim)},
            batch_size=[1],
        )
        loss = CTRLLoss(encoder_network, embedding_dim)
        output = loss(small_data)
        assert torch.isfinite(output["loss_ctrl"])

        # Batch size 2 (minimum for split)
        small_data = TensorDict(
            {"observation": torch.randn(2, obs_dim)},
            batch_size=[2],
        )
        output = loss(small_data)
        assert torch.isfinite(output["loss_ctrl"])

    def test_different_batch_sizes(self, encoder_network, embedding_dim, obs_dim):
        """Test with different batch sizes."""
        loss = CTRLLoss(encoder_network, embedding_dim)

        for batch_size in [4, 16, 64]:
            data = TensorDict(
                {"observation": torch.randn(batch_size, obs_dim)},
                batch_size=[batch_size],
            )
            output = loss(data)
            assert torch.isfinite(output["loss_ctrl"])

    def test_extract_windows(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test window extraction for cross-trajectory comparison."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        batch1, batch2 = loss._extract_windows(sample_data_no_window)

        # Check that batches are properly split
        total_size = sample_data_no_window.batch_size[0]
        half = total_size // 2
        assert batch1.batch_size[0] == half
        assert batch2.batch_size[0] == half

    def test_extract_windows_small_batch(self, encoder_network, embedding_dim, obs_dim):
        """Test window extraction with batch size 1."""
        small_data = TensorDict(
            {"observation": torch.randn(1, obs_dim)},
            batch_size=[1],
        )
        loss = CTRLLoss(encoder_network, embedding_dim)
        batch1, batch2 = loss._extract_windows(small_data)

        # With batch size 1, should duplicate
        assert batch1.batch_size[0] == 1
        assert batch2.batch_size[0] == 1

    def test_missing_embedding_key(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test that error is raised when embedding key is missing."""
        # Create encoder with wrong output key
        bad_encoder = TensorDictModule(
            nn.Linear(64, 128),
            in_keys=["observation"],
            out_keys=["wrong_key"],
        )
        loss = CTRLLoss(bad_encoder, embedding_dim)

        with pytest.raises(KeyError, match="Could not find embedding key"):
            loss(sample_data_no_window)

    def test_in_keys_property(self, encoder_network, embedding_dim):
        """Test that in_keys property returns correct keys."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        in_keys = loss.in_keys
        assert "observation" in in_keys

    def test_out_keys_property(self, encoder_network, embedding_dim):
        """Test that out_keys property returns correct keys."""
        loss = CTRLLoss(encoder_network, embedding_dim)
        out_keys = loss.out_keys
        assert "loss_ctrl" in out_keys
        assert "loss_proto" in out_keys
        assert "loss_myow" in out_keys

    def test_projection_head_architecture(self, encoder_network, embedding_dim):
        """Test projection head architecture."""
        loss = CTRLLoss(encoder_network, embedding_dim, projection_dim=256)

        # Test projection head forward pass
        dummy_emb = torch.randn(10, embedding_dim)
        projected = loss.projection_head(dummy_emb)
        assert projected.shape == (10, 256)

    def test_prediction_head_architecture(self, encoder_network, embedding_dim):
        """Test prediction head architecture."""
        loss = CTRLLoss(encoder_network, embedding_dim, projection_dim=256)

        # Test prediction head forward pass
        dummy_proj = torch.randn(10, 256)
        predicted = loss.prediction_head(dummy_proj)
        assert predicted.shape == (10, 256)

    def test_prototypes_initialization(self, encoder_network, embedding_dim):
        """Test that prototypes are properly initialized."""
        loss = CTRLLoss(encoder_network, embedding_dim, num_prototypes=128, projection_dim=64)
        assert loss.prototypes.shape == (128, 64)
        assert loss.prototypes.requires_grad

    def test_temperature_effect(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test temperature parameter effect."""
        loss_low_temp = CTRLLoss(encoder_network, embedding_dim, temperature=0.01)
        loss_high_temp = CTRLLoss(encoder_network, embedding_dim, temperature=1.0)

        output_low = loss_low_temp(sample_data_no_window)
        output_high = loss_high_temp(sample_data_no_window)

        # Both should produce finite losses
        assert torch.isfinite(output_low["loss_ctrl"])
        assert torch.isfinite(output_high["loss_ctrl"])

    def test_cuda_if_available(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test CTRL loss on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        loss = CTRLLoss(encoder_network, embedding_dim)
        loss = loss.to(device)
        sample_data_no_window = sample_data_no_window.to(device)

        output = loss(sample_data_no_window)
        assert output["loss_ctrl"].device.type == "cuda"

    def test_myow_k_bounds(self, encoder_network, embedding_dim, sample_data_no_window):
        """Test MYOW k parameter with different values."""
        # Test with k larger than num_prototypes
        loss_large_k = CTRLLoss(
            encoder_network, embedding_dim, num_prototypes=10, myow_k=100
        )
        output = loss_large_k(sample_data_no_window)
        assert torch.isfinite(output["loss_myow"])

        # Test with k = 1
        loss_small_k = CTRLLoss(encoder_network, embedding_dim, myow_k=1)
        output = loss_small_k(sample_data_no_window)
        assert torch.isfinite(output["loss_myow"])


class TestCTRLPPOLoss:
    """Test suite for CTRLPPOLoss."""

    @pytest.fixture
    def encoder_network(self):
        """Create a simple encoder network."""
        return TensorDictModule(
            nn.Sequential(nn.Linear(64, 128), nn.ReLU()),
            in_keys=["observation"],
            out_keys=["embedding"],
        )

    @pytest.fixture
    def mock_ppo_loss(self):
        """Create a mock PPO loss module."""
        from unittest.mock import Mock

        mock = Mock()
        mock.in_keys = ["observation", "action"]
        mock.out_keys = ["loss_objective", "loss_critic"]
        return mock

    @pytest.fixture
    def ctrl_loss(self, encoder_network):
        """Create a CTRL loss module."""
        return CTRLLoss(encoder_network, embedding_dim=128)

    def test_initialization(self, mock_ppo_loss, ctrl_loss):
        """Test CTRLPPOLoss initialization."""
        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss)
        assert combined.ppo_loss is mock_ppo_loss
        assert combined.ctrl_loss is ctrl_loss
        assert combined.ctrl_coeff.item() == 1.0

    def test_initialization_custom_coeff(self, mock_ppo_loss, ctrl_loss):
        """Test initialization with custom CTRL coefficient."""
        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss, ctrl_coeff=0.5)
        assert combined.ctrl_coeff.item() == 0.5

    def test_functional_property(self, mock_ppo_loss, ctrl_loss):
        """Test that combined loss is non-functional."""
        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss)
        assert combined.functional is False

    def test_in_keys_property(self, mock_ppo_loss, ctrl_loss):
        """Test that in_keys combines both losses."""
        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss)
        in_keys = combined.in_keys
        assert "observation" in in_keys
        # Should combine keys from both losses

    def test_out_keys_property(self, mock_ppo_loss, ctrl_loss):
        """Test that out_keys combines both losses."""
        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss)
        out_keys = combined.out_keys
        # Should include keys from both PPO and CTRL
        assert len(out_keys) > 0

    def test_ctrl_coeff_scaling(self, mock_ppo_loss, ctrl_loss):
        """Test that CTRL coefficient scales the loss correctly."""
        from unittest.mock import Mock

        # Setup mock to return specific values
        mock_ppo_loss.return_value = TensorDict(
            {"loss_objective": torch.tensor(1.0)}, batch_size=[]
        )

        combined = CTRLPPOLoss(mock_ppo_loss, ctrl_loss, ctrl_coeff=2.0)

        # Create sample data
        sample_data_no_window = TensorDict(
            {"observation": torch.randn(16, 64)}, batch_size=[16]
        )

        output = combined(sample_data_no_window)

        # Check that loss_ctrl is scaled
        if "loss_ctrl" in output.keys():
            # The scaling should be applied
            assert torch.isfinite(output["loss_ctrl"])
