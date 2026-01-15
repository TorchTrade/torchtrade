"""Tests for ChronosEmbeddingTransform."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec

from torchtrade.envs.transforms import ChronosEmbeddingTransform
from tests.envs.transforms.conftest import mock_chronos_pipeline


class TestChronosEmbeddingTransformInit:
    """Test ChronosEmbeddingTransform initialization."""

    def test_init_basic(self):
        """Test basic initialization with default parameters."""
        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"]
        )

        assert transform.model_name == "amazon/chronos-t5-large"
        assert transform.aggregation == "mean"
        assert transform.del_keys is True
        assert not transform._initialized
        assert transform.pipeline is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        transform = ChronosEmbeddingTransform(
            in_keys=["obs1", "obs2"],
            out_keys=["emb1", "emb2"],
            model_name="amazon/chronos-t5-small",
            aggregation="concat",
            del_keys=False,
            device="cpu",
            torch_dtype=torch.float32
        )

        assert transform.model_name == "amazon/chronos-t5-small"
        assert transform.aggregation == "concat"
        assert transform.del_keys is False
        assert transform.device == torch.device("cpu")
        assert transform.torch_dtype == torch.float32

    def test_init_mismatched_keys(self):
        """Test that mismatched in_keys and out_keys raises error."""
        with pytest.raises(ValueError, match="in_keys and out_keys must have same length"):
            ChronosEmbeddingTransform(
                in_keys=["key1", "key2"],
                out_keys=["key1"]
            )

    def test_init_invalid_aggregation(self):
        """Test that invalid aggregation raises error."""
        with pytest.raises(ValueError, match="aggregation must be"):
            ChronosEmbeddingTransform(
                in_keys=["key1"],
                out_keys=["key1"],
                aggregation="invalid"
            )

    def test_device_auto_detection(self):
        """Test that device is auto-detected when not specified."""
        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"]
        )

        # Should be either cpu or cuda depending on availability
        assert transform.device in [torch.device("cpu"), torch.device("cuda")]


class TestChronosEmbeddingTransformLazyInit:
    """Test lazy initialization behavior."""

    def test_lazy_init_not_called_on_creation(self):
        """Test that model is not loaded during __init__."""
        with mock_chronos_pipeline() as mock_pipeline:
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"]
            )

            # Pipeline should not be loaded yet
            assert not transform._initialized
            assert transform.pipeline is None
            mock_pipeline.from_pretrained.assert_not_called()

    def test_lazy_init_called_on_first_use(self):
        """Test that model is loaded on first _init() call."""
        with mock_chronos_pipeline(embedding_dim=1024):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Call _init manually
            transform._init()

            # Check model was loaded
            assert transform._initialized
            assert transform.pipeline is not None
            assert transform.embedding_dim == 1024

    def test_lazy_init_import_error(self):
        """Test that missing chronos package raises ImportError."""
        # Mock the import to raise ImportError, simulating chronos not being installed
        def mock_import(name, *args, **kwargs):
            if name == 'chronos' or name.startswith('chronos.'):
                raise ImportError(f"No module named '{name}'")
            # Use original import for everything else
            return __import__(name, *args, **kwargs)

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"]
        )

        # Patch builtins.__import__ to simulate chronos not being installed
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="chronos-forecasting package required"):
                transform._init()


class TestChronosEmbeddingTransformApplyTransform:
    """Test _apply_transform method."""

    def test_apply_transform_1d_input(self):
        """Test transformation of 1D time series."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Initialize
            transform._init()

            # Test 1D input
            obs = torch.randn(10)  # (window_size,)
            embedding = transform._apply_transform(obs)

            assert embedding.shape == (128,)  # (embedding_dim,)

    def test_apply_transform_2d_input_mean(self):
        """Test transformation of 2D multi-feature input with mean aggregation."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                aggregation="mean",
                device="cpu"
            )

            transform._init()

            # Test 2D input: (window_size, num_features)
            obs = torch.randn(10, 5)
            embedding = transform._apply_transform(obs)

            assert embedding.shape == (128,)  # Mean over features

    def test_apply_transform_2d_input_max(self):
        """Test transformation of 2D input with max aggregation."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                aggregation="max",
                device="cpu"
            )

            transform._init()

            obs = torch.randn(10, 5)
            embedding = transform._apply_transform(obs)

            assert embedding.shape == (128,)  # Max over features

    def test_apply_transform_2d_input_concat(self):
        """Test transformation of 2D input with concat aggregation."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                aggregation="concat",
                device="cpu"
            )

            transform._init()

            obs = torch.randn(10, 5)  # 5 features
            embedding = transform._apply_transform(obs)

            assert embedding.shape == (128 * 5,)  # Concat all features

    def test_apply_transform_invalid_shape(self):
        """Test that 3D+ input raises error."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            transform._init()

            obs = torch.randn(2, 10, 5)  # 3D invalid

            with pytest.raises(ValueError, match="Expected 1D or 2D tensor"):
                transform._apply_transform(obs)


class TestChronosEmbeddingTransformCall:
    """Test _call method (forward pass with tensordict)."""

    def test_call_unbatched(self):
        """Test forward pass with unbatched observation."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Create unbatched tensordict
            td = TensorDict({
                "market_data": torch.randn(10, 5),  # (window, features)
                "other_key": torch.tensor([1.0])
            }, batch_size=[])

            td_out = transform._call(td)

            assert "embedding" in td_out.keys()
            assert td_out["embedding"].shape == (128,)
            assert "other_key" in td_out.keys()  # Unchanged

    def test_call_batched(self):
        """Test forward pass with batched observations."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Create batched tensordict (batch_size=4)
            td = TensorDict({
                "market_data": torch.randn(4, 10, 5),  # (batch, window, features)
                "other_key": torch.tensor([1.0, 2.0, 3.0, 4.0])
            }, batch_size=[4])

            td_out = transform._call(td)

            assert "embedding" in td_out.keys()
            assert td_out["embedding"].shape == (4, 128)  # (batch, embedding_dim)

    def test_call_del_keys(self):
        """Test that del_keys removes input keys."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                del_keys=True,
                device="cpu"
            )

            td = TensorDict({
                "market_data": torch.randn(10, 5),
            }, batch_size=[])

            td_out = transform._call(td)

            assert "embedding" in td_out.keys()
            assert "market_data" not in td_out.keys()  # Deleted

    def test_call_no_del_keys(self):
        """Test that del_keys=False keeps input keys."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                del_keys=False,
                device="cpu"
            )

            td = TensorDict({
                "market_data": torch.randn(10, 5),
            }, batch_size=[])

            td_out = transform._call(td)

            assert "embedding" in td_out.keys()
            assert "market_data" in td_out.keys()  # Kept

    def test_call_missing_key(self):
        """Test that missing input key is skipped gracefully."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            td = TensorDict({
                "other_key": torch.tensor([1.0]),
            }, batch_size=[])

            # Should not raise error
            td_out = transform._call(td)

            # Embedding should not be added
            assert "embedding" not in td_out.keys()


class TestChronosEmbeddingTransformReset:
    """Test _reset method."""

    def test_reset_unbatched(self):
        """Test _reset applies transform to reset observations."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Create reset tensordict
            td_reset = TensorDict({
                "market_data": torch.randn(10, 5),
                "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
            }, batch_size=[])

            # Empty tensordict for first arg (unused in _reset)
            td = TensorDict({}, batch_size=[])

            td_out = transform._reset(td, td_reset)

            assert "embedding" in td_out.keys()
            assert td_out["embedding"].shape == (128,)
            assert "market_data" not in td_out.keys()  # Deleted
            assert "account_state" in td_out.keys()  # Preserved

    def test_reset_batched(self):
        """Test _reset with batched reset observations."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            # Batched reset
            td_reset = TensorDict({
                "market_data": torch.randn(4, 10, 5),  # (batch=4, window, features)
                "account_state": torch.randn(4, 7)
            }, batch_size=[4])

            td = TensorDict({}, batch_size=[4])

            td_out = transform._reset(td, td_reset)

            assert "embedding" in td_out.keys()
            assert td_out["embedding"].shape == (4, 128)


class TestChronosEmbeddingTransformObsSpec:
    """Test observation spec transformation."""

    def test_transform_observation_spec_mean(self):
        """Test spec transformation with mean aggregation."""
        with mock_chronos_pipeline(embedding_dim=256):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                aggregation="mean",
                device="cpu"
            )

            # Create input spec
            input_spec = CompositeSpec(
                market_data=BoundedTensorSpec(
                    low=-10.0,
                    high=10.0,
                    shape=(10, 5),  # (window, features)
                    dtype=torch.float32
                )
            )

            # Transform spec
            output_spec = transform.transform_observation_spec(input_spec)

            assert "embedding" in output_spec.keys()
            assert output_spec["embedding"].shape == (256,)  # Mean aggregation
            assert "market_data" not in output_spec.keys()  # Deleted by default

    def test_transform_observation_spec_concat(self):
        """Test spec transformation with concat aggregation."""
        with mock_chronos_pipeline(embedding_dim=256):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                aggregation="concat",
                device="cpu"
            )

            input_spec = CompositeSpec(
                market_data=BoundedTensorSpec(
                    low=-10.0,
                    high=10.0,
                    shape=(10, 5),  # 5 features
                    dtype=torch.float32
                )
            )

            output_spec = transform.transform_observation_spec(input_spec)

            assert "embedding" in output_spec.keys()
            assert output_spec["embedding"].shape == (256 * 5,)  # Concat: features * dim

    def test_transform_observation_spec_no_del_keys(self):
        """Test that spec keeps input keys when del_keys=False."""
        with mock_chronos_pipeline(embedding_dim=256):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                del_keys=False,
                device="cpu"
            )

            input_spec = CompositeSpec(
                market_data=BoundedTensorSpec(
                    low=-10.0,
                    high=10.0,
                    shape=(10, 5),
                    dtype=torch.float32
                )
            )

            output_spec = transform.transform_observation_spec(input_spec)

            assert "embedding" in output_spec.keys()
            assert "market_data" in output_spec.keys()  # Kept


class TestChronosEmbeddingTransformDeviceManagement:
    """Test device and dtype management."""

    def test_to_device(self):
        """Test moving transform to different device."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            transform._init()

            # Move to different device
            transform.to("cpu")

            assert transform.device == torch.device("cpu")

    def test_to_dtype(self):
        """Test converting transform to different dtype."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            transform._init()

            # Change dtype
            transform.to(torch.float32)

            assert transform.torch_dtype == torch.float32


class TestChronosEmbeddingTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_in_out_keys(self):
        """Test transform with multiple input/output key pairs."""
        transform = ChronosEmbeddingTransform(
            in_keys=["obs1", "obs2"],
            out_keys=["emb1", "emb2"],
            device="cpu"
        )

        assert len(transform.in_keys) == 2
        assert len(transform.out_keys) == 2

    def test_spec_caching(self):
        """Test that spec transformation is cached."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            input_spec = CompositeSpec(
                market_data=BoundedTensorSpec(
                    low=-10.0,
                    high=10.0,
                    shape=(10, 5),
                    dtype=torch.float32
                )
            )

            # First call
            spec1 = transform.transform_observation_spec(input_spec)
            # Second call should return cached version
            spec2 = transform.transform_observation_spec(input_spec)

            assert spec1 is spec2  # Same object (cached)

    def test_spec_cache_cleared_on_device_change(self):
        """Test that spec cache is cleared when device changes."""
        with mock_chronos_pipeline(embedding_dim=128):
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"],
                device="cpu"
            )

            input_spec = CompositeSpec(
                market_data=BoundedTensorSpec(
                    low=-10.0,
                    high=10.0,
                    shape=(10, 5),
                    dtype=torch.float32
                )
            )

            # Cache spec
            transform.transform_observation_spec(input_spec)
            assert transform._transformed_spec is not None

            # Change device - should clear cache
            transform.to("cpu")
            assert transform._transformed_spec is None
