"""Tests for ChronosEmbeddingTransform."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec

from torchtrade.envs.transforms import ChronosEmbeddingTransform


# Patch path for ChronosPipeline (imported lazily in _init)
CHRONOS_PIPELINE_PATCH = 'chronos.ChronosPipeline'


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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_lazy_init_not_called_on_creation(self, mock_pipeline):
        """Test that model is not loaded during __init__."""
        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"]
        )

        # Pipeline should not be loaded yet
        assert not transform._initialized
        assert transform.pipeline is None
        mock_pipeline.from_pretrained.assert_not_called()

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_lazy_init_called_on_first_use(self, mock_pipeline):
        """Test that model is loaded on first _init() call."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=1024)
        mock_model.encoder = mock_encoder

        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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
        mock_pipeline.from_pretrained.assert_called_once_with(
            "amazon/chronos-t5-large",
            device_map="cpu",
            torch_dtype=torch.bfloat16
        )

    def test_lazy_init_import_error(self):
        """Test that missing chronos package raises ImportError."""
        # Hide chronos module
        import sys
        chronos_module = sys.modules.get('chronos')
        if chronos_module:
            sys.modules['chronos'] = None

        try:
            transform = ChronosEmbeddingTransform(
                in_keys=["market_data"],
                out_keys=["embedding"]
            )

            with pytest.raises(ImportError, match="chronos-forecasting package required"):
                transform._init()
        finally:
            # Restore chronos module if it existed
            if chronos_module:
                sys.modules['chronos'] = chronos_module


class TestChronosEmbeddingTransformApplyTransform:
    """Test _apply_transform method."""

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_apply_transform_1d_input(self, mock_pipeline):
        """Test transformation of 1D time series."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            device="cpu"
        )

        # Initialize
        transform._init()

        # Test 1D input
        obs = torch.randn(10)  # (window_size,)

        with pytest.warns(UserWarning, match="placeholder embeddings"):
            embedding = transform._apply_transform(obs)

        assert embedding.shape == (128,)  # (embedding_dim,)

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_apply_transform_2d_input_mean(self, mock_pipeline):
        """Test transformation of 2D multi-feature input with mean aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            aggregation="mean",
            device="cpu"
        )

        transform._init()

        # Test 2D input: (window_size, num_features)
        obs = torch.randn(10, 5)

        with pytest.warns(UserWarning):
            embedding = transform._apply_transform(obs)

        assert embedding.shape == (128,)  # Mean over features

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_apply_transform_2d_input_max(self, mock_pipeline):
        """Test transformation of 2D input with max aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            aggregation="max",
            device="cpu"
        )

        transform._init()

        obs = torch.randn(10, 5)

        with pytest.warns(UserWarning):
            embedding = transform._apply_transform(obs)

        assert embedding.shape == (128,)  # Max over features

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_apply_transform_2d_input_concat(self, mock_pipeline):
        """Test transformation of 2D input with concat aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            aggregation="concat",
            device="cpu"
        )

        transform._init()

        obs = torch.randn(10, 5)  # 5 features

        with pytest.warns(UserWarning):
            embedding = transform._apply_transform(obs)

        assert embedding.shape == (128 * 5,)  # Concat all features

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_apply_transform_invalid_shape(self, mock_pipeline):
        """Test that 3D+ input raises error."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            device="cpu"
        )

        transform._init()

        obs = torch.randn(2, 10, 5)  # 3D invalid

        with pytest.raises(ValueError, match="Expected obs to be 1D or 2D"):
            transform._apply_transform(obs)


class TestChronosEmbeddingTransformCall:
    """Test _call method (forward pass with tensordict)."""

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_call_unbatched(self, mock_pipeline):
        """Test forward pass with unbatched observation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

        with pytest.warns(UserWarning):
            td_out = transform._call(td)

        assert "embedding" in td_out.keys()
        assert td_out["embedding"].shape == (128,)
        assert "other_key" in td_out.keys()  # Unchanged

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_call_batched(self, mock_pipeline):
        """Test forward pass with batched observations."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

        with pytest.warns(UserWarning):
            td_out = transform._call(td)

        assert "embedding" in td_out.keys()
        assert td_out["embedding"].shape == (4, 128)  # (batch, embedding_dim)

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_call_del_keys(self, mock_pipeline):
        """Test that del_keys removes input keys."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            del_keys=True,
            device="cpu"
        )

        td = TensorDict({
            "market_data": torch.randn(10, 5),
        }, batch_size=[])

        with pytest.warns(UserWarning):
            td_out = transform._call(td)

        assert "embedding" in td_out.keys()
        assert "market_data" not in td_out.keys()  # Deleted

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_call_no_del_keys(self, mock_pipeline):
        """Test that del_keys=False keeps input keys."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            del_keys=False,
            device="cpu"
        )

        td = TensorDict({
            "market_data": torch.randn(10, 5),
        }, batch_size=[])

        with pytest.warns(UserWarning):
            td_out = transform._call(td)

        assert "embedding" in td_out.keys()
        assert "market_data" in td_out.keys()  # Kept

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_call_missing_key(self, mock_pipeline):
        """Test that missing input key is skipped gracefully."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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


class TestChronosEmbeddingTransformObsSpec:
    """Test observation spec transformation."""

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_transform_observation_spec_mean(self, mock_pipeline):
        """Test spec transformation with mean aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=256)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_transform_observation_spec_concat(self, mock_pipeline):
        """Test spec transformation with concat aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=256)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_transform_observation_spec_no_del_keys(self, mock_pipeline):
        """Test that spec keeps input keys when del_keys=False."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=256)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_to_device(self, mock_pipeline):
        """Test moving transform to different device."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_encoder.to = Mock(return_value=mock_encoder)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        transform = ChronosEmbeddingTransform(
            in_keys=["market_data"],
            out_keys=["embedding"],
            device="cpu"
        )

        transform._init()

        # Move to different device
        transform.to("cpu")

        assert transform.device == torch.device("cpu")

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_to_dtype(self, mock_pipeline):
        """Test converting transform to different dtype."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_encoder.to = Mock(return_value=mock_encoder)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_spec_caching(self, mock_pipeline):
        """Test that spec transformation is cached."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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

    @patch(CHRONOS_PIPELINE_PATCH)
    def test_spec_cache_cleared_on_device_change(self, mock_pipeline):
        """Test that spec cache is cleared when device changes."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_encoder.to = Mock(return_value=mock_encoder)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

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
