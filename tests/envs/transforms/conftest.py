"""Shared fixtures and helpers for transform tests."""

from unittest.mock import Mock, patch
from contextlib import contextmanager


@contextmanager
def mock_chronos_pipeline(embedding_dim=128):
    """Context manager providing mocked Chronos pipeline.

    This eliminates duplicate mock setup code across all tests.

    Usage:
        with mock_chronos_pipeline(embedding_dim=256):
            transform = ChronosEmbeddingTransform(...)
            transform._init()
            # Test code here

    Args:
        embedding_dim: Dimension of encoder embeddings (default: 128)

    Yields:
        None (mock is set up in context)
    """
    with patch('chronos.ChronosPipeline') as mock_pipeline:
        # Setup mock encoder with configurable embedding dimension
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=embedding_dim)
        mock_encoder.to = Mock(return_value=mock_encoder)

        # Setup mock model
        mock_model = Mock()
        mock_model.encoder = mock_encoder

        # Setup mock pipeline instance
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()

        # Configure from_pretrained to return our mock
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        yield mock_pipeline
