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
    import torch

    with patch('chronos.ChronosPipeline') as mock_pipeline:
        # Setup mock pipeline instance
        mock_pipeline_instance = Mock()

        # Mock the embed() method to return (embeddings, _) tuple
        # embeddings shape: (batch_size, seq_len, embedding_dim)
        def mock_embed(series):
            if isinstance(series, torch.Tensor):
                batch_size = 1
                seq_len = series.shape[0] if series.ndim == 1 else series.shape[1]
            else:
                batch_size = 1
                seq_len = 10

            embeddings = torch.randn(batch_size, seq_len, embedding_dim)
            return embeddings, None

        mock_pipeline_instance.embed = Mock(side_effect=mock_embed)

        # Configure from_pretrained to return our mock
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        yield mock_pipeline
