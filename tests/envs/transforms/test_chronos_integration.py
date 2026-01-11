"""Integration tests for ChronosEmbeddingTransform with TorchTrade environments."""

import pytest
import torch
import pandas as pd
from unittest.mock import Mock, patch
from torchrl.envs import TransformedEnv, Compose, InitTracker, RewardSum
from tensordict import TensorDict

from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.transforms import ChronosEmbeddingTransform


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100.0 + pd.Series(range(1000)) * 0.01,
        "high": 101.0 + pd.Series(range(1000)) * 0.01,
        "low": 99.0 + pd.Series(range(1000)) * 0.01,
        "close": 100.5 + pd.Series(range(1000)) * 0.01,
        "volume": 1000.0,
    })
    return df


@pytest.mark.integration
class TestChronosEmbeddingIntegration:
    """Integration tests with actual environments."""

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_transform_with_seqlongonly_env(self, mock_pipeline, sample_ohlcv_df):
        """Test ChronosEmbeddingTransform with SeqLongOnlyEnv."""
        # Setup mock Chronos
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=256)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Create base environment
        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1, 5],
            window_sizes=[10, 8],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        # Wrap with Chronos transform
        env = TransformedEnv(
            base_env,
            Compose(
                ChronosEmbeddingTransform(
                    in_keys=["market_data_1Minute_10"],
                    out_keys=["chronos_embedding_1min"],
                    aggregation="mean",
                    device="cpu"
                ),
                InitTracker(),
                RewardSum(),
            )
        )

        # Reset environment
        with pytest.warns(UserWarning, match="placeholder embeddings"):
            td = env.reset()

        # Check that embedding key is present
        assert "chronos_embedding_1min" in td.keys()
        # Original key should be deleted
        assert "market_data_1Minute_10" not in td.keys()
        # Other keys should still be present
        assert "market_data_5Minute_8" in td.keys()
        assert "account_state" in td.keys()

        # Take a step
        td["action"] = torch.tensor(1)  # HOLD
        with pytest.warns(UserWarning):
            td_next = env.step(td)

        assert "chronos_embedding_1min" in td_next.keys()

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_transform_multiple_timeframes(self, mock_pipeline, sample_ohlcv_df):
        """Test transforming multiple timeframe observations."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Create environment with multiple timeframes
        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1, 5, 15],
            window_sizes=[10, 8, 6],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        # Transform all timeframes
        env = TransformedEnv(
            base_env,
            Compose(
                ChronosEmbeddingTransform(
                    in_keys=[
                        "market_data_1Minute_10",
                        "market_data_5Minute_8",
                        "market_data_15Minute_6"
                    ],
                    out_keys=[
                        "emb_1min",
                        "emb_5min",
                        "emb_15min"
                    ],
                    aggregation="mean",
                    device="cpu"
                ),
                InitTracker(),
            )
        )

        with pytest.warns(UserWarning):
            td = env.reset()

        # Check all embeddings present
        assert "emb_1min" in td.keys()
        assert "emb_5min" in td.keys()
        assert "emb_15min" in td.keys()

        # Original keys deleted
        assert "market_data_1Minute_10" not in td.keys()
        assert "market_data_5Minute_8" not in td.keys()
        assert "market_data_15Minute_6" not in td.keys()

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_transform_no_delete_keys(self, mock_pipeline, sample_ohlcv_df):
        """Test transform with del_keys=False preserves original observations."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1, 5],
            window_sizes=[10, 8],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        env = TransformedEnv(
            base_env,
            Compose(
                ChronosEmbeddingTransform(
                    in_keys=["market_data_1Minute_10"],
                    out_keys=["chronos_embedding"],
                    del_keys=False,  # Keep original
                    device="cpu"
                ),
                InitTracker(),
            )
        )

        with pytest.warns(UserWarning):
            td = env.reset()

        # Both original and embedding should be present
        assert "market_data_1Minute_10" in td.keys()
        assert "chronos_embedding" in td.keys()

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_observation_spec_updated(self, mock_pipeline, sample_ohlcv_df):
        """Test that observation spec is correctly updated."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=256)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1],
            window_sizes=[10],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        env = TransformedEnv(
            base_env,
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_10"],
                out_keys=["chronos_embedding"],
                aggregation="mean",
                device="cpu"
            )
        )

        # Get observation spec
        obs_spec = env.observation_spec

        # Check embedding spec exists
        assert "chronos_embedding" in obs_spec.keys()
        # Original spec should be deleted
        assert "market_data_1Minute_10" not in obs_spec.keys()

        # Check embedding shape
        emb_spec = obs_spec["chronos_embedding"]
        assert emb_spec.shape == (256,)  # Mean aggregation

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_concat_aggregation_spec(self, mock_pipeline, sample_ohlcv_df):
        """Test observation spec with concat aggregation."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1],
            window_sizes=[10],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        env = TransformedEnv(
            base_env,
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_10"],
                out_keys=["chronos_embedding"],
                aggregation="concat",  # Concat all features
                device="cpu"
            )
        )

        obs_spec = env.observation_spec

        # With concat, embedding dim = encoder_dim * num_features
        # SeqLongOnlyEnv default features: 12 (OHLCV + 7 technical indicators)
        emb_spec = obs_spec["chronos_embedding"]
        expected_dim = 128 * 12  # 128 encoder dim * 12 features
        assert emb_spec.shape == (expected_dim,)

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_rollout_with_transform(self, mock_pipeline, sample_ohlcv_df):
        """Test collecting rollout with Chronos transform."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1, 5],
            window_sizes=[10, 8],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        env = TransformedEnv(
            base_env,
            Compose(
                ChronosEmbeddingTransform(
                    in_keys=["market_data_1Minute_10"],
                    out_keys=["chronos_embedding"],
                    device="cpu"
                ),
                InitTracker(),
                RewardSum(),
            )
        )

        # Collect short rollout
        with pytest.warns(UserWarning):
            td = env.reset()

        for _ in range(5):
            # Random action
            td["action"] = torch.randint(0, 3, (1,)).squeeze()

            with pytest.warns(UserWarning):
                td = env.step(td)

            # Check embedding present at each step
            assert "chronos_embedding" in td.keys()

            if td["done"].item():
                break


@pytest.mark.integration
@pytest.mark.slow
class TestChronosEmbeddingPerformance:
    """Performance and memory tests."""

    @patch('torchtrade.envs.transforms.chronos_embedding.ChronosPipeline')
    def test_batched_parallel_envs(self, mock_pipeline, sample_ohlcv_df):
        """Test transform works with parallel environments (batched)."""
        # Setup mock
        mock_model = Mock()
        mock_encoder = Mock()
        mock_encoder.config = Mock(d_model=128)
        mock_model.encoder = mock_encoder
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.model = mock_model
        mock_pipeline_instance.tokenizer = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        config = SeqLongOnlyEnvConfig(
            symbol="BTC/USD",
            time_frames=[1],
            window_sizes=[10],
            execute_on=(5, "Minute"),
            initial_cash=1000,
        )
        base_env = SeqLongOnlyEnv(sample_ohlcv_df, config)

        # Wrap with transform
        env = TransformedEnv(
            base_env,
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_10"],
                out_keys=["chronos_embedding"],
                device="cpu"
            )
        )

        # Simulate batched observations (like from ParallelEnv)
        with pytest.warns(UserWarning):
            td = env.reset()

        # Create batched version manually
        td_batched = TensorDict({
            "market_data_1Minute_10": torch.randn(4, 10, 12),  # (batch=4, window, features)
            "market_data_5Minute_8": torch.randn(4, 8, 12),
            "account_state": torch.randn(4, 7),
        }, batch_size=[4])

        # Apply transform
        transform = env.transform
        with pytest.warns(UserWarning):
            td_out = transform(td_batched)

        # Check batched output
        assert "chronos_embedding" in td_out.keys()
        assert td_out["chronos_embedding"].shape == (4, 128)  # (batch, embedding_dim)
