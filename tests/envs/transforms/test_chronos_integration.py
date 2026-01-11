"""Integration tests for ChronosEmbeddingTransform with TorchTrade environments."""

import pytest
import torch
import pandas as pd
from unittest.mock import Mock, patch
from torchrl.envs import TransformedEnv, Compose, InitTracker, RewardSum
from tensordict import TensorDict

from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, get_timeframe_unit
from torchtrade.envs.transforms import ChronosEmbeddingTransform
from tests.envs.transforms.conftest import mock_chronos_pipeline


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

    def test_transform_with_seqlongonly_env(self, sample_ohlcv_df):
        """Test ChronosEmbeddingTransform with SeqLongOnlyEnv."""
        with mock_chronos_pipeline(embedding_dim=256):
            # Create base environment
            time_frames = [
                TimeFrame(t, get_timeframe_unit(f))
                for t, f in zip([1, 5], ["Min", "Min"])
            ]
            execute_on = TimeFrame(5, get_timeframe_unit("Min"))

            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=time_frames,
                window_sizes=[10, 8],
                execute_on=execute_on,
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
            td_next = env.step(td)

            assert "chronos_embedding_1min" in td_next.keys()

    def test_transform_multiple_timeframes(self, sample_ohlcv_df):
        """Test transforming multiple timeframe observations."""
        with mock_chronos_pipeline(embedding_dim=128):
            # Create environment with multiple timeframes
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1, 5, 15], ["Min", "Min", "Min"])],
                window_sizes=[10, 8, 6],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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

            td = env.reset()

            # Check all embeddings present
            assert "emb_1min" in td.keys()
            assert "emb_5min" in td.keys()
            assert "emb_15min" in td.keys()

            # Original keys deleted
            assert "market_data_1Minute_10" not in td.keys()
            assert "market_data_5Minute_8" not in td.keys()
            assert "market_data_15Minute_6" not in td.keys()

    def test_transform_no_delete_keys(self, sample_ohlcv_df):
        """Test transform with del_keys=False preserves original observations."""
        with mock_chronos_pipeline(embedding_dim=128):
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1, 5], ["Min", "Min"])],
                window_sizes=[10, 8],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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

            td = env.reset()

            # Both original and embedding should be present
            assert "market_data_1Minute_10" in td.keys()
            assert "chronos_embedding" in td.keys()

    def test_observation_spec_updated(self, sample_ohlcv_df):
        """Test that observation spec is correctly updated."""
        with mock_chronos_pipeline(embedding_dim=256):
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1], ["Min"])],
                window_sizes=[10],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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

    def test_concat_aggregation_spec(self, sample_ohlcv_df):
        """Test observation spec with concat aggregation."""
        with mock_chronos_pipeline(embedding_dim=128):
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1], ["Min"])],
                window_sizes=[10],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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
            # Sample test data has 5 basic OHLCV features (no technical indicators)
            emb_spec = obs_spec["chronos_embedding"]
            expected_dim = 128 * 5  # 128 encoder dim * 5 features
            assert emb_spec.shape == (expected_dim,)

    def test_rollout_with_transform(self, sample_ohlcv_df):
        """Test collecting rollout with Chronos transform."""
        with mock_chronos_pipeline(embedding_dim=128):
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1, 5], ["Min", "Min"])],
                window_sizes=[10, 8],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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
            td = env.reset()

            for _ in range(5):
                # Random action
                td["action"] = torch.randint(0, 3, (1,)).squeeze()
                td = env.step(td)

                # Check embedding present at each step
                assert "chronos_embedding" in td.keys()

                if td["done"].item():
                    break


@pytest.mark.integration
@pytest.mark.slow
class TestChronosEmbeddingPerformance:
    """Performance and memory tests."""

    def test_batched_parallel_envs(self, sample_ohlcv_df):
        """Test transform works with parallel environments (batched)."""
        with mock_chronos_pipeline(embedding_dim=128):
            config = SeqLongOnlyEnvConfig(
                symbol="BTC/USD",
                time_frames=[TimeFrame(t, get_timeframe_unit(f)) for t, f in zip([1], ["Min"])],
                window_sizes=[10],
                execute_on=TimeFrame(5, get_timeframe_unit("Min")),
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
            td = env.reset()

            # Create batched version manually
            td_batched = TensorDict({
                "market_data_1Minute_10": torch.randn(4, 10, 12),  # (batch=4, window, features)
                "market_data_5Minute_8": torch.randn(4, 8, 12),
                "account_state": torch.randn(4, 7),
            }, batch_size=[4])

            # Apply transform
            transform = env.transform
            td_out = transform(td_batched)

            # Check batched output
            assert "chronos_embedding" in td_out.keys()
            assert td_out["chronos_embedding"].shape == (4, 128)  # (batch, embedding_dim)
