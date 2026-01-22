"""Tests for rule-based trading actors."""

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from torchtrade.actor import MeanReversionActor
from torchtrade.envs.offline.infrastructure.utils import TimeFrame, TimeFrameUnit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def features_order():
    """Feature order for MeanReversionActor with all required features."""
    return [
        "close", "open", "high", "low", "volume",
        "features_bb_middle", "features_bb_std", "features_bb_upper",
        "features_bb_lower", "features_bb_position",
        "features_stoch_rsi_k", "features_stoch_rsi_d",
        "features_volume", "features_avg_volume"
    ]


def create_market_data_with_features(
    bb_position: float,
    stoch_k_now: float,
    stoch_d_now: float,
    stoch_k_prev: float,
    stoch_d_prev: float,
    volume_ratio: float = 1.0,
    window_size: int = 50
) -> torch.Tensor:
    """
    Create market data tensor with Bollinger Bands and Stochastic RSI features.

    Args:
        bb_position: Bollinger Band position (0=lower, 1=upper)
        stoch_k_now: Current Stochastic RSI %K value
        stoch_d_now: Current Stochastic RSI %D value
        stoch_k_prev: Previous Stochastic RSI %K value
        stoch_d_prev: Previous Stochastic RSI %D value
        volume_ratio: Ratio of current volume to average (1.5 = 50% above avg)
        window_size: Number of time steps

    Returns:
        Tensor of shape (window_size, num_features)
    """
    # Initialize base OHLCV
    close = torch.ones(window_size) * 100.0
    open_prices = torch.ones(window_size) * 99.5
    high = torch.ones(window_size) * 100.5
    low = torch.ones(window_size) * 99.0

    # Volume with average of 1000
    avg_volume = 1000.0
    volume = torch.ones(window_size) * avg_volume
    volume[-1] = avg_volume * volume_ratio  # Set last volume based on ratio

    # Bollinger Bands features
    bb_middle = torch.ones(window_size) * 100.0
    bb_std = torch.ones(window_size) * 2.0
    bb_upper = torch.ones(window_size) * 104.0  # middle + 2*std
    bb_lower = torch.ones(window_size) * 96.0   # middle - 2*std
    bb_position_tensor = torch.ones(window_size) * 0.5  # Default middle
    bb_position_tensor[-1] = bb_position  # Set last position

    # Stochastic RSI features (set last two values)
    stoch_k = torch.ones(window_size) * 50.0
    stoch_d = torch.ones(window_size) * 50.0
    stoch_k[-2] = stoch_k_prev
    stoch_d[-2] = stoch_d_prev
    stoch_k[-1] = stoch_k_now
    stoch_d[-1] = stoch_d_now

    # Volume features
    volume_feature = volume.clone()
    avg_volume_feature = torch.ones(window_size) * avg_volume

    # Stack all features: (window_size, 14)
    data = torch.stack([
        close, open_prices, high, low, volume,
        bb_middle, bb_std, bb_upper, bb_lower, bb_position_tensor,
        stoch_k, stoch_d, volume_feature, avg_volume_feature
    ], dim=1)

    return data


# ============================================================================
# Tests for MeanReversionActor
# ============================================================================


class TestMeanReversionActor:
    """Tests for MeanReversionActor with Bollinger Bands and Stochastic RSI."""

    def test_init(self):
        """Test actor initialization with new parameters."""
        actor = MeanReversionActor(
            bb_window=20,
            bb_std=2.0,
            stoch_rsi_window=14,
            oversold_threshold=20.0,
            overbought_threshold=80.0,
        )
        assert actor.bb_window == 20
        assert actor.bb_std == 2.0
        assert actor.stoch_rsi_window == 14
        assert actor.oversold_threshold == 20.0
        assert actor.overbought_threshold == 80.0

    def test_preprocessing_fn(self):
        """Test that preprocessing function computes all required features."""
        actor = MeanReversionActor(bb_window=20, stoch_rsi_window=14)
        preprocess_fn = actor.get_preprocessing_fn()

        # Create sample DataFrame
        df = pd.DataFrame({
            'close': [100 + i * 0.1 for i in range(100)],
            'open': [99.5 + i * 0.1 for i in range(100)],
            'high': [100.5 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)],
        })

        # Apply preprocessing
        df_processed = preprocess_fn(df)

        # Check that all required features are present
        required_features = [
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]
        for feat in required_features:
            assert feat in df_processed.columns, f"Missing feature: {feat}"

    def test_select_action_buy_signal(self):
        """Test BUY action when all conditions met: oversold + bullish crossover + volume."""
        features_order = [
            "close", "open", "high", "low", "volume",
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]

        # BUY conditions: bb_position < 0, bullish crossover from oversold, high volume
        data = create_market_data_with_features(
            bb_position=-0.1,      # Below lower BB
            stoch_k_now=25.0,      # K crosses above D
            stoch_d_now=20.0,
            stoch_k_prev=15.0,     # K was below D and oversold
            stoch_d_prev=18.0,
            volume_ratio=1.6       # 60% above average
        )

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),  # No position
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            features_order=features_order,
            execute_timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            oversold_threshold=20.0,
        )

        action = actor.select_action(obs)
        assert action == 2, "Expected BUY action when oversold with bullish crossover and volume"

    def test_select_action_sell_signal(self):
        """Test SELL action when all conditions met: overbought + bearish crossover + volume."""
        features_order = [
            "close", "open", "high", "low", "volume",
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]

        # SELL conditions: bb_position > 1, bearish crossover from overbought, high volume
        data = create_market_data_with_features(
            bb_position=1.1,       # Above upper BB
            stoch_k_now=75.0,      # K crosses below D
            stoch_d_now=80.0,
            stoch_k_prev=85.0,     # K was above D and overbought
            stoch_d_prev=82.0,
            volume_ratio=1.6       # 60% above average
        )

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),  # No position
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            features_order=features_order,
            execute_timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            overbought_threshold=80.0,
        )

        action = actor.select_action(obs)
        assert action == 0, "Expected SELL action when overbought with bearish crossover and volume"

    def test_select_action_hold_no_crossover(self):
        """Test HOLD action when no crossover occurs."""
        features_order = [
            "close", "open", "high", "low", "volume",
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]

        # Oversold but no crossover
        data = create_market_data_with_features(
            bb_position=-0.1,
            stoch_k_now=15.0,      # Both low but no crossover
            stoch_d_now=18.0,
            stoch_k_prev=14.0,
            stoch_d_prev=17.0,
            volume_ratio=1.6
        )

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            features_order=features_order,
            execute_timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        )

        action = actor.select_action(obs)
        assert action == 1, "Expected HOLD when no crossover"

    def test_select_action_hold_low_volume(self):
        """Test HOLD action when volume confirmation fails."""
        features_order = [
            "close", "open", "high", "low", "volume",
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]

        # Perfect setup but low volume
        data = create_market_data_with_features(
            bb_position=-0.1,
            stoch_k_now=25.0,
            stoch_d_now=20.0,
            stoch_k_prev=15.0,
            stoch_d_prev=18.0,
            volume_ratio=1.0       # Normal volume (needs 1.5x)
        )

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]),
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            features_order=features_order,
            execute_timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        )

        action = actor.select_action(obs)
        assert action == 1, "Expected HOLD when volume too low"

    def test_select_action_respects_position(self):
        """Test that actor respects existing positions (won't buy when long)."""
        features_order = [
            "close", "open", "high", "low", "volume",
            "features_bb_middle", "features_bb_std", "features_bb_upper",
            "features_bb_lower", "features_bb_position",
            "features_stoch_rsi_k", "features_stoch_rsi_d",
            "features_volume", "features_avg_volume"
        ]

        # Perfect BUY setup
        data = create_market_data_with_features(
            bb_position=-0.1,
            stoch_k_now=25.0,
            stoch_d_now=20.0,
            stoch_k_prev=15.0,
            stoch_d_prev=18.0,
            volume_ratio=1.6
        )

        # Already have a long position
        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.tensor([1000.0, 1.0, 100.0, 100.0, 100.0, 0.0, 0.0]),  # position_size > 0
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            features_order=features_order,
            execute_timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        )

        action = actor.select_action(obs)
        assert action == 1, "Expected HOLD when already long and buy signal appears"

