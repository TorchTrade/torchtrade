"""Shared observation-class tests.

Each exchange (alpaca, binance, bitget, bybit, okx) subclasses BaseObservationClassTests in
its own test_obs_class.py and implements create_observer / get_expected_symbol_format.
"""

import pytest
import numpy as np
from abc import ABC, abstractmethod
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


class BaseObservationClassTests(ABC):
    """
    Base test class for exchange observation classes.

    Tests observation fetching, preprocessing, and feature extraction.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """
        Create an observer instance for the specific exchange.

        Args:
            symbol: Trading symbol
            timeframes: Single TimeFrame or list of TimeFrames
            window_sizes: Single int or list of ints
            **kwargs: Exchange-specific parameters

        Returns:
            Observer instance
        """
        pass

    @abstractmethod
    def get_expected_symbol_format(self, symbol):
        """
        Get the expected symbol format for this exchange.

        Args:
            symbol: Input symbol (e.g., "BTC/USD", "BTCUSDT")

        Returns:
            Expected normalized symbol format
        """
        pass

    # Initialization tests

    def test_init_single_timeframe(self):
        """Test initialization with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=20,
        )

        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames', None))
        window_sizes = observer.window_sizes

        assert len(timeframes) == 1
        assert timeframes[0].value == 15
        assert window_sizes[0] == 20

    def test_init_multiple_timeframes(self):
        """Test initialization with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 30],
        )

        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames', None))
        assert len(timeframes) == 3
        assert len(observer.window_sizes) == 3

    def test_init_mismatched_lengths_raises_error(self):
        """Test that mismatched timeframes and window_sizes raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.create_observer(
                symbol="BTC/USD",
                timeframes=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 20, 30],  # 3 sizes for 2 timeframes
            )

    # get_keys tests

    def test_get_keys_single_timeframe(self):
        """Test get_keys with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        keys = observer.get_keys()
        assert len(keys) == 1
        assert "15Minute_10" in keys[0] or "15m_10" in keys[0].lower()

    def test_get_keys_multiple_timeframes(self):
        """Test get_keys with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20],
        )

        keys = observer.get_keys()
        assert len(keys) == 2

    # get_observations tests

    def test_get_observations_single_timeframe(self):
        """Test getting observations for single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()

        assert isinstance(observations, dict)
        assert len(observations) >= 1  # At least one key

        # Check first observation
        key = observer.get_keys()[0]
        assert key in observations
        assert isinstance(observations[key], np.ndarray)
        assert observations[key].shape[0] == 10  # window_size
        assert observations[key].shape[1] >= 4  # At least 4 features (OHLC-based)

    def test_get_observations_multiple_timeframes(self):
        """Test getting observations for multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 20],
        )

        observations = observer.get_observations()

        assert len(observations) >= 2
        keys = observer.get_keys()
        assert observations[keys[0]].shape == (10, 4) or observations[keys[0]].shape[0] == 10
        assert observations[keys[1]].shape == (20, 4) or observations[keys[1]].shape[0] == 20

    def test_get_observations_with_base_ohlc(self):
        """Test getting observations with base OHLC data."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations(return_base_ohlc=True)

        assert "base_features" in observations
        assert observations["base_features"].shape[1] == 4  # OHLC

    def test_observations_are_float32(self):
        """Test that observations are float32."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].dtype == np.float32

    def test_observations_no_nan_values(self):
        """Test that observations don't contain NaN values."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert not np.isnan(observations[key]).any()

    # get_features tests

    def test_get_features_default_preprocessing(self):
        """Test get_features with default preprocessing."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        features = observer.get_features()

        assert "observation_features" in features
        assert "original_features" in features
        assert len(features["observation_features"]) >= 4  # At least OHLC features

    # Custom preprocessing tests

    def test_custom_preprocessing(self):
        """Test with custom preprocessing function."""
        def custom_preprocessing(df):
            df = df.copy()
            df.dropna(inplace=True)
            df["feature_volatility"] = df["high"] - df["low"]
            df["feature_volume_ma"] = df["volume"].rolling(window=3).mean()
            df.dropna(inplace=True)
            return df

        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocessing,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        # Custom preprocessing has 2 features
        assert observations[key].shape[1] == 2

    # Edge cases

    def test_window_size_one(self):
        """Test with window size of 1."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=1,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 1

    def test_large_window_size(self):
        """Test with large window size."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=100,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 100



def assert_a_direct_flip_does_not_age_the_new_position(
    env, trader, PositionStatus, long_action, short_action, set_side=None
):
    """A long flipped straight to a short is ONE step old, not the long's age.

    THE TRAP: the exchange must still report the LONG when _step syncs -- the trade has not
    executed yet. Reporting the short there makes _sync_position_from_exchange see an EXTERNAL
    change and reset hold_counter itself (PR #245 / SLTPMixin), masking the bug: the test goes
    vacuous. Two of these passed on the buggy code before I found that.
    """
    from unittest.mock import MagicMock, patch

    import torch
    from tensordict import TensorDict

    # the four PositionStatus dataclasses are field-identical in order (only field 8 is named
    # margin_type on binance, margin_mode elsewhere), so build them positionally
    def pos(qty):
        return {"position_status": PositionStatus(
            qty, 500.0, 50000.0, 0.0, 0.0, 50000.0, 5, "isolated", 45000.0)}

    with patch.object(env, "_wait_for_next_timestamp"):
        if set_side:
            set_side("BUY")
        trader.get_status = MagicMock(return_value=pos(0.01))
        env.reset()

        for _ in range(5):
            td = env.step(TensorDict({"action": torch.tensor(long_action)}, batch_size=()))
        aged = td["next"]["account_state"][3].item()
        assert aged > 1.0, f"the long never aged ({aged}) -- the assertion below would be vacuous"

        if set_side:
            set_side("SELL")
        first = iter([pos(0.01)])          # sync reads the OLD long; every later read is short
        trader.get_status = MagicMock(side_effect=lambda: next(first, pos(-0.01)))
        td = env.step(TensorDict({"action": torch.tensor(short_action)}, batch_size=()))

    holding_time = td["next"]["account_state"][3].item()
    assert holding_time == 1.0, (
        f"a one-step-old short reports {holding_time} bars (the long aged to {aged}): the flip "
        f"never passed through flat, so the counter was never reset"
    )
