"""Alpaca crypto bars, on the shared observation base."""

from typing import List, Union, Callable, Dict, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from torchtrade.envs.live.shared.base_obs import BaseObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit, timeframe_to_alpaca
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient


class AlpacaObservationClass(BaseObservationClass):
    """Spot crypto bars from alpaca.

    Everything about the WINDOW is inherited. This class is the part alpaca does
    differently: it fetches a date RANGE rather than a bar count, and its SDK returns a
    frame with a (symbol, timestamp) MultiIndex instead of a list of klines.
    """

    def __init__(
        self,
        symbol: str,
        timeframes: Union[List[TimeFrame], TimeFrame],
        window_sizes: Union[List[int], int] = 1,
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[CryptoHistoricalDataClient] = None,
    ):
        """
        Initialize the AlpacaObservationClass. Default observation features are close, open, high, low.

        Args:
            symbol: The cryptocurrency symbol to fetch data for
            timeframes: Single custom TimeFrame or list of custom TimeFrames to fetch data for
            window_sizes: Single integer or list of integers specifying window_sizes.
                        If a list is provided, it must have the same length as timeframes.
            feature_preprocessing_fn: Optional custom preprocessing function that takes a DataFrame
                                   and returns a DataFrame with feature columns
            client: Optional pre-configured CryptoHistoricalDataClient for dependency injection (useful for testing)

        `timeframes`, not the base's `time_frames`: the spelling is public and renaming it
        would break every existing alpaca config for no behavioural gain.
        """
        self.default_lookback = 60
        super().__init__(
            symbol=symbol,
            time_frames=timeframes,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
        )

    def _create_client(self) -> object:
        return CryptoHistoricalDataClient()

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """The 60-day fetch window cannot fill a daily-bar observation."""
        if timeframe.unit == TimeFrameUnit.Day and self.default_lookback > 30:
            raise ValueError(
                "Default lookback is greater than 30 days, which is not allowed for daily data"
            )

    def _get_timestamp_column(self) -> str:
        return "timestamp"

    def _fetch_single_timeframe(self, timeframe: TimeFrame, limit: int = None) -> pd.DataFrame:
        """A date RANGE, not a bar count, so `limit` has nothing to bind to.

        That is why the short-window refusal in `get_observations` matters more here than
        on the futures venues: they over-fetch by a fixed +50 bars, and this fetches
        whatever 60 days happens to contain (#400).
        """
        now = datetime.now(ZoneInfo("America/New_York"))
        request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=timeframe_to_alpaca(timeframe),
            start=now - timedelta(days=self.default_lookback),
            end=now,
        )
        return self.client.get_crypto_bars(request).df

    def _normalise_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The SDK hands back a (symbol, timestamp) MultiIndex and a constant `symbol`.

        Dropped BEFORE the shared `dropna`, where it used to be dropped after. For a
        single-symbol request the column is constant and the surviving rows are the same
        either way -- measured. The one input that differs is a row whose `symbol` is
        null: it used to be discarded, and now survives if its OHLCV is intact. That is
        the better answer. Losing a usable bar to a gap in a METADATA column is the
        over-eager row removal #400 exists to catch, and `symbol` is a field this class
        already knows the value of -- it is the one it asked for.
        """
        return df.reset_index().drop(columns=["symbol"])

    def _dummy_frame(self, window_size: int) -> pd.DataFrame:
        """Carries `symbol` because `_normalise_frame` drops it -- the dummy has to have
        the shape real data has, or `get_features` measures a frame the venue never
        produces (the dtype-fidelity lesson from binance's #289 fold)."""
        df = super()._dummy_frame(window_size)
        df.insert(0, "symbol", [self.symbol] * window_size)
        return df

    def get_current_price(self) -> float:
        """
        Get the most recent close price for the symbol.

        This is useful for determining the current market price when there's no position.
        Uses the first timeframe's most recent close price.

        Returns:
            float: Most recent close price from the first timeframe
        """
        if not self.time_frames:
            raise ValueError("No timeframes configured")

        df = self._fetch_single_timeframe(self.time_frames[0])

        if df.empty:
            raise ValueError(f"No data available for {self.symbol}")

        return float(df['close'].iloc[-1])


# Example usage:
if __name__ == "__main__":
    # Note: Examples now use custom TimeFrame from torchtrade.envs.timeframe
    # instead of Alpaca's TimeFrame class

    # Single timeframe example
    print("Testing single timeframe...")
    window_size = 10
    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(15, TimeFrameUnit.Minute),
        window_sizes=window_size,
    )
    expected_keys = observer.get_keys()
    observations = observer.get_observations()
    #features = observer.get_features()

    assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
    # Default preprocessing has 4 features: feature_close, feature_open, feature_high, feature_low
    assert observations[expected_keys[0]].shape == (window_size, 4), \
        f"Expected shape (10, 4) for default features, got {observations[expected_keys[0]].shape}"
    print("Single timeframe test passed!")

    # Example with multiple timeframes and window sizes
    print("\nTesting multiple timeframes...")
    window_sizes = [10, 20]
    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=[
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour)
        ],
        window_sizes=window_sizes
    )

    expected_keys = observer.get_keys()
    print("Expected keys:", expected_keys)
    observations = observer.get_observations()
    #features = observer.get_features()

    assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
    assert len(observations) == 2, "Expected exactly 2 observations"

    # Check shapes for each timeframe/window combination
    expected_shapes = { key: (w, 4) for key, w in zip(expected_keys, window_sizes)
    }

    for key, expected_shape in expected_shapes.items():
        assert observations[key].shape == expected_shape, \
            f"Shape mismatch for {key}: expected {expected_shape}, got {observations[key].shape}"
    print("Multiple timeframes test passed!")

    # Custom preprocessing example
    print("\nTesting custom preprocessing...")
    def custom_preprocessing(df):
        df = df.reset_index()
        df.dropna(inplace=True)
        df["feature_volatility"] = df["high"] - df["low"]
        df["feature_volume_ma"] = df["volume"].rolling(window=3).mean()
        df.dropna(inplace=True)  # Drop NaN values from rolling window
        return df

    observer_custom = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(15, TimeFrameUnit.Minute),
        window_sizes=10,
        feature_preprocessing_fn=custom_preprocessing,
    )

    observations_custom = observer_custom.get_observations()
    key = observer_custom.get_keys()[0]
    #features_custom = observer_custom.get_features()
    # Custom preprocessing has 2 features and loses 2 rows due to rolling window
    assert observations_custom[key].shape == (8, 2), \
        f"Expected shape (8, 2) for custom features, got {observations_custom[key].shape}"
    print("Custom preprocessing test passed!")