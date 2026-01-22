from typing import List, Union, Callable, Dict, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit, timeframe_to_alpaca
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient

class AlpacaObservationClass:
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
        """
        self.symbol = symbol
        self.timeframes = (
            [timeframes] if isinstance(timeframes, TimeFrame) else timeframes
        )
        self.window_sizes = (
            [window_sizes] if isinstance(window_sizes, int) else window_sizes
        )
        self.default_lookback = 60
        # Validate that timeframes and window_sizes have matching lengths when both are lists
        if isinstance(timeframes, list) and isinstance(window_sizes, list):
            if len(self.timeframes) != len(self.window_sizes):
                raise ValueError("If both timeframes and window_sizes are lists, they must have the same length")

        self.feature_preprocessing_fn = (
            feature_preprocessing_fn or self._default_preprocessing
        )
        self.client = client if client is not None else CryptoHistoricalDataClient()

    def _default_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default preprocessing function if none is provided."""
        df = df.reset_index()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df.drop(columns=["symbol"])

        # Generating base features
        # TODO: should those be unnormalized?
        df["feature_close"] = df["close"].pct_change().fillna(0)
        df["feature_open"] = df["open"] / df["close"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]
        df.dropna(inplace=True)
        return df

    def _get_numpy_obs(self, df: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """Convert specified columns to numpy array."""
        if columns is None:
            columns = [col for col in df.columns if "feature" in col]
        if not columns:
            raise ValueError(f"No columns found in preprocessed DataFrame matching: {columns}")
        return np.array(df[columns], dtype=np.float32)

    def _fetch_single_timeframe(
        self, timeframe: TimeFrame #, window_size: int
    ) -> pd.DataFrame:
        """Fetch and preprocess data for a single timeframe.

        Args:
            timeframe: Custom TimeFrame object

        Returns:
            DataFrame with OHLCV data
        """
        if timeframe.unit == TimeFrameUnit.Day and self.default_lookback > 30:
            raise ValueError("Default lookback is greater than 30 days, which is not allowed for daily data")

        # Convert custom TimeFrame to Alpaca TimeFrame for API calls
        alpaca_timeframe = timeframe_to_alpaca(timeframe)

        now = datetime.now(ZoneInfo("America/New_York"))
        request = CryptoBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=alpaca_timeframe,
            start=now - timedelta(days=self.default_lookback), # TODO: this is critical
            end=now,
            #limit=window_size
        )
        return self.client.get_crypto_bars(request).df

    def get_keys(self) -> List[str]:
        """
        Get the list of keys that will be present in the observations dictionary.

        Returns:
            List of strings formatted as '{timeframe}_{window_size}'
            (e.g., '5Minute_10', '1Hour_20')
        """
        return [
            f"{tf.obs_key_freq()}_{ws}"
            for tf, ws in zip(self.timeframes, self.window_sizes)
        ]

    def get_features(self) -> Dict[str, List[str]]:
        """Returns a dictionary with the observation features and the original features."""
        def get_dummy_data(window_size: int):
            df = pd.DataFrame()
            df["symbol"] = [self.symbol]*window_size
            df["open"] = np.random.rand(window_size)
            df["high"] = np.random.rand(window_size)
            df["low"] = np.random.rand(window_size)
            df["close"] = np.random.rand(window_size)
            df["volume"] = np.random.rand(window_size)
            return df
        # TODO: we could do this for all window sizes in case we have different processings per time frame
        #features = []
        #for window_size in self.window_sizes:
            #dummy_df = get_dummy_data(window_size)
            #features.extend([f for f in self.feature_preprocessing_fn(dummy_df).columns if "feature" in f])
        dummy_df = self.feature_preprocessing_fn(get_dummy_data(self.window_sizes[0]))
        observation_features = [f for f in dummy_df.columns if "feature" in f]
        original_features = [f for f in dummy_df.columns if "feature" not in f]
        return {"observation_features": observation_features, "original_features": original_features}

    def get_observations(self, return_base_ohlc: bool = False) -> Dict[str, np.ndarray]:
        """
        Fetch and process observations for all specified timeframes and window sizes.

        Args:
            return_base_ohlc: If True, includes the raw OHLC data from the first timeframe
                            in the observations dictionary under the 'base_features' key.

        Returns:
            Dictionary with keys formatted as '{timeframe}_{window_size}' and numpy array values.
            If return_base_ohlc is True, includes 'base_features'and 'base_timestamps' keys with raw OHLC and timestamp data.
        """
        observations = {}

        for timeframe, window_size in zip(self.timeframes, self.window_sizes):
            key = f"{timeframe.obs_key_freq()}_{window_size}"
            df = self._fetch_single_timeframe(timeframe)
            
            # Store base OHLC features if this is the first timeframe and return_base_ohlc is True
            if return_base_ohlc and timeframe == self.timeframes[0]:
                base_df = df.reset_index()
                base_df.dropna(inplace=True)
                base_df.drop_duplicates(inplace=True)
                observations['base_features'] = self._get_numpy_obs(
                    base_df, 
                    columns=['open', 'high', 'low', 'close']
                )
                observations['base_timestamps'] = base_df['timestamp'].values
            processed_df = self.feature_preprocessing_fn(df)
            # apply window size
            processed_df = processed_df.iloc[-window_size:]
            observations[key] = self._get_numpy_obs(processed_df)

        return observations


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