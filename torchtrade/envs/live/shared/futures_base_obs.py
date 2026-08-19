"""Base observation class for futures exchanges."""

from abc import abstractmethod

import pandas as pd

from torchtrade.envs.live.shared.base_obs import BaseObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame


class BaseFuturesObservationClass(BaseObservationClass):
    """
    Market data from a futures exchange's kline endpoint.

    The window logic lives in `BaseObservationClass`; what is left here is the shape of a
    kline API -- fetch a raw list, parse it, name the interval -- which the four futures
    venues share and alpaca does not have at all.
    """

    @abstractmethod
    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Fetch raw kline/candle data from the provider's API.

        Args:
            symbol: Trading symbol
            interval: Provider-specific interval string
            limit: Number of candles to fetch

        Returns:
            Raw kline data from the API
        """
        pass
    @abstractmethod
    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw kline data into a standardized DataFrame.

        The returned DataFrame must have at least these columns:
        - timestamp: datetime column
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float

        Args:
            raw_klines: Raw kline data from the API

        Returns:
            DataFrame with standardized OHLCV columns
        """
        pass
    @abstractmethod
    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame to provider-specific interval format.

        Args:
            timeframe: TimeFrame object

        Returns:
            Provider-specific interval string (e.g., "1H" for Bitget, "1h" for Binance)
        """
        pass
    @abstractmethod
    def _get_default_lookback(self) -> int:
        """Get the default number of candles to fetch."""
        pass

    def _fetch_single_timeframe(self, timeframe: TimeFrame, limit: int = None) -> pd.DataFrame:
        """Fetch and preprocess data for a single timeframe."""
        if limit is None:
            limit = self._get_default_lookback()

        # Convert TimeFrame to provider-specific interval format
        provider_interval = self._convert_timeframe(timeframe)

        try:
            # Fetch klines using provider-specific method
            raw_klines = self._fetch_klines(self.symbol, provider_interval, limit)

            # Validate we got data
            if not raw_klines or len(raw_klines) == 0:
                raise ValueError(f"No candle data returned for {self.symbol} on {timeframe.obs_key_freq()}")

            # Parse klines into standardized DataFrame
            df = self._parse_klines(raw_klines)

            # Sort by timestamp (ascending) - some exchanges return in reverse order
            timestamp_col = self._get_timestamp_column()
            df = df.sort_values(timestamp_col).reset_index(drop=True)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch candles for {self.symbol} on {timeframe.obs_key_freq()}: {str(e)}")

    def _normalise_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """`_parse_klines` already returns a plain frame, so a copy is the whole job."""
        return df.copy()
