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
        """A date RANGE, not a bar count, so `limit` has nothing to bind to."""
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

        Dropped BEFORE the shared dropna/drop_duplicates, where it used to go after. Safe
        only because `symbol: str` means one symbol per observer: with the column gone,
        two bars alike in every OHLCV field but differing in symbol would dedupe to one.
        Revisit this ordering before any multi-symbol support. (A row with a null symbol
        and sound OHLCV also survives now, which is the better answer -- losing a usable
        bar to a gap in metadata is what pushes a window under its spec, #400.)
        """
        return df.reset_index().drop(columns=["symbol"])

    def _dummy_frame(self, window_size: int) -> pd.DataFrame:
        """Carries `symbol` because `_normalise_frame` drops it: the dummy has to have
        the shape real data has, or `get_features` measures a frame that never occurs."""
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
