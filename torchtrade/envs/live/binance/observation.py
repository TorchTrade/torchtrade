"""Observation class for fetching market data from Binance."""
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from torchtrade.envs.live.shared.futures_base_obs import BaseFuturesObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame, timeframe_to_binance

# Kline layout: [open_time, open, high, low, close, volume, close_time, quote_volume,
#               trades, taker_buy_base, taker_buy_quote, ignore]
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
_NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]


class BinanceObservationClass(BaseFuturesObservationClass):
    """Binance market data. The last venue still carrying a parallel copy (#289).

    Four of its nine shared methods were already byte-identical to the base's; what
    genuinely differs is the API call and the kline layout, which are what the base's
    hooks are for.
    """

    def __init__(
        self,
        symbol: str,
        time_frames: Union[List[TimeFrame], TimeFrame],
        window_sizes: Union[List[int], int] = 10,
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[object] = None,
        demo: bool = True,
    ):
        super().__init__(
            symbol=symbol.replace("/", ""),
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
            demo=demo,
        )

    def _create_client(self) -> object:
        """Public market data only, so no keys."""
        try:
            from binance.client import Client
        except ImportError:
            raise ImportError(
                "python-binance is required. Install with: pip install python-binance"
            )
        return Client()

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        timeframe_to_binance(timeframe)

    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        return timeframe_to_binance(timeframe)

    def _get_default_lookback(self) -> int:
        return 500  # binance's own per-request limit

    def _get_timestamp_column(self) -> str:
        return "open_time"

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        df = pd.DataFrame(raw_klines, columns=_KLINE_COLUMNS)
        # to_numeric with coerce, not astype: a malformed row becomes NaN and is dropped
        # by preprocessing rather than killing the whole fetch.
        for col in _NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return df.drop(columns=["ignore"])

    def _dummy_frame(self, window_size: int) -> pd.DataFrame:
        """Binance klines carry more than OHLCV, and a preprocessing fn may read it."""
        df = super()._dummy_frame(window_size)
        df["volume"] += 1e-9  # a preprocessing fn that divides by volume must not hit 0
        df["quote_volume"] = np.random.rand(window_size)
        df["trades"] = np.random.randint(1, 100, window_size)
        df["taker_buy_base"] = np.random.rand(window_size)
        df["taker_buy_quote"] = np.random.rand(window_size)
        return df
