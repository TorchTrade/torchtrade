"""Observation class for fetching market data from OKX."""
from typing import Union, List, Optional, Callable
import pandas as pd

from torchtrade.envs.live.shared.futures_base_obs import BaseFuturesObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame
from torchtrade.envs.live.okx.utils import timeframe_to_okx, normalize_symbol


class OKXObservationClass(BaseFuturesObservationClass):
    """
    Observation class for fetching market data from OKX.

    Inherits common functionality from BaseFuturesObservationClass.
    Implements OKX-specific API interaction using python-okx for futures market data.
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
        """
        Initialize the OKXObservationClass.

        Args:
            symbol: The trading symbol (e.g., "BTC-USDT-SWAP")
            time_frames: Single TimeFrame or list of TimeFrame objects
            window_sizes: Single integer or list of integers specifying window sizes
            feature_preprocessing_fn: Optional custom preprocessing function
            client: Optional pre-configured OKX MarketData client for dependency injection
            demo: Whether to use demo trading environment (default: True)
        """
        symbol = normalize_symbol(symbol)
        super().__init__(
            symbol=symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
            demo=demo,
        )

    def _create_client(self) -> object:
        """Create OKX MarketData API client (public endpoints, no keys needed)."""
        import okx.MarketData as MarketData

        flag = "1" if self.demo else "0"
        return MarketData.MarketAPI(flag=flag)

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """Validate that a timeframe is supported by OKX."""
        timeframe_to_okx(timeframe)

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Fetch raw kline data from OKX API.

        Args:
            symbol: Trading symbol (e.g., "BTC-USDT-SWAP")
            interval: OKX-specific interval string (e.g., "1m", "1H", "1D")
            limit: Number of candles to fetch

        Returns:
            Raw kline data from OKX: result["data"] (reverse chronological order)
        """
        response = self.client.get_candlesticks(
            instId=symbol,
            bar=interval,
            limit=str(limit),
        )
        code = response.get("code", "-1")
        if str(code) != "0":
            msg = response.get("msg", "unknown error")
            raise RuntimeError(f"get_candlesticks failed (code={code}): {msg}")
        return response["data"]

    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw OKX kline data into standardized DataFrame.

        OKX kline format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        Note: OKX returns data in reverse chronological order (newest first).

        Args:
            raw_klines: Raw kline data from OKX

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', *ohlcv_cols, 'vol_ccy', 'vol_ccy_quote', 'confirm'
        ])

        # Convert types (OKX returns strings)
        df[ohlcv_cols] = df[ohlcv_cols].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

        # Ensure chronological order (OKX returns newest first)
        df = df.sort_values('timestamp', ascending=True, kind='stable').reset_index(drop=True)

        return df[['timestamp', *ohlcv_cols]]

    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame to OKX-specific interval format.

        Args:
            timeframe: TimeFrame object

        Returns:
            OKX interval string (e.g., "1m", "1H", "1D")
        """
        return timeframe_to_okx(timeframe)

    def _get_default_lookback(self) -> int:
        """Get the default number of candles to fetch (OKX max per request is 300)."""
        return 200

    def _get_timestamp_column(self) -> str:
        """Get the name of the timestamp column."""
        return "timestamp"
