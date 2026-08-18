"""Tests for BinanceObservationClass.

Common observation-class behavior is inherited from BaseObservationClassTests.
Only Binance-specific tests (symbol normalization, exchange-specific kline columns,
default feature names) and stricter assertions live here.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.shared.futures_base_obs import BaseFuturesObservationClass
from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.okx.observation import OKXObservationClass
from tests.envs.base_exchange_tests import BaseObservationClassTests


def _make_binance_client():
    """Mock Binance client returning `limit` 12-column klines (chronological)."""
    client = MagicMock()

    def mock_get_klines(symbol, interval, limit=500):
        base_time = 1700000000000
        return [
            [base_time + i * 60000, "50000.0", "50100.0", "49900.0", "50050.0", "100.0",
             base_time + i * 60000 + 59999, "5000000.0", "100", "50.0", "2500000.0", "0"]
            for i in range(limit)
        ]

    client.get_klines = MagicMock(side_effect=mock_get_klines)
    return client


class TestBinanceObservationClass(BaseObservationClassTests):
    """Binance observation class — common tests inherited from the base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        client = kwargs.pop("client", None) or _make_binance_client()
        return BinanceObservationClass(
            symbol=symbol, time_frames=timeframes, window_sizes=window_sizes,
            client=client, **kwargs,
        )

    def get_expected_symbol_format(self, symbol):
        return symbol.replace("/", "")

    @pytest.fixture
    def mock_client(self):
        return _make_binance_client()

    @pytest.fixture
    def observer_single(self, mock_client):
        return BinanceObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client)

    @pytest.fixture
    def observer_multi(self, mock_client):
        return BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute),
                         TimeFrame(1, TimeFrameUnit.Hour)],
            window_sizes=[10, 20, 15], client=mock_client)

    # --- Binance-specific / stricter tests ---

    def test_single_interval_initialization(self, observer_single):
        """Symbol + timeframe unit preserved (base checks neither)."""
        assert observer_single.symbol == "BTCUSDT"
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        assert observer_multi.symbol == "BTCUSDT"
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_client):
        """Slash is stripped from the symbol."""
        observer = BinanceObservationClass(
            symbol="BTC/USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client)
        assert observer.symbol == "BTCUSDT"

    def test_get_keys_multi(self, observer_multi):
        assert observer_multi.get_keys() == ["1Minute_10", "5Minute_20", "1Hour_15"]

    def test_get_observations_single_dtype(self, observer_single):
        """Single-timeframe observation is exactly (10, 4) float32 (stricter than base)."""
        obs = observer_single.get_observations()
        assert obs["15Minute_10"].shape == (10, 4)
        assert obs["15Minute_10"].dtype == np.float32

    def test_get_observations_multi_exact_shapes(self, observer_multi):
        obs = observer_multi.get_observations()
        assert obs["1Minute_10"].shape == (10, 4)
        assert obs["5Minute_20"].shape == (20, 4)
        assert obs["1Hour_15"].shape == (15, 4)

    def test_get_observations_with_base_ohlc(self, observer_single):
        """base_features + base_timestamps present with exact shape (adds base_timestamps)."""
        obs = observer_single.get_observations(return_base_ohlc=True)
        assert "15Minute_10" in obs
        assert "base_features" in obs
        assert "base_timestamps" in obs
        assert obs["base_features"].shape == (10, 4)

    def test_default_preprocessing_output(self, observer_single):
        """Default preprocessing produces the expected named features."""
        features = observer_single.get_features()
        for feat in ["feature_close", "feature_open", "feature_high", "feature_low"]:
            assert feat in features["observation_features"]


class TestBinanceObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Binance API connection")
    def test_live_data_fetch(self):
        """Test fetching live data from Binance."""
        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[10, 10])
        observations = observer.get_observations()
        assert "1Minute_10" in observations
        assert "5Minute_10" in observations


class TestBinanceSharesTheObservationBase:
    """binance was the last FUTURES venue with a parallel observation class (#289).

    alpaca still hand-rolls its own; it is spot, so outside this issue's scope.
    """

    @staticmethod
    def _klines(n, descending=False):
        base = 1700000000000
        rows = [[base + i * 60000, "100", "101", "99", "100.5", "10",
                 base + (i + 1) * 60000 - 1, "1000", "50", "5", "500", "0"]
                for i in range(n)]
        return list(reversed(rows)) if descending else rows

    def _obs(self, rows, fn=None):
        client = MagicMock()
        client.get_klines = MagicMock(return_value=rows)
        return BinanceObservationClass(
            symbol="BTC/USDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5,
            feature_preprocessing_fn=fn,
            client=client,
        )

    @pytest.mark.parametrize("venue_cls", [
        BinanceObservationClass, BitgetObservationClass,
        BybitObservationClass, OKXObservationClass,
    ], ids=lambda c: c.__name__)
    def test_no_venue_reimplements_the_shared_base(self, venue_cls):
        """Derived from the base, not hand-listed, and over every venue.

        The named subset guards the guard: a discovered set can SHRINK silently (making
        `get_features` a property drops it from a callable filter), and a merely non-empty
        check would still pass.
        """
        shared = {
            n for n, v in vars(BaseFuturesObservationClass).items()
            if callable(v) and not getattr(v, "__isabstractmethod__", False)
            and not n.startswith("__")
        }
        assert {"get_features", "get_observations", "_fetch_single_timeframe"} <= shared
        # _dummy_frame is an extension point, not a re-fork: the venues return different
        # kline columns and the base cannot know their dtypes.
        redeclared = (shared & set(vars(venue_cls))) - {"_dummy_frame"}
        assert not redeclared, f"{venue_cls.__name__} re-forked: {sorted(redeclared)}"

    def test_a_dtype_conditional_preprocessing_fn_gets_the_real_dtype(self):
        """The dummy frame must match the parsed klines' DTYPES, not just its columns.

        `trades` is int64 in real klines. A names-only dummy (every column a float in
        [0,1)) made this fn take the else-branch on the dummy and the if-branch on live
        data: get_features() declared 1 feature, get_observations() emitted 2, and
        `BinanceOHLCVTransform` sizes its spec from the former -- so check_env_specs
        failed on a shape mismatch. Measured, after making exactly that simplification.
        """
        def fn(df):
            df = df.copy()
            df["feature_close"] = df["close"].pct_change().fillna(0)
            if pd.api.types.is_integer_dtype(df["trades"]):
                df["feature_trades"] = df["trades"] / 100.0
            return df.dropna()

        obs = self._obs(self._klines(60), fn=fn)
        declared = len(obs.get_features()["observation_features"])
        assert declared == obs.get_observations()["1Minute_5"].shape[1] == 2

    def test_klines_are_sorted_even_when_the_response_is_reversed(self):
        """The base sorts; binance's copy did not, so it was the one venue unprotected.

        Binance returns ascending today, so nothing bit -- but a reversed response
        produced TIME-REVERSED observations with no error. This is also the only test in
        the four venue suites that reaches the base's sort: bybit and okx sort inside
        their own _parse_klines, so their sort tests stay green if the base's is deleted.
        """
        df = self._obs(self._klines(20, descending=True))._fetch_single_timeframe(
            TimeFrame(1, TimeFrameUnit.Minute), limit=20
        )
        assert df["open_time"].is_monotonic_increasing

    def test_a_malformed_candle_never_reaches_base_features(self):
        """`to_numeric(errors="coerce")` turns an unparseable close into NaN by design.

        Measured before the fix: a bad candle INSIDE the window gave
        `base_features NaN=True` while the feature array stayed clean, because the
        feature path drops it in preprocessing and base_features was built from the raw
        frame. Outside the window it never showed. (#395)
        """
        rows = self._klines(60)
        rows[57][4] = "NOT_A_NUMBER"  # close, inside the last 10
        obs = self._obs(rows).get_observations(return_base_ohlc=True)
        assert not np.isnan(obs["base_features"]).any()
        assert not np.isnan(obs["1Minute_5"]).any()
