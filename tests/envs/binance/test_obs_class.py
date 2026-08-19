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


def _basic_features(df):
    """Minimal stand-in for a user's preprocessing fn: adds a feature, drops nothing else."""
    df = df.copy()
    df["feature_close"] = df["close"].pct_change().fillna(0)
    return df.dropna()


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
        # Over the MRO, not one class's __dict__: the window logic now lives on
        # BaseObservationClass and only the kline half on BaseFuturesObservationClass
        # (#288). A __dict__-scoped set silently stopped covering `get_observations` the
        # moment it was hoisted, which is how this guard would have gone quiet.
        shared = {
            n for klass in BaseFuturesObservationClass.__mro__
            if klass.__module__.startswith("torchtrade.")
            for n, v in vars(klass).items()
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


    @pytest.mark.parametrize("corrupt", [
        pytest.param(lambda r: r[57].__setitem__(4, "NOT_A_NUMBER"), id="nan-close"),
        pytest.param(lambda r: r[57].__setitem__(5, "NOT_A_NUMBER"), id="nan-volume"),
        pytest.param(lambda r: r[57].__setitem__(7, "NOT_A_NUMBER"), id="nan-quote-volume"),
        pytest.param(lambda r: r.__setitem__(57, list(r[56])), id="duplicate-bar"),
        pytest.param(
            lambda r: (r[57].__setitem__(1, "0"), r[57].__setitem__(4, "0")),
            id="zero-priced-bar",
        ),
    ])
    def test_base_features_and_market_data_stay_the_same_bars(self, corrupt):
        """Row i of base_features must be the SAME BAR as row i of market_data.

        The first fix here dropped only the OHLC columns. A NaN in `volume` then removed
        the bar from the feature window -- whose preprocessing does a bare dropna -- and
        KEPT it in base_features, so the two arrays described different bars inside one
        observation, silently:

            base : 23:08 23:09 23:10 23:11 23:12
            feat : 23:07 23:08 23:09 23:11 23:12

        Worse than the NaN it was fixing, and invisible to a NaN-only assertion. Then a
        bare dropna desynced on a REPEATED bar; then dropna+drop_duplicates desynced on a
        ZERO-PRICED bar, whose 0/0 feature is removed by the dropna that runs AFTER the
        features are built. Three attempts, each matching part of the rule.

        base_features is now sliced from the frame the preprocessing fn returned, so this
        holds by construction rather than by copying the rule -- including for a custom
        fn, which no copy could ever cover. Each parametrized case is one attempt's
        counterexample.
        """
        rows = self._klines(60)
        corrupt(rows)
        observer = self._obs(rows)
        obs = observer.get_observations(return_base_ohlc=True)

        surviving = observer.feature_preprocessing_fn(
            observer._fetch_single_timeframe(TimeFrame(1, TimeFrameUnit.Minute), limit=55)
        )["open_time"].iloc[-5:].values
        assert list(pd.to_datetime(obs["base_timestamps"]).values) == list(
            pd.to_datetime(surviving).values
        )
        # Alignment alone is satisfiable by two equally-wrong arrays: the reference above
        # re-uses the same preprocessing fn, so deleting its final dropna lets NaN into
        # both and the timestamps still agree. base_features must also be usable.
        #
        assert np.isfinite(obs["base_features"]).all()
        # market_data too, since #398: pct_change off a zero close was inf, and dropna
        # removed NaN but not inf, so the bar AFTER a zero-priced one carried it through.
        assert np.isfinite(obs["1Minute_5"]).all()

    def test_a_custom_preprocessing_fn_keeps_base_features_aligned(self):
        """The whole point of slicing the processed frame: it holds for ANY fn.

        Every other case here uses the default preprocessing, so the claim that a custom
        fn is covered was the one thing untested -- and it is what caught a real
        regression: reading the timestamp as a COLUMN broke a fn that leaves it in the
        index, which is how alpaca's SDK hands it back.
        """
        def fn(df):
            df = df.copy()
            df["feature_range"] = (df["high"] - df["low"]) / df["close"]
            return df[df["volume"] > 0].dropna()  # a row filter the default never does

        rows = self._klines(60)
        rows[57][5] = "0"  # volume 0 -> this fn drops the bar, the default would not
        observer = self._obs(rows, fn=fn)
        obs = observer.get_observations(return_base_ohlc=True)

        surviving = fn(observer._fetch_single_timeframe(
            TimeFrame(1, TimeFrameUnit.Minute), limit=55
        ))["open_time"].iloc[-5:].values
        assert list(pd.to_datetime(obs["base_timestamps"]).values) == list(
            pd.to_datetime(surviving).values
        )

    @pytest.mark.parametrize("venue_cls", [
        BinanceObservationClass, BitgetObservationClass,
        BybitObservationClass, OKXObservationClass,
    ], ids=lambda c: c.__name__)
    def test_no_venue_reforks_get_observations(self, venue_cls):
        """The #395 fix lives in the base's get_observations, inherited by all four.

        The env layer has `test_no_futures_env_reforks_the_shared_observation`; the
        OBSERVATION layer had no equivalent, so a venue re-forking this method would
        silently stop inheriting the fix and nothing would notice.
        """
        assert "get_observations" not in vars(venue_cls), (
            f"{venue_cls.__name__} re-forks get_observations -- it would not inherit the "
            f"base_features/market_data row alignment (#395)"
        )

    def test_a_stale_last_bar_is_refused_not_silently_backfilled(self):
        """`base_features[-1, 3]` is the current price at three SLTP call sites.

        Dropping a non-finite bar makes `[-1]` the PREVIOUS bar, so the isfinite/`<= 0`
        guard there passes on a stale price. Measured: main read 0.0 and REFUSED, this
        branch read the prior close and would have TRADED. Older bars may be
        dropped; the most recent one may not be (#398).
        """
        rows = self._klines(60)
        rows[59][4] = "0"  # the most recent candle is unusable
        with pytest.raises(ValueError, match="most recent candle"):
            self._obs(rows).get_observations(return_base_ohlc=True)

    def test_a_fn_that_moves_the_timestamp_into_the_index_still_works(self):
        """`_timestamps_of` on the futures base -- alpaca got this in #397, the futures
        half was lost before that commit landed, so reading the column unconditionally
        still raised KeyError here for a fn that sets it as an index.
        """
        def fn(df):
            df = df.copy()
            df["feature_close"] = df["close"].pct_change().fillna(0)
            return df.dropna().set_index("open_time")

        obs = self._obs(self._klines(60), fn=fn).get_observations(return_base_ohlc=True)
        assert obs["base_features"].shape == (5, 4)
        assert len(obs["base_timestamps"]) == 5

    def test_a_fn_that_loses_the_timestamp_entirely_is_refused(self):
        """Neither column nor index level: raise rather than pass positions off as times."""
        def fn(df):
            df = df.copy()
            df["feature_close"] = df["close"].pct_change().fillna(0)
            return df.dropna().reset_index(drop=True).drop(columns=["open_time"])

        with pytest.raises(KeyError, match="open_time"):
            self._obs(self._klines(60), fn=fn).get_observations(return_base_ohlc=True)

    @pytest.mark.parametrize("fn,label", [
        pytest.param(
            lambda df: _basic_features(df).iloc[:-1], "trims the forming last bar",
            id="trims-the-forming-bar",
        ),
        pytest.param(
            lambda df: _basic_features(df).set_index("open_time")
            .resample("5min").last().dropna().reset_index(),
            "resamples to a coarser timeframe", id="resamples",
        ),
    ])
    def test_a_custom_fn_may_change_what_the_last_bar_is(self, fn, label):
        """The stale-bar refusal is scoped to the DEFAULT preprocessing, deliberately.

        Its premise -- the last processed row is the last fetched row -- is only a
        promise the default makes. Trimming a still-forming candle and resampling to a
        coarser timeframe are both documented uses of `feature_preprocessing_fn`, and
        guarding them raised on EVERY call, permanently (#398 review).
        """
        obs = self._obs(self._klines(60), fn=fn).get_observations(return_base_ohlc=True)
        assert obs["base_features"].shape[0] > 0
