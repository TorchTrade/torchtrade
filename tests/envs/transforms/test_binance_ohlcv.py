"""Tests for BinanceOHLCVTransform."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import check_env_specs

from torchtrade.envs.live.polymarket import (
    PolymarketBetEnv,
    PolymarketBetEnvConfig,
)
from torchtrade.envs.live.polymarket.market_scanner import PolymarketMarket
from torchtrade.envs.transforms import BinanceOHLCVTransform
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


# --- Fixtures --------------------------------------------------------------- #


def _stub_observer(time_frames, window_sizes, n_features=4):
    """Build a stand-in for BinanceObservationClass that needs no network."""
    obs = MagicMock()
    obs.time_frames = list(time_frames)
    obs.window_sizes = list(window_sizes)
    obs.get_features.return_value = {
        "observation_features": [f"feature_{i}" for i in range(n_features)],
        "original_features": ["open", "high", "low", "close"],
    }

    def _get_observations():
        result = {}
        for tf, window in zip(time_frames, window_sizes):
            key = f"{tf.obs_key_freq()}_{window}"
            result[key] = np.random.randn(window, n_features).astype(np.float32)
        return result

    obs.get_observations.side_effect = _get_observations
    return obs


def _make_polymarket_env():
    """Minimal PolymarketBetEnv for transform integration tests."""
    market = PolymarketMarket(
        market_id="m1",
        condition_id="0xcond",
        question="Bitcoin Up or Down",
        description="",
        slug="btc-updown-5m-1234",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        yes_price=0.5,
        no_price=0.5,
        volume_24h=1.0,
        total_volume=1.0,
        liquidity=1.0,
        spread=0.0,
        end_date="2030-01-01T00:00:00Z",
        tags=[],
        neg_risk=False,
    )
    scanner = MagicMock()
    scanner.next_active_market.return_value = market
    trader = MagicMock()
    trader.buy.return_value = {"success": True}
    env = PolymarketBetEnv(
        PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", dry_run=True),
        scanner=scanner,
        trader=trader,
    )
    env._wait_for_resolution = lambda *a, **k: None
    env._fetch_resolved_outcome = lambda *a, **k: 1
    return env


# --- Spec-level tests ------------------------------------------------------- #


class TestSpec:
    def test_observation_spec_extends_with_per_window_keys(self):
        tfs = [TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)]
        windows = [60, 30]
        observer = _stub_observer(tfs, windows, n_features=4)
        transform = BinanceOHLCVTransform(observer=observer)

        env = TransformedEnv(_make_polymarket_env(), transform)
        keys = set(env.observation_spec.keys())
        assert "ohlcv_1Minute_60" in keys
        assert "ohlcv_5Minute_30" in keys
        # Original key is preserved
        assert "market_state" in keys

    def test_each_key_has_matching_shape_and_dtype(self):
        tfs = [TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)]
        windows = [60, 30]
        n_feat = 7  # arbitrary feature width
        observer = _stub_observer(tfs, windows, n_features=n_feat)
        transform = BinanceOHLCVTransform(observer=observer)

        env = TransformedEnv(_make_polymarket_env(), transform)
        assert env.observation_spec["ohlcv_1Minute_60"].shape == (60, n_feat)
        assert env.observation_spec["ohlcv_5Minute_30"].shape == (30, n_feat)
        assert env.observation_spec["ohlcv_1Minute_60"].dtype == torch.float32

    def test_check_env_specs_passes_on_wrapped_env(self):
        tfs = [TimeFrame(1, TimeFrameUnit.Minute)]
        observer = _stub_observer(tfs, [10], n_features=4)
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))
        check_env_specs(env)

    @pytest.mark.parametrize("prefix", ["ohlcv", "btc", "side_signal"])
    def test_key_prefix_is_configurable(self, prefix):
        observer = _stub_observer([TimeFrame(1, TimeFrameUnit.Minute)], [5])
        transform = BinanceOHLCVTransform(observer=observer, key_prefix=prefix)
        env = TransformedEnv(_make_polymarket_env(), transform)
        assert f"{prefix}_1Minute_5" in env.observation_spec.keys()


# --- Behavior tests --------------------------------------------------------- #


class TestBehavior:
    def test_reset_populates_ohlcv_keys(self):
        tfs = [TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)]
        observer = _stub_observer(tfs, [10, 6], n_features=3)
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))

        td = env.reset()
        assert td["ohlcv_1Minute_10"].shape == (10, 3)
        assert td["ohlcv_5Minute_6"].shape == (6, 3)
        assert td["ohlcv_1Minute_10"].dtype == torch.float32

    def test_step_populates_ohlcv_keys_under_next(self):
        tfs = [TimeFrame(1, TimeFrameUnit.Minute)]
        observer = _stub_observer(tfs, [4], n_features=2)
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))

        td = env.reset()
        out = env.step(td.set("action", torch.tensor(1)))
        assert out["next", "ohlcv_1Minute_4"].shape == (4, 2)

    def test_observer_called_per_step(self):
        observer = _stub_observer([TimeFrame(1, TimeFrameUnit.Minute)], [5])
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))

        td = env.reset()                                                     # call 1
        env.step(td.set("action", torch.tensor(1)))                           # call 2
        td2 = env.reset()                                                    # call 3
        env.step(td2.set("action", torch.tensor(0)))                          # call 4

        assert observer.get_observations.call_count == 4

    def test_missing_source_key_filled_with_zeros_to_honor_spec(self, caplog):
        """Every key declared in ``transform_observation_spec`` MUST also be
        present in the runtime output, otherwise downstream collectors that
        trust the spec will crash on missing keys. If the observer omits a key,
        the transform fills with zeros of the declared shape and logs a
        warning, never silently drops the key."""
        observer = _stub_observer(
            [TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
            [4, 4],
            n_features=2,
        )
        # Force the observer to omit the 5Minute_4 key
        observer.get_observations.side_effect = lambda: {
            "1Minute_4": np.ones((4, 2), dtype=np.float32),
        }
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))

        with caplog.at_level("WARNING"):
            td = env.reset()

        # Both keys present at runtime, the spec contract holds.
        assert "ohlcv_1Minute_4" in td.keys()
        assert "ohlcv_5Minute_4" in td.keys()
        # Provided key gets the real value
        assert torch.allclose(td["ohlcv_1Minute_4"], torch.ones((4, 2)))
        # Missing key gets zeros of the declared shape
        assert td["ohlcv_5Minute_4"].shape == (4, 2)
        assert torch.equal(td["ohlcv_5Minute_4"], torch.zeros((4, 2)))
        # And the warning surfaced, silent fallback would be the wrong choice
        assert any("5Minute_4" in rec.message for rec in caplog.records)

    def test_observer_exception_propagates(self):
        """A failure inside the observer must surface, silently substituting
        zeros would let the policy train on bogus side-channel data."""
        observer = _stub_observer([TimeFrame(1, TimeFrameUnit.Minute)], [4])
        observer.get_observations.side_effect = RuntimeError("binance 451")
        env = TransformedEnv(_make_polymarket_env(), BinanceOHLCVTransform(observer=observer))
        with pytest.raises(RuntimeError, match="binance 451"):
            env.reset()


# --- Construction tests ----------------------------------------------------- #


class TestConstruction:
    def test_default_construction_uses_btc_1m_5m_15m(self, monkeypatch):
        """No-arg construction defaults to BTC + 1m/5m/15m × 60/30/20."""
        observed_kwargs = {}

        def fake_init(self, **kwargs):
            observed_kwargs.update(kwargs)
            self.time_frames = kwargs["time_frames"]
            self.window_sizes = kwargs["window_sizes"]

        def fake_get_features(self):
            return {"observation_features": ["a", "b", "c", "d"], "original_features": []}

        monkeypatch.setattr(
            "torchtrade.envs.transforms.binance_ohlcv.BinanceObservationClass.__init__",
            fake_init,
        )
        monkeypatch.setattr(
            "torchtrade.envs.transforms.binance_ohlcv.BinanceObservationClass.get_features",
            fake_get_features,
        )
        BinanceOHLCVTransform()
        assert observed_kwargs["symbol"] == "BTCUSDT"
        assert [tf.obs_key_freq() for tf in observed_kwargs["time_frames"]] == [
            "1Minute", "5Minute", "15Minute",
        ]
        assert observed_kwargs["window_sizes"] == [60, 30, 20]

    def test_injected_observer_is_used_directly(self):
        """When ``observer=`` is provided, the other kwargs are ignored."""
        observer = _stub_observer([TimeFrame(1, TimeFrameUnit.Minute)], [3])
        # symbol/time_frames/window_sizes here would be wrong if applied,
        # the test passes only because the injected observer wins.
        transform = BinanceOHLCVTransform(
            observer=observer,
            symbol="WRONG",
            time_frames=[TimeFrame(1, TimeFrameUnit.Day)],
            window_sizes=[999],
        )
        assert transform.observer is observer
