"""Tests for replay components."""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def _make_test_df(n=100):
    """Create a simple test DataFrame with OHLCV data."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = np.linspace(50000, 51000, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": np.ones(n) * 100,
    })


class TestReplayOrderExecutor:
    """Test ReplayOrderExecutor simulated trading."""

    @pytest.fixture
    def executor(self):
        return ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.001)

    def test_initial_state(self, executor):
        """Executor starts flat with full balance."""
        status = executor.get_status()
        assert status["position_status"] is None
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] == 10000.0

    def test_open_long_position(self, executor):
        """trade(BUY) opens a long position."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        success = executor.trade(side="BUY", quantity=0.1, order_type="market")
        assert success is True
        status = executor.get_status()
        pos = status["position_status"]
        assert pos is not None
        assert pos.qty == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(50000.0)

    def test_open_short_position(self, executor):
        """trade(SELL) opens a short position."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, order_type="market")
        status = executor.get_status()
        pos = status["position_status"]
        assert pos.qty == pytest.approx(-0.1)

    def test_close_position_with_pnl(self, executor):
        """Closing a long position updates balance with P&L."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, order_type="market")
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 51000})
        executor.close_position()
        status = executor.get_status()
        assert status["position_status"] is None
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] > 10000.0

    def test_get_mark_price(self, executor):
        """get_mark_price returns the latest close price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50050})
        assert executor.get_mark_price() == 50050.0

    def test_bracket_status_set_on_trade(self, executor):
        """trade() with SL/TP sets bracket_status."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, take_profit=51000, stop_loss=49000)
        assert executor.bracket_status == {"tp_placed": True, "sl_placed": True}

    def test_cancel_open_orders_clears_brackets(self, executor):
        """cancel_open_orders clears active SL/TP brackets."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, take_profit=51000, stop_loss=49000)
        executor.cancel_open_orders()
        assert executor.sl_price == 0.0
        assert executor.tp_price == 0.0
        assert executor.bracket_status == {"tp_placed": False, "sl_placed": False}


class TestReplayOrderExecutorSLTPTriggers:
    """Test intrabar SL/TP trigger detection via advance_bar."""

    @pytest.fixture
    def executor(self):
        return ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.0)

    def test_long_sl_triggers_on_low(self, executor):
        """Long SL triggers when bar low <= stop_loss price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49500, take_profit=51000)
        executor.advance_bar({"open": 50000, "high": 50200, "low": 49000, "close": 49800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_long_tp_triggers_on_high(self, executor):
        """Long TP triggers when bar high >= take_profit price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49000, take_profit=50500)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 50800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_short_sl_triggers_on_high(self, executor):
        """Short SL triggers when bar high >= stop_loss price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, stop_loss=50500, take_profit=49000)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 50800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_short_tp_triggers_on_low(self, executor):
        """Short TP triggers when bar low <= take_profit price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, stop_loss=51000, take_profit=49500)
        executor.advance_bar({"open": 50000, "high": 50200, "low": 49000, "close": 49800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_sl_checked_before_tp(self, executor):
        """When both SL and TP trigger on same bar, SL wins (pessimistic)."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49500, take_profit=50500)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49000, "close": 50000})
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] < 10000.0

    def test_no_trigger_when_flat(self, executor):
        """advance_bar with no position should not crash."""
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49000, "close": 50000})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_transaction_fee_applied(self):
        """Fees deducted on open and close."""
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.001)
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, order_type="market")
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.close_position()
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] == pytest.approx(10000.0 - 10.0, rel=0.01)

    def test_reset_restores_all_state(self):
        """reset() must restore executor to initial state completely."""
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.001)
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49000, take_profit=51000)

        executor.reset()

        assert executor.position_qty == 0.0
        assert executor.entry_price == 0.0
        assert executor.balance == 10000.0
        assert executor.sl_price == 0.0
        assert executor.tp_price == 0.0
        assert executor.current_price == 0.0
        assert executor.bracket_status == {"tp_placed": False, "sl_placed": False}
        assert executor.last_order_id is None
        status = executor.get_status()
        assert status["position_status"] is None

    def test_short_profit_pnl(self):
        """Short position with price drop should produce profit."""
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.0)
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1)
        # Price drops — short profits
        executor.advance_bar({"open": 49000, "high": 49100, "low": 48900, "close": 49000})
        executor.close_position()
        balance = executor.get_account_balance()
        # PnL: -0.1 * (49000 - 50000) = +100
        assert balance["total_wallet_balance"] > 10000.0

    def test_short_loss_pnl(self):
        """Short position with price rise should produce loss."""
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.0)
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1)
        # Price rises — short loses
        executor.advance_bar({"open": 51000, "high": 51100, "low": 50900, "close": 51000})
        executor.close_position()
        balance = executor.get_account_balance()
        # PnL: -0.1 * (51000 - 50000) = -100
        assert balance["total_wallet_balance"] < 10000.0


class TestReplayObserver:
    """Test ReplayObserver wrapping MarketDataObservationSampler."""

    @pytest.fixture
    def df(self):
        return _make_test_df(100)

    @pytest.fixture
    def observer(self, df):
        return ReplayObserver(
            df=df,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    def test_get_keys(self, observer):
        """get_keys returns sampler observation keys."""
        keys = observer.get_keys()
        assert len(keys) == 1

    def test_get_observations_returns_numpy(self, observer):
        """get_observations returns numpy arrays (not tensors)."""
        obs = observer.get_observations()
        key = observer.get_keys()[0]
        assert isinstance(obs[key], np.ndarray)
        assert obs[key].shape[0] == 10  # window_size

    def test_get_observations_advances_bar(self, observer):
        """Each call to get_observations advances to the next bar."""
        obs1 = observer.get_observations(return_base_ohlc=True)
        price1 = obs1["base_features"][-1, 3]  # close

        obs2 = observer.get_observations(return_base_ohlc=True)
        price2 = obs2["base_features"][-1, 3]

        assert price1 != price2

    def test_get_observations_with_base_ohlc(self, observer):
        """return_base_ohlc=True includes base_features."""
        obs = observer.get_observations(return_base_ohlc=True)
        assert "base_features" in obs
        assert obs["base_features"].shape == (10, 4)

    def test_observer_feeds_executor(self, df):
        """Observer calls executor.advance_bar on each get_observations."""
        executor = ReplayOrderExecutor(initial_balance=10000.0)
        observer = ReplayObserver(
            df=df,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            executor=executor,
        )

        observer.get_observations()
        assert executor.current_price > 0

    def test_reset_resets_sampler(self, observer):
        """reset() resets to start."""
        obs1 = observer.get_observations(return_base_ohlc=True)
        first_price = obs1["base_features"][-1, 3]

        observer.get_observations()
        observer.get_observations()
        observer.reset()

        obs_after_reset = observer.get_observations(return_base_ohlc=True)
        reset_price = obs_after_reset["base_features"][-1, 3]
        assert reset_price == pytest.approx(first_price, rel=1e-4)


class TestReplayIntegrationWithLiveEnv:
    """Test replay components injected into a real live SLTP env."""

    @pytest.fixture
    def df(self):
        return _make_test_df(200)

    def test_replay_with_bybit_sltp_env(self, df):
        """ReplayObserver + ReplayOrderExecutor work as drop-in for Bybit SLTP."""
        from torchtrade.envs.live.bybit.env_sltp import (
            BybitFuturesSLTPTorchTradingEnv,
            BybitFuturesSLTPTradingEnvConfig,
        )

        config = BybitFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
            leverage=5,
            trade_mode="quantity",
            quantity_per_trade=0.01,
        )

        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
        observer = ReplayObserver(
            df=df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            executor=executor,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config,
                observer=observer,
                trader=executor,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

            for i in range(10):
                action = 1 if i % 3 == 0 else 0
                action_td = td.clone()
                action_td["action"] = torch.tensor(action)
                td = env.step(action_td)["next"]

                assert "reward" in td.keys()
                assert "done" in td.keys()

            assert executor.current_price > 0
