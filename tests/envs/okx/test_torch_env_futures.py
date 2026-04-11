"""Tests for OKXFuturesTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestOKXFuturesTorchTradingEnv:
    """Tests for OKXFuturesTorchTradingEnv."""

    @pytest.fixture
    def mock_observer(self):
        """Create a mock observer with two timeframes."""
        observer = MagicMock()
        observer.get_keys = MagicMock(return_value=["1Minute_10", "5Minute_10"])

        def mock_observations(return_base_ohlc=False):
            obs = {
                "1Minute_10": np.random.randn(10, 4).astype(np.float32),
                "5Minute_10": np.random.randn(10, 4).astype(np.float32),
            }
            if return_base_ohlc:
                obs["base_features"] = np.random.randn(10, 4).astype(np.float32)
                obs["base_timestamps"] = np.arange(10)
            return obs

        observer.get_observations = MagicMock(side_effect=mock_observations)
        observer.get_features = MagicMock(return_value={
            "observation_features": ["feature_close", "feature_open", "feature_high", "feature_low"],
            "original_features": ["open", "high", "low", "close", "volume"],
        })
        return observer

    @pytest.fixture
    def mock_trader(self):
        """Create a mock trader."""
        trader = MagicMock()
        trader.cancel_open_orders = MagicMock(return_value=True)
        trader.close_position = MagicMock(return_value=True)
        trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })
        trader.get_mark_price = MagicMock(return_value=50000.0)
        trader.get_status = MagicMock(return_value={"position_status": None})
        trader.trade = MagicMock(return_value=True)
        trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        return trader

    @pytest.fixture
    def env_config(self):
        from torchtrade.envs.live.okx.env import OKXFuturesTradingEnvConfig
        return OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", demo=True,
            time_frames=["1m", "5m"], window_sizes=[10, 10], execute_on="1m", leverage=5,
        )

    @pytest.fixture
    def env(self, env_config, mock_observer, mock_trader):
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv

        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesTorchTradingEnv(
                config=env_config, observer=mock_observer, trader=mock_trader,
            )

    def test_action_spec(self, env):
        """Test action spec and levels are correctly defined."""
        assert env.action_spec.n == 5
        assert env.action_levels == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys with correct shapes."""
        obs_spec = env.observation_spec
        assert "account_state" in obs_spec.keys()
        assert "market_data_1Minute_10" in obs_spec.keys()
        assert "market_data_5Minute_10" in obs_spec.keys()
        assert obs_spec["account_state"].shape == (6,)

    def test_reset(self, env, mock_trader):
        """Test environment reset returns expected keys and shapes."""
        td = env.reset()
        assert "account_state" in td.keys()
        assert td["account_state"].shape == (6,)
        assert td["market_data_1Minute_10"].shape == (10, 4)
        mock_trader.cancel_open_orders.assert_called()

    def test_step_hold_action(self, env):
        """Test step with hold action produces valid output."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            next_td = env.step(action_td)
            assert "reward" in next_td["next"].keys()
            assert "done" in next_td["next"].keys()

    @pytest.mark.parametrize("action_idx,label", [
        (4, "long"), (0, "short"),
    ], ids=["long", "short"])
    def test_step_trade_action(self, env, mock_trader, action_idx, label):
        """Test step with long/short action calls trade."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
            mock_trader.trade.assert_called()

    def test_reward_and_done_tensor_shapes(self, env):
        """Test that reward and done flags have correct tensor shapes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
            assert next_td["next"]["reward"].shape == (1,)
            assert next_td["next"]["done"].shape == (1,)
            assert next_td["next"]["terminated"].shape == (1,)
            assert next_td["next"]["truncated"].shape == (1,)

    def test_no_bankruptcy_when_disabled(self, env_config, mock_observer, mock_trader):
        """Test that bankruptcy check can be disabled."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv
        env_config.done_on_bankruptcy = False

        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesTorchTradingEnv(config=env_config, observer=mock_observer, trader=mock_trader)

        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 10.0, "available_balance": 10.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 10.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
            assert next_td["next"]["done"].item() is False


class TestOKXActionIndexClamping:
    """Tests for action index out-of-range clamping."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", action_levels=[-1.0, 0.0, 1.0],
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

    @pytest.mark.parametrize("action_idx", [-1, 5, float("nan")], ids=["negative", "too-high", "nan"])
    def test_invalid_action_index_handled(self, env, action_idx):
        """Out-of-range and NaN action indices must be handled without crashing."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
            assert "reward" in next_td["next"].keys()


class TestOKXZeroLiquidationPrice:
    """Test distance_to_liquidation with zero/missing liquidation price."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

    @pytest.mark.parametrize("qty,liq_price,expected_dtl", [
        (0.001, 45000.0, pytest.approx(0.1018, rel=1e-2)),
        (0.001, 0.0, 1.0),
        (-0.001, 55000.0, pytest.approx(0.0978, rel=1e-2)),
        (-0.001, 0.0, 1.0),
    ], ids=["long-normal", "long-zero-liq", "short-normal", "short-zero-liq"])
    def test_distance_to_liquidation(self, env, mock_env_trader, qty, liq_price, expected_dtl):
        """Zero liquidation price must return 1.0 (consistent with other exchanges)."""
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=qty, notional_value=50.1, entry_price=50000.0,
                unrealized_pnl=0.1, unrealized_pnl_pct=0.002,
                mark_price=50100.0, leverage=10, margin_mode="isolated",
                liquidation_price=liq_price,
            )
        })
        td = env._get_observation()
        assert td["account_state"][5].item() == expected_dtl


class TestOKXInitCleanup:
    """Test init/reset cleanup behavior."""

    @pytest.mark.parametrize("close_on_init,expect_close", [(True, True), (False, False)])
    def test_init_close_position_configurable(self, mock_env_observer, mock_env_trader, close_on_init, expect_close):
        """close_position_on_init controls whether positions are closed on startup."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", close_position_on_init=close_on_init,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

        mock_env_trader.cancel_open_orders.assert_called_once()
        if expect_close:
            mock_env_trader.close_position.assert_called_once()
        else:
            mock_env_trader.close_position.assert_not_called()

    @pytest.mark.parametrize("cancel_ok,close_ok", [
        (False, True), (True, False), (False, False),
    ], ids=["cancel-fails", "close-fails", "both-fail"])
    def test_reset_logs_warning_on_cleanup_failure(self, mock_env_observer, mock_env_trader, cancel_ok, close_ok):
        """reset() must warn but not raise when cleanup calls return False."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", close_position_on_reset=True,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

        mock_env_trader.cancel_open_orders = MagicMock(return_value=cancel_ok)
        mock_env_trader.close_position = MagicMock(return_value=close_ok)
        assert env.reset() is not None

    def test_close_resilient_when_cancel_raises(self, mock_env_observer, mock_env_trader):
        """close() must not raise even if cancel_open_orders fails."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

        mock_env_trader.cancel_open_orders = MagicMock(side_effect=Exception("API down"))
        env.close()  # Must not raise


class TestOKXObservationSpecsNoNetwork:
    """Test that _build_observation_specs doesn't make live API calls."""

    def test_build_specs_uses_get_features_not_get_observations(self, mock_env_observer, mock_env_trader):
        """_build_observation_specs must use get_features() (dummy data), not get_observations()."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

        mock_env_observer.get_features.assert_called_once()
        assert "account_state" in env.observation_spec.keys()
        assert "market_data_1Minute_10" in env.observation_spec.keys()


class TestOKXFractionalPositionResizing:
    """Tests for fractional position resizing."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesTorchTradingEnv(config=config, observer=mock_env_observer, trader=mock_env_trader)

    def test_qty_step_rounding_no_float_artifacts(self, env, mock_env_trader):
        """Quantity must be rounded to avoid float artifacts like 0.00300000000003."""
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.0, notional_value=0, entry_price=0,
                unrealized_pnl=0, unrealized_pnl_pct=0,
                mark_price=50000.0, leverage=1, margin_mode="isolated",
                liquidation_price=0,
            )
        })
        mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)
        mock_env_trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        mock_env_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })

        env.reset()
        result = env._execute_fractional_action(1.0, current_qty=0.0, current_price=50000.0)
        if result["executed"]:
            qty = mock_env_trader.trade.call_args[1]["quantity"]
            qty_str = str(qty)
            assert len(qty_str.split('.')[-1]) <= 3, f"Float artifact in quantity: {qty}"


class TestWithReplayData:
    """Integration tests using ReplayObserver + ReplayOrderExecutor."""

    @pytest.fixture
    def replay_df(self):
        import pandas as pd

        n = 200
        rng = np.random.default_rng(42)
        base = 50000 + np.cumsum(rng.normal(0, 50, n))
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": base,
            "high": base + np.abs(rng.normal(30, 20, n)),
            "low": base - np.abs(rng.normal(30, 20, n)),
            "close": base + rng.normal(0, 20, n),
            "volume": rng.uniform(100, 1000, n),
        })

    def test_multi_step_episode_with_replay(self, replay_df):
        """Run a multi-step episode with realistic price data."""
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", leverage=5, demo=True,
        )
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
        observer = ReplayObserver(
            df=replay_df, time_frames=config.time_frames,
            window_sizes=config.window_sizes, execute_on=config.execute_on, executor=executor,
        )

        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesTorchTradingEnv(config=config, observer=observer, trader=executor)

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()
            for i in range(20):
                action_idx = i % len(env.action_levels)
                action_td = td.clone()
                action_td["action"] = torch.tensor(action_idx)
                result = env.step(action_td)
                td = result["next"]
                assert td["account_state"].shape == (6,)
                if td["done"].item():
                    break
            assert executor.current_price > 0
