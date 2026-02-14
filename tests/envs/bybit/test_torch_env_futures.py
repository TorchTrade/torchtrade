"""Tests for BybitFuturesTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestBybitFuturesTorchTradingEnv:
    """Tests for BybitFuturesTorchTradingEnv."""

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
            "total_wallet_balance": 1000.0,
            "available_balance": 900.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 1000.0,
        })
        trader.get_mark_price = MagicMock(return_value=50000.0)
        trader.get_status = MagicMock(return_value={"position_status": None})
        trader.trade = MagicMock(return_value=True)
        trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        return trader

    @pytest.fixture
    def env_config(self):
        """Create environment configuration."""
        from torchtrade.envs.live.bybit.env import BybitFuturesTradingEnvConfig

        return BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m", "5m"],
            window_sizes=[10, 10],
            execute_on="1m",
            leverage=5,
        )

    @pytest.fixture
    def env(self, env_config, mock_observer, mock_trader):
        """Create environment with mocks."""
        from torchtrade.envs.live.bybit.env import BybitFuturesTorchTradingEnv

        with patch("time.sleep"):
            with patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                return BybitFuturesTorchTradingEnv(
                    config=env_config,
                    observer=mock_observer,
                    trader=mock_trader,
                )

    def test_action_spec(self, env):
        """Test action spec and levels are correctly defined."""
        assert env.action_spec.n == 5  # [-1.0, -0.5, 0.0, 0.5, 1.0]
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
        assert "market_data_1Minute_10" in td.keys()
        assert "market_data_5Minute_10" in td.keys()
        assert td["account_state"].shape == (6,)
        assert td["market_data_1Minute_10"].shape == (10, 4)
        mock_trader.cancel_open_orders.assert_called()

    def test_step_hold_action(self, env, mock_trader):
        """Test step with hold action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # 0.0
            next_td = env.step(action_td)

            assert "reward" in next_td["next"].keys()
            assert "done" in next_td["next"].keys()
            assert "account_state" in next_td["next"].keys()

    @pytest.mark.parametrize("action_idx,label", [
        (4, "long"),   # action_levels[4] = 1.0
        (0, "short"),  # action_levels[0] = -1.0
    ], ids=["long", "short"])
    def test_step_trade_action(self, env, mock_trader, action_idx, label):
        """Test step with long/short action calls trade."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
            env.step(action_td)
            mock_trader.trade.assert_called()

    def test_reward_and_done_tensor_shapes(self, env):
        """Test that reward and done flags have correct tensor shapes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            next_td = env.step(action_td)

            assert next_td["next"]["reward"].shape == (1,)
            assert next_td["next"]["done"].shape == (1,)
            assert next_td["next"]["terminated"].shape == (1,)
            assert next_td["next"]["truncated"].shape == (1,)

    def test_no_bankruptcy_when_disabled(self, env_config, mock_observer, mock_trader):
        """Test that bankruptcy check can be disabled."""
        from torchtrade.envs.live.bybit.env import BybitFuturesTorchTradingEnv

        env_config.done_on_bankruptcy = False

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
                config=env_config, observer=mock_observer, trader=mock_trader,
            )

        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 10.0, "available_balance": 10.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 10.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            next_td = env.step(action_td)
            assert next_td["next"]["done"].item() is False

    def test_config_post_init(self):
        """Test config post_init normalization."""
        from torchtrade.envs.live.bybit.env import BybitFuturesTradingEnvConfig

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames="1m",
            window_sizes=10,
        )

        assert isinstance(config.time_frames, list)
        assert isinstance(config.window_sizes, list)
        assert all(isinstance(tf, TimeFrame) for tf in config.time_frames)


class TestBybitActionIndexClamping:
    """Tests for action index out-of-range clamping."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            action_levels=[-1.0, 0.0, 1.0],
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("action_idx", [-1, 5], ids=["negative", "too-high"])
    def test_action_index_clamping(self, env, action_idx):
        """Out-of-range action indices must be clamped with warning."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
            # Should not raise IndexError
            next_td = env.step(action_td)
            assert "reward" in next_td["next"].keys()

    def test_nan_action_defaults_to_zero(self, env):
        """NaN action must default to action index 0 without crashing."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(float("nan"))}, batch_size=())
            next_td = env.step(action_td)
            assert "reward" in next_td["next"].keys()


class TestBybitZeroLiquidationPrice:
    """Test distance_to_liquidation with zero/missing liquidation price."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("qty,liq_price,expected_dtl", [
        (0.001, 45000.0, pytest.approx(0.1018, rel=1e-2)),  # Long normal: (50100-45000)/50100
        (0.001, 0.0, 1.0),                                   # Long zero liq: unknown → 1.0
        (-0.001, 55000.0, pytest.approx(0.0978, rel=1e-2)),  # Short normal: (55000-50100)/50100
        (-0.001, 0.0, 1.0),                                   # Short zero liq: unknown → 1.0
    ], ids=["long-normal", "long-zero-liq", "short-normal", "short-zero-liq"])
    def test_distance_to_liquidation(self, env, mock_env_trader, qty, liq_price, expected_dtl):
        """Zero liquidation price must return 1.0 (consistent with Binance/Bitget)."""
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=qty, notional_value=50.1, entry_price=50000.0,
                unrealized_pnl=0.1, unrealized_pnl_pct=0.002,
                mark_price=50100.0, leverage=10, margin_mode="1",
                liquidation_price=liq_price,
            )
        })

        td = env._get_observation()
        distance_to_liq = td["account_state"][5].item()
        assert distance_to_liq == expected_dtl


class TestBybitNoInitSideEffects:
    """Test that __init__ does not call cancel_open_orders or close_position."""

    def test_init_does_not_cancel_orders(self, mock_env_observer, mock_env_trader):
        """Environment construction must not cancel orders or close positions."""
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            close_position_on_reset=True,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        # __init__ should NOT have side effects on the exchange
        mock_env_trader.cancel_open_orders.assert_not_called()
        mock_env_trader.close_position.assert_not_called()

    def test_reset_calls_close_position_when_configured(self, mock_env_observer, mock_env_trader):
        """reset() must call close_position when close_position_on_reset=True."""
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            close_position_on_reset=True,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )
            env.reset()

        mock_env_trader.cancel_open_orders.assert_called()
        mock_env_trader.close_position.assert_called()

    def test_close_resilient_when_cancel_raises(self, mock_env_observer, mock_env_trader):
        """close() must not raise even if cancel_open_orders fails."""
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        mock_env_trader.cancel_open_orders = MagicMock(side_effect=Exception("API down"))
        # close() must not propagate the exception
        env.close()


class TestBybitObservationSpecsNoNetwork:
    """Test that _build_observation_specs doesn't make live API calls."""

    def test_build_specs_uses_get_features_not_get_observations(self, mock_env_observer, mock_env_trader):
        """_build_observation_specs must use get_features() (dummy data), not get_observations()."""
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        # Set up mock get_features (returns feature info without network call)
        mock_env_observer.get_features = MagicMock(return_value={
            "observation_features": ["feature_close", "feature_open", "feature_high", "feature_low"],
            "original_features": ["open", "high", "low", "close", "volume"],
        })

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        # get_features should have been called (for spec building)
        mock_env_observer.get_features.assert_called_once()
        # Verify specs were built correctly
        assert "account_state" in env.observation_spec.keys()
        assert "market_data_1Minute_10" in env.observation_spec.keys()


class TestBybitFractionalPositionResizing:
    """Tests for fractional position resizing."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )

        with patch("time.sleep"), \
             patch("torchtrade.envs.live.bybit.env.BybitFuturesTorchTradingEnv._wait_for_next_timestamp"):
            return BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("first_action,second_action,should_execute", [
        (0.5, 1.0, True),    # Scale up long
        (-0.5, -1.0, True),  # Scale up short
        (1.0, 0.5, True),    # Scale down long
        (1.0, 1.0, False),   # Same level: skip
        (0.0, 0.0, False),   # Both flat: skip
    ])
    def test_fractional_resizing_executes(self, env, first_action, second_action, should_execute):
        """Changing action level within same direction must trigger trade."""
        trade_executed = {"executed": True, "amount": 0.01, "side": "buy",
                         "success": True, "closed_position": False}

        with patch.object(env, '_execute_fractional_action', return_value=trade_executed) as mock_exec:
            env.position.current_action_level = first_action
            result = env._execute_trade_if_needed(second_action)

            if should_execute:
                mock_exec.assert_called_once_with(second_action)
            else:
                mock_exec.assert_not_called()
                assert result["executed"] is False

    def test_qty_step_rounding_no_float_artifacts(self, env, mock_env_trader):
        """Quantity must be rounded to avoid float artifacts like 0.00300000000003."""
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.0, notional_value=0, entry_price=0,
                unrealized_pnl=0, unrealized_pnl_pct=0,
                mark_price=50000.0, leverage=1, margin_mode="1",
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
        result = env._execute_fractional_action(1.0)
        if result["executed"]:
            call_kwargs = mock_env_trader.trade.call_args[1]
            qty = call_kwargs["quantity"]
            # Verify no float artifacts (e.g., 0.00300000000003)
            qty_str = str(qty)
            assert len(qty_str.split('.')[-1]) <= 3, f"Float artifact in quantity: {qty}"
