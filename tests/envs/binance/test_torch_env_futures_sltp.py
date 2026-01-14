"""Tests for BinanceFuturesSLTPTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict


class TestBinanceFuturesSLTPTorchTradingEnv:
    """Tests for BinanceFuturesSLTPTorchTradingEnv."""

    @pytest.fixture
    def mock_observer(self):
        """Create a mock observer."""
        observer = MagicMock()

        # Mock get_keys
        observer.get_keys = MagicMock(return_value=["1m_10"])

        # Mock get_observations
        def mock_observations(return_base_ohlc=False):
            obs = {
                "1m_10": np.random.randn(10, 4).astype(np.float32),
            }
            if return_base_ohlc:
                # OHLC features: [open, high, low, close]
                obs["base_features"] = np.array([
                    [50000, 50100, 49900, 50050]  # Current price ~50050
                ] * 10, dtype=np.float32)
            return obs

        observer.get_observations = MagicMock(side_effect=mock_observations)
        observer.intervals = ["1m"]
        observer.window_sizes = [10]

        return observer

    @pytest.fixture
    def mock_trader(self):
        """Create a mock trader."""
        trader = MagicMock()

        # Mock methods
        trader.cancel_open_orders = MagicMock(return_value=True)
        trader.close_position = MagicMock(return_value=True)

        trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0,
            "available_balance": 900.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 1000.0,
        })

        trader.get_mark_price = MagicMock(return_value=50000.0)

        trader.get_status = MagicMock(return_value={
            "position_status": None,
        })

        trader.trade = MagicMock(return_value=True)

        return trader

    @pytest.fixture
    def env_config(self):
        """Create environment configuration."""
        from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTradingEnvConfig

        return BinanceFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            intervals=["1m"],
            window_sizes=[10],
            execute_on="1m",
            leverage=5,
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
            include_short_positions=True,
            quantity_per_trade=0.001,
        )

    @pytest.fixture
    def env(self, env_config, mock_observer, mock_trader):
        """Create environment with mocks."""
        from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTorchTradingEnv

        # Patch time.sleep to avoid waiting
        with patch("time.sleep"):
            with patch.object(BinanceFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BinanceFuturesSLTPTorchTradingEnv(
                    config=env_config,
                    observer=mock_observer,
                    trader=mock_trader,
                )
                return env

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.config.symbol == "BTCUSDT"
        assert env.config.leverage == 5
        assert env.config.demo is True
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0

    def test_action_map_structure(self, env):
        """Test action map has correct structure."""
        # With 2 SL levels and 2 TP levels:
        # 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9 actions
        assert len(env.action_map) == 9
        assert env.action_map[0] == (None, None, None)  # HOLD

    def test_action_map_long_actions(self, env):
        """Test action map long actions."""
        # Actions 1-4 should be LONG with different SL/TP combos
        for i in range(1, 5):
            side, sl, tp = env.action_map[i]
            assert side == "long"
            assert sl < 0  # SL should be negative (below entry)
            assert tp > 0  # TP should be positive (above entry)

    def test_action_map_short_actions(self, env):
        """Test action map short actions."""
        # Actions 5-8 should be SHORT with flipped SL/TP
        for i in range(5, 9):
            side, sl, tp = env.action_map[i]
            assert side == "short"
            # For shorts: SL is above entry (positive), TP is below entry (negative)
            assert sl > 0  # SL above entry for shorts
            assert tp < 0  # TP below entry for shorts

    def test_action_spec_size(self, env):
        """Test action spec has correct size."""
        # 1 HOLD + 4 LONG + 4 SHORT = 9 actions
        assert env.action_spec.n == 9

    def test_action_spec_long_only(self, env_config, mock_observer, mock_trader):
        """Test action spec when short positions disabled."""
        env_config.include_short_positions = False

        with patch("time.sleep"):
            with patch("torchtrade.envs.binance.torch_env_futures_sltp.BinanceFuturesSLTPTorchTradingEnv._wait_for_next_timestamp"):
                from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTorchTradingEnv
                env = BinanceFuturesSLTPTorchTradingEnv(
                    config=env_config,
                    observer=mock_observer,
                    trader=mock_trader,
                )

        # 1 HOLD + 4 LONG only = 5 actions
        assert env.action_spec.n == 5

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys."""
        obs_spec = env.observation_spec

        assert "account_state" in obs_spec.keys()
        assert "market_data_1m_10" in obs_spec.keys()

    def test_account_state_shape(self, env):
        """Test account state has correct shape (10 elements for futures)."""
        obs_spec = env.observation_spec
        assert obs_spec["account_state"].shape == (10,)

    def test_reset(self, env, mock_trader):
        """Test environment reset."""
        td = env.reset()

        assert "account_state" in td.keys()
        assert "market_data_1m_10" in td.keys()

        mock_trader.cancel_open_orders.assert_called()
        # Active SL/TP should be reset
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0

    def test_reset_observation_shapes(self, env):
        """Test observation shapes after reset."""
        td = env.reset()

        assert td["account_state"].shape == (10,)
        assert td["market_data_1m_10"].shape == (10, 4)

    def test_step_hold_action(self, env, mock_trader):
        """Test step with HOLD action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())  # HOLD
            next_td = env.step(action_td)

            # Trade should NOT have been called
            mock_trader.trade.assert_not_called()

            # Check output structure
            assert "reward" in next_td["next"].keys()
            assert "done" in next_td["next"].keys()

    def test_step_long_action(self, env, mock_trader):
        """Test step with LONG action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())  # LONG with SL/TP
            next_td = env.step(action_td)

            # Trade should have been called with BUY
            assert mock_trader.trade.called
            call_kwargs = mock_trader.trade.call_args.kwargs
            assert call_kwargs["side"] == "BUY"
            assert "stop_loss" in call_kwargs
            assert "take_profit" in call_kwargs

    def test_step_short_action(self, env, mock_trader):
        """Test step with SHORT action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(5)}, batch_size=())  # SHORT with SL/TP
            next_td = env.step(action_td)

            # Trade should have been called with SELL
            assert mock_trader.trade.called
            call_kwargs = mock_trader.trade.call_args.kwargs
            assert call_kwargs["side"] == "SELL"
            assert "stop_loss" in call_kwargs
            assert "take_profit" in call_kwargs

    def test_bracket_order_prices_long(self, env, mock_trader, mock_observer):
        """Test bracket order prices are calculated correctly for LONG."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Action 1: LONG with first SL/TP combo (-0.02, 0.03)
            action_tuple = ("long", -0.02, 0.03)
            trade_info = env._execute_trade_if_needed(action_tuple)

            if trade_info["executed"]:
                # Current price is ~50050
                expected_sl = 50050 * (1 - 0.02)  # ~49049
                expected_tp = 50050 * (1 + 0.03)  # ~51551.5

                assert trade_info["stop_loss"] == pytest.approx(expected_sl, rel=1e-2)
                assert trade_info["take_profit"] == pytest.approx(expected_tp, rel=1e-2)

    def test_bracket_order_prices_short(self, env, mock_trader, mock_observer):
        """Test bracket order prices are calculated correctly for SHORT."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Action 5: SHORT with flipped SL/TP
            # For short: SL above entry, TP below entry
            action_tuple = ("short", 0.03, -0.02)  # Note: already flipped in action_map
            trade_info = env._execute_trade_if_needed(action_tuple)

            if trade_info["executed"]:
                # Current price is ~50050
                expected_sl = 50050 * (1 + 0.03)  # ~51551.5 (above entry)
                expected_tp = 50050 * (1 - 0.02)  # ~49049 (below entry)

                assert trade_info["stop_loss"] == pytest.approx(expected_sl, rel=1e-2)
                assert trade_info["take_profit"] == pytest.approx(expected_tp, rel=1e-2)

    def test_position_closed_detection(self, env, mock_trader):
        """Test detection of position closed by SL/TP."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.current_position = 1  # Simulate having a long position

            # Simulate position being closed
            mock_trader.get_status = MagicMock(return_value={
                "position_status": None
            })

            closed = env._check_position_closed()
            assert closed is True

    def test_position_not_closed_detection(self, env, mock_trader):
        """Test detection when position is still open."""
        from torchtrade.envs.binance.futures_order_executor import PositionStatus

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.current_position = 1

            # Position still exists
            mock_trader.get_status = MagicMock(return_value={
                "position_status": PositionStatus(
                    qty=0.001,
                    notional_value=50.0,
                    entry_price=50000.0,
                    unrealized_pnl=0.5,
                    unrealized_pnl_pct=0.01,
                    mark_price=50500.0,
                    leverage=5,
                    margin_type="isolated",
                    liquidation_price=45000.0,
                )
            })

            closed = env._check_position_closed()
            assert closed is False

    def test_active_sltp_tracking(self, env, mock_trader):
        """Test that active SL/TP levels are tracked."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Execute a long trade
            action_tuple = ("long", -0.02, 0.03)
            trade_info = env._execute_trade_if_needed(action_tuple)

            if trade_info["executed"]:
                # Active SL/TP should be set
                assert env.active_stop_loss > 0
                assert env.active_take_profit > 0
                assert env.active_stop_loss == trade_info["stop_loss"]
                assert env.active_take_profit == trade_info["take_profit"]

    def test_sltp_reset_on_position_close(self, env, mock_trader):
        """Test that SL/TP are reset when position closes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Set active SL/TP
            env.active_stop_loss = 49000.0
            env.active_take_profit = 51000.0
            env.current_position = 1

            # Simulate position closed
            mock_trader.get_status = MagicMock(return_value={
                "position_status": None
            })

            # Step should detect closure and reset SL/TP
            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0

    def test_cannot_open_position_while_holding(self, env, mock_trader):
        """Test that new positions cannot be opened while holding."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.current_position = 1  # Simulate holding long

            # Try to open another long
            action_tuple = ("long", -0.02, 0.03)
            trade_info = env._execute_trade_if_needed(action_tuple)

            assert trade_info["executed"] is False

    def test_account_state_with_position(self, env, mock_trader):
        """Test account state when there's an open position."""
        from torchtrade.envs.binance.futures_order_executor import PositionStatus

        # Mock a position
        mock_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.001,
                notional_value=50.0,
                entry_price=50000.0,
                unrealized_pnl=0.5,
                unrealized_pnl_pct=0.01,
                mark_price=50500.0,
                leverage=5,
                margin_type="isolated",
                liquidation_price=45000.0,
            )
        })

        td = env._get_observation()

        account_state = td["account_state"]
        assert account_state[1].item() == pytest.approx(0.001, rel=1e-3)  # position_size
        assert account_state[4].item() == pytest.approx(50500.0, rel=1e-3)  # current_price
        assert account_state[6].item() == pytest.approx(5.0, rel=1e-3)  # leverage

    def test_done_on_bankruptcy(self, env, mock_trader):
        """Test termination on bankruptcy."""
        env.initial_portfolio_value = 1000.0
        env.config.bankrupt_threshold = 0.1

        done = env._check_termination(50.0)  # Below 10% of 1000
        assert done is True

    def test_close_method(self, env, mock_trader):
        """Test environment close method."""
        env.close()
        mock_trader.cancel_open_orders.assert_called()


class TestBinanceFuturesSLTPTradingEnvConfig:
    """Tests for BinanceFuturesSLTPTradingEnvConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTradingEnvConfig

        config = BinanceFuturesSLTPTradingEnvConfig()

        assert config.symbol == "BTCUSDT"
        assert config.demo is True
        assert config.leverage == 1
        assert config.include_short_positions is True
        assert len(config.stoploss_levels) == 3
        assert len(config.takeprofit_levels) == 3

    def test_custom_sltp_levels(self):
        """Test custom SL/TP levels."""
        from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTradingEnvConfig

        config = BinanceFuturesSLTPTradingEnvConfig(
            stoploss_levels=(-0.01, -0.03, -0.05, -0.10),
            takeprofit_levels=(0.02, 0.05, 0.10, 0.20),
        )

        assert len(config.stoploss_levels) == 4
        assert len(config.takeprofit_levels) == 4

    def test_long_only_config(self):
        """Test long-only configuration."""
        from torchtrade.envs.binance.torch_env_futures_sltp import BinanceFuturesSLTPTradingEnvConfig

        config = BinanceFuturesSLTPTradingEnvConfig(
            include_short_positions=False
        )

        assert config.include_short_positions is False


class TestCombinatoryActionMap:
    """Tests for combinatory_action_map function."""

    def test_action_map_basic(self):
        """Test basic action map generation."""
        from torchtrade.envs.binance.torch_env_futures_sltp import combinatory_action_map

        action_map = combinatory_action_map(
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.06],
            include_short_positions=True
        )

        # 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9
        assert len(action_map) == 9
        assert action_map[0] == (None, None, None)

    def test_action_map_long_only(self):
        """Test action map with shorts disabled."""
        from torchtrade.envs.binance.torch_env_futures_sltp import combinatory_action_map

        action_map = combinatory_action_map(
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.06],
            include_short_positions=False
        )

        # 1 HOLD + 4 LONG (2x2) = 5
        assert len(action_map) == 5

    def test_action_map_long_sides(self):
        """Test that long actions have correct structure."""
        from torchtrade.envs.binance.torch_env_futures_sltp import combinatory_action_map

        action_map = combinatory_action_map(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
            include_short_positions=False
        )

        # Action 1 should be the only long action
        side, sl, tp = action_map[1]
        assert side == "long"
        assert sl == -0.02
        assert tp == 0.03

    def test_action_map_short_sign_flip(self):
        """Test that short actions have flipped signs."""
        from torchtrade.envs.binance.torch_env_futures_sltp import combinatory_action_map

        action_map = combinatory_action_map(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
            include_short_positions=True
        )

        # Action 2 should be the short action
        side, sl, tp = action_map[2]
        assert side == "short"
        # For shorts: SL is above entry (positive), TP is below entry (negative)
        assert sl == 0.03  # From takeprofit_levels (positive values become SL for shorts)
        assert tp == -0.02  # From stoploss_levels (negative values become TP for shorts)


class TestMultipleSteps:
    """Test multiple environment steps."""

    @pytest.fixture
    def env_with_mocks(self):
        """Create environment for multi-step testing."""
        from torchtrade.envs.binance.torch_env_futures_sltp import (
            BinanceFuturesSLTPTorchTradingEnv,
            BinanceFuturesSLTPTradingEnvConfig,
        )

        mock_observer = MagicMock()
        mock_observer.get_keys = MagicMock(return_value=["1m_10"])
        mock_observer.get_observations = MagicMock(return_value={
            "1m_10": np.random.randn(10, 4).astype(np.float32),
            "base_features": np.array([[50000, 50100, 49900, 50050]] * 10, dtype=np.float32),
        })
        mock_observer.intervals = ["1m"]
        mock_observer.window_sizes = [10]

        mock_trader = MagicMock()
        mock_trader.cancel_open_orders = MagicMock(return_value=True)
        mock_trader.close_position = MagicMock(return_value=True)
        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0,
            "available_balance": 900.0,
            "total_margin_balance": 1000.0,
        })
        mock_trader.get_mark_price = MagicMock(return_value=50000.0)
        mock_trader.get_status = MagicMock(return_value={"position_status": None})
        mock_trader.trade = MagicMock(return_value=True)

        config = BinanceFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            intervals=["1m"],
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        with patch("time.sleep"):
            with patch.object(BinanceFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BinanceFuturesSLTPTorchTradingEnv(
                    config=config,
                    observer=mock_observer,
                    trader=mock_trader,
                )
                return env

    def test_multiple_steps(self, env_with_mocks):
        """Test running multiple environment steps."""
        with patch.object(env_with_mocks, "_wait_for_next_timestamp"):
            env_with_mocks.reset()

            for _ in range(10):
                action = torch.randint(0, env_with_mocks.action_spec.n, (1,)).item()
                action_td = TensorDict({"action": torch.tensor(action)}, batch_size=())
                next_td = env_with_mocks.step(action_td)

                assert "reward" in next_td["next"].keys()
                assert "done" in next_td["next"].keys()

    def test_rollout(self, env_with_mocks):
        """Test environment rollout."""
        with patch.object(env_with_mocks, "_wait_for_next_timestamp"):
            env_with_mocks.reset()

            rewards = []
            for _ in range(5):
                action = torch.randint(0, env_with_mocks.action_spec.n, (1,)).item()
                action_td = TensorDict({"action": torch.tensor(action)}, batch_size=())
                next_td = env_with_mocks.step(action_td)
                rewards.append(next_td["next", "reward"].item())

            assert len(rewards) == 5
