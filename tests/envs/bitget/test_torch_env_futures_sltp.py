"""Tests for BitgetFuturesSLTPTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestBitgetFuturesSLTPTorchTradingEnv:
    """Tests for BitgetFuturesSLTPTorchTradingEnv."""

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
        from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTradingEnvConfig

        return BitgetFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m"],
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
        from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTorchTradingEnv

        # Patch time.sleep to avoid waiting
        with patch("time.sleep"):
            with patch.object(BitgetFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesSLTPTorchTradingEnv(
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
        from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTorchTradingEnv

        env_config.include_short_positions = False

        with patch("time.sleep"):
            with patch.object(BitgetFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesSLTPTorchTradingEnv(
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
        """Test account state has correct shape (6 elements)."""
        obs_spec = env.observation_spec
        assert obs_spec["account_state"].shape == (6,)

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

        assert td["account_state"].shape == (6,)
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

    def test_step_long_action_with_sltp(self, env, mock_trader):
        """Test step with LONG action places bracket order."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())  # Long action
            next_td = env.step(action_td)

            # Trade should have been called with SL/TP
            mock_trader.trade.assert_called()
            call_kwargs = mock_trader.trade.call_args[1]

            assert call_kwargs["side"] == "buy"
            assert "take_profit" in call_kwargs
            assert "stop_loss" in call_kwargs

    def test_step_short_action_with_sltp(self, env, mock_trader):
        """Test step with SHORT action places bracket order."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(5)}, batch_size=())  # Short action
            next_td = env.step(action_td)

            # Trade should have been called with SL/TP
            mock_trader.trade.assert_called()
            call_kwargs = mock_trader.trade.call_args[1]

            assert call_kwargs["side"] == "sell"
            assert "take_profit" in call_kwargs
            assert "stop_loss" in call_kwargs

    def test_sltp_prices_calculated_correctly(self, env, mock_trader):
        """Test that SL/TP prices are calculated from percentages."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Action 1: Long with first SL/TP combo
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            call_kwargs = mock_trader.trade.call_args[1]

            # Current price is ~50050 from base_features
            current_price = 50050.0

            # First SL/TP combo: -0.02, 0.03
            expected_sl = current_price * (1 - 0.02)  # 49049
            expected_tp = current_price * (1 + 0.03)  # 51551.5

            assert call_kwargs["stop_loss"] == pytest.approx(expected_sl, rel=1e-2)
            assert call_kwargs["take_profit"] == pytest.approx(expected_tp, rel=1e-2)

    def test_active_sltp_tracking(self, env, mock_trader):
        """Test that active SL/TP levels are tracked."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Place order with SL/TP
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            # Active SL/TP should be set
            assert env.active_stop_loss > 0
            assert env.active_take_profit > 0

    def test_position_closed_resets_sltp(self, env, mock_trader):
        """Test that position closure resets SL/TP tracking."""

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Place order
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            # Set active SL/TP
            env.active_stop_loss = 49000.0
            env.active_take_profit = 51000.0
            env.position.current_position = 1

            # Mock position closed (SL/TP triggered)
            mock_trader.get_status = MagicMock(return_value={
                "position_status": None,  # Position closed
            })

            # Next step should detect closure and reset
            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())  # HOLD
            env._step(action_td)

            # SL/TP should be reset
            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0
            assert env.position.current_position == 0

    def test_reward_tensor_shape(self, env):
        """Test that reward is returned as tensor with correct shape."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            reward = next_td["next"]["reward"]
            assert isinstance(reward, torch.Tensor)
            assert reward.shape == (1,)

    def test_done_tensor_shape(self, env):
        """Test that done flags are tensors with correct shape."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            done = next_td["next"]["done"]
            terminated = next_td["next"]["terminated"]
            truncated = next_td["next"]["truncated"]

            assert isinstance(done, torch.Tensor)
            assert isinstance(terminated, torch.Tensor)
            assert isinstance(truncated, torch.Tensor)
            assert done.shape == (1,)

    def test_bankruptcy_termination(self, env, mock_trader):
        """Test that environment terminates on bankruptcy."""
        # Mock low balance
        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 50.0,  # Below 10% of initial 1000
            "available_balance": 50.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 50.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            done = next_td["next"]["done"]
            assert done.item() is True

    def test_no_trade_when_position_exists(self, env, mock_trader):
        """Test that no trade is placed when position already exists."""
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        # Mock existing position
        mock_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.001,
                notional_value=50.0,
                entry_price=50000.0,
                unrealized_pnl=0.5,
                unrealized_pnl_pct=0.01,
                mark_price=50500.0,
                leverage=5,
                margin_mode="isolated",
                liquidation_price=45000.0,
            )
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.position.current_position = 1  # Mark as having position

            # Try to place another order
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            # Trade should NOT have been called (already in position)
            mock_trader.trade.assert_not_called()

    def test_config_post_init(self):
        """Test config post_init normalization."""
        from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTradingEnvConfig

        config = BitgetFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames="1m",  # Single string
            window_sizes=10,  # Single int
        )

        assert isinstance(config.time_frames, list)
        assert isinstance(config.window_sizes, list)
        assert len(config.time_frames) == 1
        assert all(isinstance(tf, TimeFrame) for tf in config.time_frames)
        assert config.window_sizes == [10]

    def test_stoploss_takeprofit_levels(self, env):
        """Test that SL/TP levels are stored correctly."""
        assert env.stoploss_levels == [-0.02, -0.05]
        assert env.takeprofit_levels == [0.03, 0.06]


class TestBitgetFuturesSLTPTorchTradingEnvIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection and credentials")
    def test_live_environment(self):
        """Test environment with live Bitget testnet."""
        import os
        from torchtrade.envs.live.bitget.env_sltp import (
            BitgetFuturesSLTPTorchTradingEnv,
            BitgetFuturesSLTPTradingEnvConfig,
        )

        config = BitgetFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            leverage=5,
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        env = BitgetFuturesSLTPTorchTradingEnv(
            config=config,
            api_key=os.getenv("BITGET_API_KEY"),
            api_secret=os.getenv("BITGET_SECRET"),
            api_passphrase=os.getenv("BITGET_PASSPHRASE"),
        )

        td = env.reset()
        assert "account_state" in td.keys()

        action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
        next_td = env.step(action_td)
        assert "reward" in next_td["next"].keys()


class TestDuplicateActionPrevention:
    """Test duplicate action prevention and position switch logic (PR #XXX)."""

    @pytest.fixture
    def env_with_mocks(self):
        """Create environment for duplicate action testing."""
        from torchtrade.envs.live.bitget.env_sltp import (
            BitgetFuturesSLTPTorchTradingEnv,
            BitgetFuturesSLTPTradingEnvConfig,
        )

        mock_observer = MagicMock()
        mock_observer.get_keys = MagicMock(return_value=["1m_10"])

        def mock_get_observations(return_base_ohlc=False):
            obs = {"1m_10": np.random.randn(10, 4).astype(np.float32)}
            if return_base_ohlc:
                obs["base_features"] = np.array(
                    [[50000, 50100, 49900, 50050]] * 10, dtype=np.float32
                )
            return obs

        mock_observer.get_observations = MagicMock(side_effect=mock_get_observations)
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

        config = BitgetFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
            include_short_positions=True,
        )

        with patch("time.sleep"):
            with patch.object(BitgetFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesSLTPTorchTradingEnv(
                    config=config,
                    observer=mock_observer,
                    trader=mock_trader,
                )
                return env, mock_trader

    def test_long_to_long_ignored(self, env_with_mocks):
        """Test that Long → Long duplicate action is ignored."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # Open a long position (action 1 = long)
        env.position.current_position = 1

        # Try to open another long
        action_tuple = ("long", -0.02, 0.03)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Trade should NOT be executed (duplicate action)
        assert trade_info["executed"] is False
        mock_trader.trade.assert_not_called()
        mock_trader.close_position.assert_not_called()

    def test_short_to_short_ignored(self, env_with_mocks):
        """Test that Short → Short duplicate action is ignored."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # Open a short position
        env.position.current_position = -1

        # Try to open another short
        action_tuple = ("short", 0.03, -0.02)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Trade should NOT be executed (duplicate action)
        assert trade_info["executed"] is False
        mock_trader.trade.assert_not_called()
        mock_trader.close_position.assert_not_called()

    def test_long_to_short_switches_position(self, env_with_mocks):
        """Test that Long → Short switches position (closes long, opens short)."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # Start with long position
        env.position.current_position = 1

        # Switch to short
        action_tuple = ("short", 0.03, -0.02)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Should close existing position
        mock_trader.close_position.assert_called_once()
        # Should open new short position
        mock_trader.trade.assert_called_once()
        call_kwargs = mock_trader.trade.call_args.kwargs
        assert call_kwargs["side"] == "sell"

    def test_short_to_long_switches_position(self, env_with_mocks):
        """Test that Short → Long switches position (closes short, opens long)."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # Start with short position
        env.position.current_position = -1

        # Switch to long
        action_tuple = ("long", -0.02, 0.03)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Should close existing position
        mock_trader.close_position.assert_called_once()
        # Should open new long position
        mock_trader.trade.assert_called_once()
        call_kwargs = mock_trader.trade.call_args.kwargs
        assert call_kwargs["side"] == "buy"

    def test_hold_action_with_no_position(self, env_with_mocks):
        """Test that HOLD action does nothing when no position."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # No position
        env.position.current_position = 0

        # HOLD action
        action_tuple = (None, None, None)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Should not execute any trade
        assert trade_info["executed"] is False
        mock_trader.trade.assert_not_called()
        mock_trader.close_position.assert_not_called()

    def test_hold_action_maintains_position(self, env_with_mocks):
        """Test that HOLD action maintains existing position."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        # Start with long position
        env.position.current_position = 1

        # HOLD action
        action_tuple = (None, None, None)
        trade_info = env._execute_trade_if_needed(action_tuple)

        # Should not execute any trade or close position
        assert trade_info["executed"] is False
        mock_trader.trade.assert_not_called()
        mock_trader.close_position.assert_not_called()
        # Position should remain unchanged
        assert env.position.current_position == 1
