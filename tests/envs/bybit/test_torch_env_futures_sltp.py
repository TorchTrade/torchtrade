"""Tests for BybitFuturesSLTPTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestBybitFuturesSLTPTorchTradingEnv:
    """Tests for BybitFuturesSLTPTorchTradingEnv."""

    @pytest.fixture
    def env_config(self):
        """Create environment configuration."""
        from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTradingEnvConfig

        return BybitFuturesSLTPTradingEnvConfig(
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
    def env(self, env_config, mock_env_observer, mock_env_trader):
        """Create environment with mocks."""
        from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTorchTradingEnv

        with patch("time.sleep"):
            with patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                return BybitFuturesSLTPTorchTradingEnv(
                    config=env_config,
                    observer=mock_env_observer,
                    trader=mock_env_trader,
                )

    def test_action_map_structure(self, env):
        """Test action map: 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9 actions."""
        assert len(env.action_map) == 9
        assert env.action_spec.n == 9
        assert env.action_map[0] == (None, None, None)  # HOLD

    def test_action_map_long_actions(self, env):
        """Test long actions have negative SL and positive TP."""
        for i in range(1, 5):
            side, sl, tp = env.action_map[i]
            assert side == "long"
            assert sl < 0
            assert tp > 0

    def test_action_map_short_actions(self, env):
        """Test short actions have positive SL and negative TP."""
        for i in range(5, 9):
            side, sl, tp = env.action_map[i]
            assert side == "short"
            assert sl > 0  # SL above entry for shorts
            assert tp < 0  # TP below entry for shorts

    def test_action_spec_long_only(self, env_config, mock_env_observer, mock_env_trader):
        """Test action spec when short positions disabled: 1 HOLD + 4 LONG = 5."""
        from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTorchTradingEnv

        env_config.include_short_positions = False

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=env_config, observer=mock_env_observer, trader=mock_env_trader,
            )

        assert env.action_spec.n == 5

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys with correct shapes."""
        assert "account_state" in env.observation_spec.keys()
        assert "market_data_1m_10" in env.observation_spec.keys()
        assert env.observation_spec["account_state"].shape == (6,)

    def test_reset(self, env, mock_env_trader):
        """Test environment reset returns expected keys and resets SLTP state."""
        td = env.reset()

        assert "account_state" in td.keys()
        assert "market_data_1m_10" in td.keys()
        assert td["account_state"].shape == (6,)
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0
        mock_env_trader.cancel_open_orders.assert_called()

    def test_step_hold_action(self, env, mock_env_trader):
        """Test step with HOLD action does not trade."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            mock_env_trader.trade.assert_not_called()
            assert "reward" in next_td["next"].keys()

    @pytest.mark.parametrize("action_idx,expected_side", [
        (1, "buy"),   # LONG action
        (5, "sell"),  # SHORT action
    ])
    def test_step_bracket_order(self, env, mock_env_trader, action_idx, expected_side):
        """Test step with LONG/SHORT action places bracket order with SL/TP."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
            env.step(action_td)

            mock_env_trader.trade.assert_called()
            call_kwargs = mock_env_trader.trade.call_args[1]
            assert call_kwargs["side"] == expected_side
            assert "take_profit" in call_kwargs
            assert "stop_loss" in call_kwargs

    def test_sltp_prices_calculated_correctly(self, env, mock_env_trader):
        """Test that SL/TP prices are calculated from percentages."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            call_kwargs = mock_env_trader.trade.call_args[1]
            current_price = 50050.0

            expected_sl = current_price * (1 - 0.02)
            expected_tp = current_price * (1 + 0.03)

            assert call_kwargs["stop_loss"] == pytest.approx(expected_sl, rel=1e-2)
            assert call_kwargs["take_profit"] == pytest.approx(expected_tp, rel=1e-2)

    def test_active_sltp_tracking(self, env, mock_env_trader):
        """Test that active SL/TP levels are tracked after order."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            assert env.active_stop_loss > 0
            assert env.active_take_profit > 0

    def test_position_closed_resets_sltp(self, env, mock_env_trader):
        """Test that position closure resets SL/TP tracking."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            env.active_stop_loss = 49000.0
            env.active_take_profit = 51000.0
            env.position.current_position = 1

            mock_env_trader.get_status = MagicMock(return_value={"position_status": None})

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            env._step(action_td)

            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0
            assert env.position.current_position == 0

    def test_reward_and_done_tensor_shapes(self, env):
        """Test that reward and done flags have correct shapes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)

            assert next_td["next"]["reward"].shape == (1,)
            assert next_td["next"]["done"].shape == (1,)
            assert next_td["next"]["terminated"].shape == (1,)
            assert next_td["next"]["truncated"].shape == (1,)

    def test_bankruptcy_termination(self, env, mock_env_trader):
        """Test that environment terminates on bankruptcy."""
        mock_env_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 50.0, "available_balance": 50.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 50.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())
            next_td = env.step(action_td)
            assert next_td["next"]["done"].item() is True

    def test_no_trade_when_position_exists(self, env, mock_env_trader):
        """Test that no trade is placed when already in same position."""
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.001, notional_value=50.0, entry_price=50000.0,
                unrealized_pnl=0.5, unrealized_pnl_pct=0.01, mark_price=50500.0,
                leverage=5, margin_mode="isolated", liquidation_price=45000.0,
            )
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.position.current_position = 1

            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())
            env._step(action_td)

            mock_env_trader.trade.assert_not_called()

    def test_config_post_init(self):
        """Test config post_init normalization."""
        from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTradingEnvConfig

        config = BybitFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames="1m",
            window_sizes=10,
        )

        assert isinstance(config.time_frames, list)
        assert isinstance(config.window_sizes, list)
        assert all(isinstance(tf, TimeFrame) for tf in config.time_frames)


class TestBybitDuplicateActionPrevention:
    """Test duplicate action prevention and position switch logic."""

    @pytest.fixture
    def env_with_mocks(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.bybit.env_sltp import (
            BybitFuturesSLTPTorchTradingEnv,
            BybitFuturesSLTPTradingEnvConfig,
        )

        config = BybitFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
            include_short_positions=True,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )
            return env, mock_env_trader

    @pytest.mark.parametrize("position,action_tuple,should_trade", [
        (1, ("long", -0.02, 0.03), False),    # Long -> Long: ignored
        (-1, ("short", 0.03, -0.02), False),   # Short -> Short: ignored
        (0, (None, None, None), False),         # No pos + HOLD: ignored
        (1, (None, None, None), False),         # Long + HOLD: ignored
    ])
    def test_duplicate_and_hold_actions(self, env_with_mocks, position, action_tuple, should_trade):
        """Test that duplicate and hold actions don't trigger trades."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()

        env.position.current_position = position
        trade_info = env._execute_trade_if_needed(action_tuple)

        assert trade_info["executed"] is should_trade
        mock_trader.trade.assert_not_called()
        mock_trader.close_position.assert_not_called()

    @pytest.mark.parametrize("initial_pos,action_tuple,expected_side", [
        (1, ("short", 0.03, -0.02), "sell"),   # Long -> Short
        (-1, ("long", -0.02, 0.03), "buy"),    # Short -> Long
    ])
    def test_position_switch(self, env_with_mocks, initial_pos, action_tuple, expected_side):
        """Test position switching closes old and opens new."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()
        env.position.current_position = initial_pos

        env._execute_trade_if_needed(action_tuple)

        mock_trader.close_position.assert_called_once()
        mock_trader.trade.assert_called_once()
        assert mock_trader.trade.call_args.kwargs["side"] == expected_side
