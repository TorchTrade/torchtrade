"""Tests for OKXFuturesSLTPTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestOKXFuturesSLTPTorchTradingEnv:
    """Tests for OKXFuturesSLTPTorchTradingEnv."""

    @pytest.fixture
    def env_config(self):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTradingEnvConfig
        return OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", demo=True, time_frames=["1m"], window_sizes=[10],
            execute_on="1m", leverage=5, stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06), include_short_positions=True,
            quantity_per_trade=0.001,
        )

    @pytest.fixture
    def env(self, env_config, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv

        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesSLTPTorchTradingEnv(
                config=env_config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_action_map_structure(self, env):
        """Test action map: 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9 actions."""
        assert len(env.action_map) == 9
        assert env.action_spec.n == 9
        assert env.action_map[0] == (None, None, None)  # HOLD

    def test_action_spec_long_only(self, env_config, mock_env_observer, mock_env_trader):
        """Test action spec when short positions disabled: 1 HOLD + 4 LONG = 5."""
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv
        env_config.include_short_positions = False

        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(
                config=env_config, observer=mock_env_observer, trader=mock_env_trader,
            )
        assert env.action_spec.n == 5

    def test_reset(self, env, mock_env_trader):
        """Test environment reset returns expected keys and resets SLTP state."""
        td = env.reset()
        assert "account_state" in td.keys()
        assert "market_data_1Minute_10" in td.keys()
        assert td["account_state"].shape == (6,)
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0

    def test_step_hold_action(self, env, mock_env_trader):
        """Test step with HOLD action does not trade."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(0)}, batch_size=()))
            mock_env_trader.trade.assert_not_called()
            assert "reward" in next_td["next"].keys()

    @pytest.mark.parametrize("action_idx,expected_side", [(1, "buy"), (5, "sell")])
    def test_step_bracket_order(self, env, mock_env_trader, action_idx, expected_side):
        """Test step with LONG/SHORT action places bracket order with SL/TP."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
            call_kwargs = mock_env_trader.trade.call_args[1]
            assert call_kwargs["side"] == expected_side
            assert "take_profit" in call_kwargs
            assert "stop_loss" in call_kwargs

    def test_bracket_uses_mark_price(self, env, mock_env_trader):
        """Bracket order SL/TP must be calculated from mark price, not candle close."""
        mock_env_trader.get_mark_price = MagicMock(return_value=51000.0)

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._execute_trade_if_needed(("long", -0.02, 0.03))
            call_kwargs = mock_env_trader.trade.call_args[1]
            # SL/TP should be based on mark price (51000), not candle close (50050)
            assert call_kwargs["stop_loss"] == pytest.approx(51000.0 * (1 - 0.02), rel=1e-4)
            assert call_kwargs["take_profit"] == pytest.approx(51000.0 * (1 + 0.03), rel=1e-4)

    def test_position_closed_resets_sltp(self, env, mock_env_trader):
        """Test that position closure resets SL/TP tracking."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

            env.active_stop_loss = 49000.0
            env.active_take_profit = 51000.0
            env.position.current_position = 1

            mock_env_trader.get_status = MagicMock(return_value={"position_status": None})
            env._step(TensorDict({"action": torch.tensor(0)}, batch_size=()))

            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0
            assert env.position.current_position == 0

    def test_reward_and_done_tensor_shapes(self, env):
        """Test that reward and done flags have correct shapes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(0)}, batch_size=()))
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
            next_td = env.step(TensorDict({"action": torch.tensor(0)}, batch_size=()))
            assert next_td["next"]["done"].item() is True

    def test_no_trade_when_position_exists(self, env, mock_env_trader):
        """Test that no trade is placed when already in same position."""
        from torchtrade.envs.live.okx.order_executor import PositionStatus

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
            env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
            mock_env_trader.trade.assert_not_called()


class TestOKXDuplicateActionPrevention:
    """Test duplicate action prevention and position switch logic."""

    @pytest.fixture
    def env_with_mocks(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), include_short_positions=True,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )
            return env, mock_env_trader

    @pytest.mark.parametrize("position,action_tuple,should_trade", [
        (1, ("long", -0.02, 0.03), False),
        (-1, ("short", 0.03, -0.02), False),
        (0, (None, None, None), False),
        (1, (None, None, None), False),
    ])
    def test_duplicate_and_hold_actions(self, env_with_mocks, position, action_tuple, should_trade):
        """Test that duplicate and hold actions don't trigger trades."""
        env, mock_trader = env_with_mocks
        env.reset()
        mock_trader.reset_mock()
        env.position.current_position = position
        trade_info = env._execute_trade_if_needed(action_tuple)
        assert trade_info["executed"] is should_trade

    @pytest.mark.parametrize("initial_pos,action_tuple,expected_side", [
        (1, ("short", 0.03, -0.02), "sell"),
        (-1, ("long", -0.02, 0.03), "buy"),
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


class TestOKXSLTPCloseAction:
    """Tests for close action when include_close_action=True."""

    @pytest.fixture
    def env_with_close(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), include_close_action=True,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )
        mock_env_trader.reset_mock()
        return env

    def test_close_action_in_action_map(self, env_with_close):
        """Close action must be present in action map at index 1."""
        assert env_with_close.action_map[0] == (None, None, None)
        assert env_with_close.action_map[1] == ("close", None, None)

    def test_close_action_closes_position(self, env_with_close, mock_env_trader):
        """Close action must close an existing position."""
        env_with_close.position.current_position = 1
        trade_info = env_with_close._execute_trade_if_needed(("close", None, None))
        assert trade_info["executed"] is True
        assert trade_info["closed_position"] is True

    def test_close_action_no_position(self, env_with_close, mock_env_trader):
        """Close action with no position should be a no-op."""
        env_with_close.position.current_position = 0
        trade_info = env_with_close._execute_trade_if_needed(("close", None, None))
        assert trade_info["executed"] is False


class TestOKXSLTPNotionalTradeMode:
    """Test notional (USD) trade mode for OKX SLTP environment."""

    @pytest.fixture
    def notional_env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
            include_short_positions=True, trade_mode="notional", quantity_per_trade=500.0,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("action_tuple,expected_side", [
        (("long", -0.02, 0.03), "buy"),
        (("short", 0.02, -0.03), "sell"),
    ], ids=["long-buy", "short-sell"])
    def test_notional_converts_usd_to_quantity(self, notional_env, mock_env_trader, action_tuple, expected_side):
        """Notional mode must convert USD to base-asset quantity using current price."""
        mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)
        with patch.object(notional_env, "_wait_for_next_timestamp"):
            notional_env.reset()
            notional_env._execute_trade_if_needed(action_tuple)
            call_kwargs = mock_env_trader.trade.call_args[1]
            assert call_kwargs["side"] == expected_side
            assert call_kwargs["quantity"] == pytest.approx(0.01, rel=1e-6)

    def test_notional_zero_price_aborts_trade(self, notional_env, mock_env_trader):
        """Zero mark price must abort trade without calling trader.trade()."""
        mock_env_trader.get_mark_price = MagicMock(return_value=0.0)
        with patch.object(notional_env, "_wait_for_next_timestamp"):
            notional_env.reset()
            trade_info = notional_env._execute_trade_if_needed(("long", -0.02, 0.03))
            mock_env_trader.trade.assert_not_called()
            assert trade_info["success"] is False

    def test_quantity_mode_passes_raw_value(self, mock_env_observer, mock_env_trader):
        """Quantity mode must pass quantity_per_trade directly without conversion."""
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
            trade_mode="quantity", quantity_per_trade=0.001,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._execute_trade_if_needed(("long", -0.02, 0.03))
            assert mock_env_trader.trade.call_args[1]["quantity"] == pytest.approx(0.001, rel=1e-6)

    def test_fractional_converts_balance_to_quantity(self, mock_env_observer, mock_env_trader):
        """Fractional mode must compute quantity from balance * fraction * leverage / price."""
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
            trade_mode="fractional", position_fraction=0.1, leverage=5,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)
        mock_env_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._execute_trade_if_needed(("long", -0.02, 0.03))
            # balance=1000 * fraction=0.1 * leverage=5 / price=50000 = 0.01
            assert mock_env_trader.trade.call_args[1]["quantity"] == pytest.approx(0.01, rel=1e-4)


class TestOKXSLTPLockPosition:
    """Test lock_position_until_sltp for OKX SLTP environment."""

    @pytest.fixture
    def locked_env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
            include_short_positions=True, lock_position_until_sltp=True,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_locked_ignores_switch_action(self, locked_env, mock_env_trader):
        """With lock=True, a short action while long should be ignored."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()
            locked_env._execute_trade_if_needed(("long", -0.02, 0.03))
            locked_env.position.current_position = 1
            mock_env_trader.reset_mock()

            trade_info = locked_env._execute_trade_if_needed(("short", 0.02, -0.03))
            assert trade_info["executed"] is False
            mock_env_trader.trade.assert_not_called()

    def test_locked_ignores_close_action(self, locked_env, mock_env_trader):
        """With lock=True, close action while in position should be ignored."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()
            locked_env.position.current_position = 1
            mock_env_trader.reset_mock()

            trade_info = locked_env._execute_trade_if_needed(("close", None, None))
            assert trade_info["executed"] is False
            mock_env_trader.close_position.assert_not_called()

    def test_locked_allows_open_from_flat(self, locked_env, mock_env_trader):
        """With lock=True, opening a position from flat should still work."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()
            assert locked_env.position.current_position == 0
            locked_env._execute_trade_if_needed(("long", -0.02, 0.03))
            mock_env_trader.trade.assert_called_once()


class TestOKXSLTPPositionClosedClobber:
    """Regression: position_closed must not overwrite a newly-opened position."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), include_short_positions=True,
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_new_trade_after_sltp_close_preserves_position(self, env, mock_env_trader):
        """When SL/TP closes a position and a new trade opens in the same step,
        the new position state must be preserved (not overwritten to 0)."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.position.current_position = 1  # Was long

            mock_env_trader.get_status = MagicMock(return_value={"position_status": None})
            mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)

            short_action_idx = len(env.action_map) - 1
            env._step(TensorDict({"action": torch.tensor(short_action_idx)}, batch_size=()))
            assert env.position.current_position == -1


class TestOKXSLTPActionIndexClamping:
    """Test SLTP action index clamping."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
        )
        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("action_idx", [-1, 99, float("nan")], ids=["negative", "too-high", "nan"])
    def test_invalid_action_index_handled(self, env, action_idx):
        """Out-of-range and NaN SLTP action indices must be handled without crashing."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
            assert "reward" in next_td["next"].keys()


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
        """Run a full multi-step episode with realistic price data."""
        from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = OKXFuturesSLTPTradingEnvConfig(
            symbol="BTC-USDT-SWAP", time_frames=["1m"], window_sizes=[10],
            execute_on="1m", stoploss_levels=(-0.02,), takeprofit_levels=(0.03,),
            leverage=5, trade_mode="quantity", quantity_per_trade=0.01,
        )
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
        observer = ReplayObserver(
            df=replay_df, time_frames=config.time_frames,
            window_sizes=config.window_sizes, execute_on=config.execute_on, executor=executor,
        )

        with patch("time.sleep"), \
             patch.object(OKXFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = OKXFuturesSLTPTorchTradingEnv(config=config, observer=observer, trader=executor)

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()
            for i in range(50):
                action = [0, 1, 0, 0, len(env.action_map) - 1, 0][i % 6]
                action_td = td.clone()
                action_td["action"] = torch.tensor(action)
                result = env.step(action_td)
                td = result["next"]
                assert td["account_state"].shape == (6,)
                if td["done"].item():
                    break
