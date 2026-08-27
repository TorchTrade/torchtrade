"""Tests for BybitFuturesSLTPTorchTradingEnv."""

import pytest
import torch
from torchrl.envs.utils import check_env_specs
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame

from tests.envs.base_exchange_tests import (
    INVALID_ACTIONS,
    assert_an_invalid_action_cannot_move_an_open_position,
    assert_an_invalid_action_raises_before_trading,
)


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

    def test_step_emits_the_whole_done_family(self, env):
        from tests.envs.base_exchange_tests import (
            assert_the_step_emits_the_whole_done_family as assert_done_family,
        )
        assert_done_family(env)

    def test_check_env_specs_passes(self, env):
        """check_env_specs compares the emitted step against every declared spec;
        the done family comes from the spec on both sides here, so a narrowed done spec
        is NOT what this catches -- see assert_the_step_emits_the_whole_done_family."""
        with patch.object(type(env), "_wait_for_next_timestamp"):
            check_env_specs(env)

    def test_a_direct_flip_does_not_age_the_new_position(self, env, mock_env_trader):
        """A long flipped straight to a short (never through flat) is one step old (#49).

        Covers this SLTP env's _step -> _get_observation(advance_hold=True) aging path, which
        is otherwise untested here (the shared base._get_observation is exercised by the
        non-SLTP futures file, but this env_sltp._step is not)."""
        from torchtrade.envs.live.bybit.order_executor import PositionStatus
        from tests.envs.base_exchange_tests import (
            assert_a_direct_flip_does_not_age_the_new_position as assert_flip,
        )
        assert_flip(env, mock_env_trader, PositionStatus, long_action=1, short_action=5)

    def test_action_map_structure(self, env):
        """Test action map: 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9 actions."""
        assert len(env.action_map) == 9
        assert env.action_spec.n == 9
        assert env.action_map[0] == (None, None, None)  # HOLD

    def test_initial_portfolio_value_uses_margin_balance(self, env_config, mock_env_observer, mock_env_trader):
        """initial_portfolio_value (the bankruptcy baseline) must be equity
        (total_margin_balance), not total_wallet_balance -- matching offline's
        portfolio_value basis (#65)."""
        from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTorchTradingEnv

        mock_env_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0,
            "available_balance": 900.0,
            "total_unrealized_profit": 100.0,
            "total_margin_balance": 1100.0,
        })

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=env_config, observer=mock_env_observer, trader=mock_env_trader,
            )

        assert env.initial_portfolio_value == 1100.0

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
        assert "market_data_1Minute_10" in env.observation_spec.keys()
        assert env.observation_spec["account_state"].shape == (6,)

    def test_reset(self, env, mock_env_trader):
        """Test environment reset returns expected keys and resets SLTP state."""
        td = env.reset()

        assert "account_state" in td.keys()
        assert "market_data_1Minute_10" in td.keys()
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
            mark_price = 50000.0  # From mock_env_trader.get_mark_price()

            expected_sl = mark_price * (1 - 0.02)
            expected_tp = mark_price * (1 + 0.03)

            assert call_kwargs["stop_loss"] == pytest.approx(expected_sl, rel=1e-4)
            assert call_kwargs["take_profit"] == pytest.approx(expected_tp, rel=1e-4)

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

    @pytest.mark.parametrize("done_on_bankruptcy,expected_done", [
        (True, True),    # portfolio collapses below the threshold -> episode terminates
        (False, False),  # same collapse, check disabled -> keep trading
    ], ids=["enabled-terminates", "disabled-keeps-trading"])
    def test_bankruptcy_termination(self, env, mock_env_trader, done_on_bankruptcy, expected_done):
        """A collapsed portfolio ends the episode through _step iff done_on_bankruptcy.

        Threshold arithmetic is covered in tests/envs/test_live_env_base.py; the disabled
        case is this file's only guard against a _step that hardcodes done=True.
        """
        env.config.done_on_bankruptcy = done_on_bankruptcy

        mock_env_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 50.0,  # below 10% of the 1000 captured at __init__
            "available_balance": 50.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 50.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(0)}, batch_size=()))
            assert next_td["next"]["done"].item() is expected_done

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
        trade_info = env._execute_trade_if_needed(action_tuple, current_price=50000.0)

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

        env._execute_trade_if_needed(action_tuple, current_price=50000.0)

        mock_trader.close_position.assert_called_once()
        mock_trader.trade.assert_called_once()
        assert mock_trader.trade.call_args.kwargs["side"] == expected_side


class TestBybitSLTPCloseAction:
    """Tests for close action when include_close_action=True."""

    @pytest.fixture
    def env_with_close(self, mock_env_observer, mock_env_trader):
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
            include_close_action=True,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        # Reset mock call counts from __init__ cleanup so tests only see
        # calls from the action under test
        mock_env_trader.reset_mock()
        return env

class TestBybitSLTPMarkPrice:
    """Test that SLTP bracket orders use mark price instead of candle close."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
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
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_bracket_uses_mark_price(self, env, mock_env_trader):
        """Bracket order SL/TP must be calculated from mark price, not candle close."""
        mock_env_trader.get_mark_price = MagicMock(return_value=51000.0)

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            # Execute a long action with SL/TP
            env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=51000.0)

            call_kwargs = mock_env_trader.trade.call_args[1]
            # SL/TP should be based on mark price (51000), not candle close (50050)
            expected_sl = 51000.0 * (1 - 0.02)
            expected_tp = 51000.0 * (1 + 0.03)
            assert call_kwargs["stop_loss"] == pytest.approx(expected_sl, rel=1e-4)
            assert call_kwargs["take_profit"] == pytest.approx(expected_tp, rel=1e-4)


class TestBybitSLTPInvalidAction:
    """An invalid SLTP action must refuse to trade, not pick an endpoint."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
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
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    @pytest.mark.parametrize("action", INVALID_ACTIONS)
    def test_an_invalid_action_raises_before_trading(self, env, action):
        """Venue wiring: this `_step` must route through the shared validator (#288)."""
        assert_an_invalid_action_raises_before_trading(env, action)

    @pytest.mark.parametrize("action", INVALID_ACTIONS)
    def test_an_invalid_action_cannot_move_an_open_position(self, env, action):
        """The expensive direction -- every other case here starts flat (#288)."""
        assert_an_invalid_action_cannot_move_an_open_position(env, action)


class TestBybitSLTPPositionClosedClobber:
    """Regression: position_closed must not overwrite a newly-opened position."""

    @pytest.fixture
    def env(self, mock_env_observer, mock_env_trader):
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
            return BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_new_trade_after_sltp_close_preserves_position(self, env, mock_env_trader):
        """When SL/TP closes a position and a new trade opens in the same step,
        the new position state must be preserved (not overwritten to 0)."""
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.position.current_position = 1  # Was long

            # SL/TP triggered: first get_status call returns None (position closed),
            # subsequent calls also return None (flat after close, before new trade fills)
            mock_env_trader.get_status = MagicMock(return_value={"position_status": None})
            mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)

            # Action index for first short action (HOLD=0, LONG_1=1, SHORT_1=2 with 1 SL x 1 TP)
            short_action_idx = len(env.action_map) - 1  # Last action is a short
            action_td = TensorDict({"action": torch.tensor(short_action_idx)}, batch_size=())
            env._step(action_td)

            # The new short position must NOT be overwritten to 0 by position_closed
            assert env.position.current_position == -1


class TestBybitSLTPNotionalTradeMode:
    """Test notional (USD) trade mode for Bybit SLTP environment."""

    @pytest.fixture
    def notional_env(self, mock_env_observer, mock_env_trader):
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
            trade_mode="notional",
            quantity_per_trade=500.0,  # $500 USD per trade
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesSLTPTorchTradingEnv(
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
            notional_env._execute_trade_if_needed(action_tuple, current_price=50000.0)

            call_kwargs = mock_env_trader.trade.call_args[1]
            assert call_kwargs["side"] == expected_side
            # $500 / $50000 = 0.01 BTC
            assert call_kwargs["quantity"] == pytest.approx(0.01, rel=1e-6)

    def test_notional_zero_price_refuses_to_trade(self, notional_env, mock_env_trader):
        """A venue price of 0 is unusable data, not a graceful abort condition (#347).

        It used to return `success: False` -- indistinguishable from a legitimate refusal
        -- while the same value silently divided in the sizing path. It now raises, and
        the thing that matters is unchanged: no order is submitted.
        """
        mock_env_trader.get_mark_price = MagicMock(return_value=0.0)

        with patch.object(notional_env, "_wait_for_next_timestamp"):
            notional_env.reset()
            with pytest.raises(ValueError, match="unusable mark price"):
                notional_env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=0.0)

            mock_env_trader.trade.assert_not_called()

    def test_quantity_mode_passes_raw_value(self, mock_env_observer, mock_env_trader):
        """Quantity mode must pass quantity_per_trade directly without conversion."""
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
            trade_mode="quantity",
            quantity_per_trade=0.001,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=50000.0)

            call_kwargs = mock_env_trader.trade.call_args[1]
            assert call_kwargs["quantity"] == pytest.approx(0.001, rel=1e-6)

    def test_fractional_converts_balance_to_quantity(self, mock_env_observer, mock_env_trader):
        """Fractional mode sizes from balance * fraction * leverage / price, NET OF FEE.

        The raw figure asks for margin equal to the whole allocation, leaving nothing for
        the entry fee, so the affordability check refuses the open (#278). The shared
        sizer reserves it via fee_multiplier = 1 + leverage*fee, which is the same rule
        the non-SLTP path and the offline envs have always used. Asserted as the rule
        rather than a constant, so it tracks the venue's taker rate.
        """
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
            trade_mode="fractional",
            position_fraction=0.1,
            leverage=5,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        mock_env_trader.get_mark_price = MagicMock(return_value=50000.0)
        mock_env_trader.get_account_balance = MagicMock(return_value={
            # wallet != margin: sizing must use total_margin_balance (equity, incl.
            # unrealized PnL), matching offline's portfolio_value basis (#65).
            "total_wallet_balance": 1000.0,
            "available_balance": 900.0,
            "total_unrealized_profit": 100.0,
            "total_margin_balance": 1100.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=50000.0)

            call_kwargs = mock_env_trader.trade.call_args[1]
            # margin_balance=1100 * fraction=0.1 * leverage=5 / price=50000 = 0.011
            from torchtrade.envs.live.bybit.order_executor import TAKER_FEE
            expected = 0.011 * 0.98 / (1 + 5 * TAKER_FEE)

            # The ACCEPTED branch of the fee guard, which nothing else exercises: every
            # SLTP test passes a MagicMock trader, and a MagicMock is REJECTED, so they
            # all assert the venue-constant fallback. Tighten the guard to reject
            # everything and the suite stays green while #278 returns.
            baseline_qty = mock_env_trader.trade.call_args[1]["quantity"]
            mock_env_trader.transaction_fee = 0.002
            mock_env_trader.trade.reset_mock()
            env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=50000.0)
            assert mock_env_trader.trade.call_args[1]["quantity"] < baseline_qty, (
                "the env must reserve the trader's higher rate, sizing SMALLER than the "
                "venue constant -- otherwise the venue refuses the open"
            )
        assert call_kwargs["quantity"] == pytest.approx(expected, rel=1e-4)


class TestBybitSLTPLockPosition:
    """Test lock_position_until_sltp for Bybit SLTP environment."""

    @pytest.fixture
    def locked_env(self, mock_env_observer, mock_env_trader):
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
            lock_position_until_sltp=True,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            return BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

    def test_locked_ignores_switch_action(self, locked_env, mock_env_trader):
        """With lock=True, a short action while long should be ignored."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()

            # Open long first
            locked_env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=50000.0)
            mock_env_trader.trade.assert_called_once()
            locked_env.position.current_position = 1

            mock_env_trader.reset_mock()

            # Try to switch to short — should be ignored
            trade_info = locked_env._execute_trade_if_needed(("short", 0.02, -0.03), current_price=50000.0)

            assert trade_info["executed"] is False
            mock_env_trader.trade.assert_not_called()
            mock_env_trader.close_position.assert_not_called()

    def test_locked_ignores_close_action(self, locked_env, mock_env_trader):
        """With lock=True, close action while in position should be ignored."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()
            locked_env.position.current_position = 1
            mock_env_trader.reset_mock()  # Clear calls from reset/init

            trade_info = locked_env._execute_trade_if_needed(("close", None, None), current_price=50000.0)

            assert trade_info["executed"] is False
            mock_env_trader.close_position.assert_not_called()

    def test_locked_allows_open_from_flat(self, locked_env, mock_env_trader):
        """With lock=True, opening a position from flat should still work."""
        with patch.object(locked_env, "_wait_for_next_timestamp"):
            locked_env.reset()
            assert locked_env.position.current_position == 0

            trade_info = locked_env._execute_trade_if_needed(("long", -0.02, 0.03), current_price=50000.0)

            mock_env_trader.trade.assert_called_once()

    def test_unlocked_allows_switch(self, mock_env_observer, mock_env_trader):
        """With lock=False (default), switching positions works normally."""
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
            lock_position_until_sltp=False,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.position.current_position = 1

            # Switch to short — should work (calls close + trade)
            env._execute_trade_if_needed(("short", 0.02, -0.03), current_price=50000.0)
            mock_env_trader.close_position.assert_called()
            mock_env_trader.trade.assert_called()


class TestWithReplayData:
    """Integration tests using ReplayObserver + ReplayOrderExecutor with real price data."""

    @pytest.fixture
    def replay_df(self):
        """Create realistic OHLCV test data with price movement."""
        import numpy as np
        import pandas as pd

        n = 200
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        base = 50000 + np.cumsum(np.random.default_rng(42).normal(0, 50, n))
        # close drawn off base can land outside a high/low drawn off base alone (#326).
        close = base + np.random.default_rng(45).normal(0, 20, n)
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": base,
            "high": np.maximum(base + np.abs(np.random.default_rng(43).normal(30, 20, n)), np.maximum(base, close)),
            "low": np.minimum(base - np.abs(np.random.default_rng(44).normal(30, 20, n)), np.minimum(base, close)),
            "close": close,
            "volume": np.random.default_rng(46).uniform(100, 1000, n),
        })

    def test_multi_step_episode_with_replay(self, replay_df):
        """Run a full multi-step episode with realistic price data."""
        from torchtrade.envs.live.bybit.env_sltp import (
            BybitFuturesSLTPTorchTradingEnv,
            BybitFuturesSLTPTradingEnvConfig,
        )
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

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
            df=replay_df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            executor=executor,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=observer, trader=executor,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

            for i in range(50):
                action = [0, 1, 0, 0, len(env.action_map) - 1, 0][i % 6]
                action_td = td.clone()
                action_td["action"] = torch.tensor(action)
                result = env.step(action_td)
                td = result["next"]

                assert "reward" in td.keys()
                assert "done" in td.keys()
                assert td["account_state"].shape == (6,)

                if td["done"].item():
                    break

    def test_replay_portfolio_tracks_price_movement(self, replay_df):
        """Portfolio value should change with price movement, not stay static."""
        from torchtrade.envs.live.bybit.env_sltp import (
            BybitFuturesSLTPTorchTradingEnv,
            BybitFuturesSLTPTradingEnvConfig,
        )
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = BybitFuturesSLTPTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            stoploss_levels=(-0.05,),
            takeprofit_levels=(0.05,),
            leverage=5,
            trade_mode="quantity",
            quantity_per_trade=0.01,
        )

        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
        observer = ReplayObserver(
            df=replay_df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            executor=executor,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesSLTPTorchTradingEnv(
                config=config, observer=observer, trader=executor,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

            # Open a long position
            action_td = td.clone()
            action_td["action"] = torch.tensor(1)
            td = env.step(action_td)["next"]

            # Hold for several steps -- price should move, changing portfolio value
            balances = []
            for _ in range(10):
                action_td = td.clone()
                action_td["action"] = torch.tensor(0)  # HOLD
                td = env.step(action_td)["next"]
                balances.append(executor.get_account_balance()["total_wallet_balance"])

            # With real price movement, portfolio value should not stay static
            assert max(balances) != min(balances), "Portfolio value should vary with price movement"
