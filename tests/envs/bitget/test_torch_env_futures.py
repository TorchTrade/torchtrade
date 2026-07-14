"""Tests for BitgetFuturesTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict

from torchtrade.envs import TimeFrame


class TestBitgetFuturesTorchTradingEnv:
    """Tests for BitgetFuturesTorchTradingEnv."""

    @pytest.fixture
    def mock_observer(self):
        """Create a mock observer."""
        observer = MagicMock()

        # Mock get_keys
        observer.get_keys = MagicMock(return_value=["1m_10", "5m_10"])

        # Mock get_observations
        def mock_observations(return_base_ohlc=False):
            obs = {
                "1m_10": np.random.randn(10, 4).astype(np.float32),
                "5m_10": np.random.randn(10, 4).astype(np.float32),
            }
            if return_base_ohlc:
                obs["base_features"] = np.random.randn(10, 4).astype(np.float32)
                obs["base_timestamps"] = np.arange(10)
            return obs

        observer.get_observations = MagicMock(side_effect=mock_observations)
        observer.intervals = ["1m", "5m"]
        observer.window_sizes = [10, 10]

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
        trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        trader._round_amount = MagicMock(side_effect=lambda amount: amount)

        trader.get_status = MagicMock(return_value={
            "position_status": None,
        })

        trader.trade = MagicMock(return_value=True)

        return trader

    @pytest.fixture
    def env_config(self):
        """Create environment configuration."""
        from torchtrade.envs.live.bitget.env import BitgetFuturesTradingEnvConfig

        return BitgetFuturesTradingEnvConfig(
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
        from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv

        # Patch time.sleep to avoid waiting
        with patch("time.sleep"):
            with patch.object(BitgetFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesTorchTradingEnv(
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

    def test_action_spec(self, env):
        """Test action spec is correctly defined."""
        assert env.action_spec.n == 5  # fractional: -1.0, -0.5, 0.0, 0.5, 1.0

    def test_action_levels(self, env):
        """Test action levels are correctly set."""
        assert env.action_levels == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys."""
        obs_spec = env.observation_spec

        assert "account_state" in obs_spec.keys()
        assert "market_data_1m_10" in obs_spec.keys()
        assert "market_data_5m_10" in obs_spec.keys()

    def test_account_state_shape(self, env):
        """Test account state has correct shape (6 elements)."""
        obs_spec = env.observation_spec
        assert obs_spec["account_state"].shape == (6,)

    def test_reset(self, env, mock_trader):
        """Test environment reset."""
        td = env.reset()

        assert "account_state" in td.keys()
        assert "market_data_1m_10" in td.keys()
        assert "market_data_5m_10" in td.keys()

        mock_trader.cancel_open_orders.assert_called()

    def test_reset_observation_shapes(self, env):
        """Test observation shapes after reset."""
        td = env.reset()

        assert td["account_state"].shape == (6,)
        assert td["market_data_1m_10"].shape == (10, 4)
        assert td["market_data_5m_10"].shape == (10, 4)

    def test_step_hold_action(self, env, mock_trader):
        """Test step with hold action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # Hold/Close (0.0)
            next_td = env.step(action_td)

            # TorchRL step returns results under "next" key
            assert "reward" in next_td["next"].keys()
            assert "done" in next_td["next"].keys()
            assert "account_state" in next_td["next"].keys()

    def test_step_long_action(self, env, mock_trader):
        """Test step with long action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(4)}, batch_size=())  # Long (1.0)
            next_td = env.step(action_td)

            # Trade should have been attempted
            mock_trader.trade.assert_called()

    def test_step_short_action(self, env, mock_trader):
        """Test step with short action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(0)}, batch_size=())  # Short
            next_td = env.step(action_td)

            mock_trader.trade.assert_called()

    def test_reward_tensor_shape(self, env):
        """Test that reward is returned as tensor with correct shape."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # 0.0
            next_td = env.step(action_td)

            reward = next_td["next"]["reward"]
            assert isinstance(reward, torch.Tensor)
            assert reward.shape == (1,)

    def test_done_tensor_shape(self, env):
        """Test that done flags are tensors with correct shape."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # 0.0
            next_td = env.step(action_td)

            done = next_td["next"]["done"]
            terminated = next_td["next"]["terminated"]
            truncated = next_td["next"]["truncated"]

            assert isinstance(done, torch.Tensor)
            assert isinstance(terminated, torch.Tensor)
            assert isinstance(truncated, torch.Tensor)
            assert done.shape == (1,)

    @pytest.mark.parametrize("done_on_bankruptcy,expected_done", [
        (True, True),    # portfolio collapses below the threshold -> episode terminates
        (False, False),  # same collapse, check disabled -> keep trading
    ], ids=["enabled-terminates", "disabled-keeps-trading"])
    def test_bankruptcy_termination(self, env, mock_trader, done_on_bankruptcy, expected_done):
        """A collapsed portfolio ends the episode through _step iff done_on_bankruptcy.

        Threshold arithmetic is covered in tests/envs/test_live_env_base.py; the disabled
        case is this file's only guard against a _step that hardcodes done=True.
        """
        env.config.done_on_bankruptcy = done_on_bankruptcy

        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 50.0,  # below 10% of the 1000 captured at __init__
            "available_balance": 50.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 50.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
            assert next_td["next"]["done"].item() is expected_done

    def test_close_position_action(self, env, mock_trader):
        """Commanding flat (level 0.0) must close a position the env did NOT open.

        Regression: a stale current_action_level let the duplicate-action guard
        short-circuit the close, silently leaving the position open.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        # A long position already open on the exchange (not opened by this env)
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
            # The constructor closes any position (close_position_on_init); drop that call
            # so assert_called_once() counts only what _step does.
            mock_trader.close_position.reset_mock()

            # Fractional action levels: [0=-1.0, 1=-0.5, 2=0.0, 3=0.5, 4=1.0]
            # Action index 2 -> level 0.0 -> close the open position.
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            env._step(action_td)

        mock_trader.close_position.assert_called_once()

    def test_long_from_short(self, env, mock_trader):
        """Test going long from short position executes correct trade."""
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        # Mock existing short position
        mock_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=-0.001,  # Short
                notional_value=50.0,
                entry_price=50000.0,
                unrealized_pnl=0.5,
                unrealized_pnl_pct=0.01,
                mark_price=49500.0,
                leverage=5,
                margin_mode="isolated",
                liquidation_price=55000.0,
            )
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Go long (should buy to flip from short to long)
            action_td = TensorDict({"action": torch.tensor(4)}, batch_size=())  # Long (1.0)
            env._step(action_td)

            # With fractional sizing, it should call trade() with the delta amount to flip position
            # No need to call close_position() separately - just execute one trade
            mock_trader.trade.assert_called()
            # Verify it's a buy order (to flip from short to long)
            call_kwargs = mock_trader.trade.call_args[1]
            assert call_kwargs["side"] == "buy"

    def test_config_post_init(self):
        """Test config post_init normalization."""
        from torchtrade.envs.live.bitget.env import BitgetFuturesTradingEnvConfig

        config = BitgetFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames="1m",  # Single string
            window_sizes=10,  # Single int
        )

        assert isinstance(config.time_frames, list)
        assert isinstance(config.window_sizes, list)
        assert len(config.time_frames) == 1
        assert all(isinstance(tf, TimeFrame) for tf in config.time_frames)
        assert config.window_sizes == [10]

    def test_reenters_after_external_position_close(self, env, mock_trader):
        """A position closed on the exchange must not leave the guard refusing to re-enter.

        Both halves matter: the guard must still suppress a genuinely redundant re-command,
        or a fix that resynced on every step would pass too.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            long_idx = len(env.action_levels) - 1

            # 1. Agent opens a long.
            env.step(TensorDict({"action": torch.tensor(long_idx)}, batch_size=()))
            mock_trader.trade.assert_called()

            # 2. Exchange confirms the position. Re-commanding the SAME level is redundant.
            mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
                qty=0.01, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_mode="isolated", liquidation_price=45000.0,
            )})
            mock_trader.trade.reset_mock()
            env.step(TensorDict({"action": torch.tensor(long_idx)}, batch_size=()))
            mock_trader.trade.assert_not_called()   # guard still works

            # 3. The exchange liquidates it out from under us.
            mock_trader.get_status = MagicMock(return_value={"position_status": None})
            mock_trader.trade.reset_mock()
            env.step(TensorDict({"action": torch.tensor(long_idx)}, batch_size=()))

            # The agent still wants to be long -> the env must actually re-enter.
            mock_trader.trade.assert_called()

    def test_reset_reads_dust_as_flat(self, env, mock_trader):
        """A dust residual (1e-12) left behind a close must read as FLAT, not as a position.

        The fixture is hostile in every field on purpose: a zeroed one made every element
        read 0 whatever the code did.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=1e-12, notional_value=500.0, entry_price=47500.0, unrealized_pnl=26.3,
            unrealized_pnl_pct=0.0526, mark_price=50000.0, leverage=20,
            margin_mode="isolated", liquidation_price=48000.0,
        )})

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

        assert env.position.current_position == 0

        # EVERY field on the residual is hostile on purpose. Earlier versions of this test
        # passed 0.0 for notional / pnl / liquidation_price -- the exact values that make the
        # position branch produce a flat-looking vector whatever the code does, so the bug it
        # was guarding could be deleted with the suite green.
        exposure, direction, pnl, holding_time, leverage, dist_to_liq = td["account_state"].tolist()
        assert exposure == 0.0        # 500 notional attached to a position that is not there
        assert direction == 0.0
        assert pnl == 0.0             # a position that does not exist cannot be up 5.26%
        assert holding_time == 0.0    # nor can it have been held for a bar
        assert leverage == 5.0        # the CONFIG leverage, not the 20 on the residual
        assert dist_to_liq == 1.0     # no position -> no liquidation to be near

    def test_a_direct_flip_does_not_age_the_new_position(self, env, mock_trader):
        from torchtrade.envs.live.bitget.order_executor import PositionStatus
        from tests.envs.base_exchange_tests import (
            assert_a_direct_flip_does_not_age_the_new_position as assert_flip,
        )
        assert_flip(env, mock_trader, PositionStatus,
                    long_action=len(env.action_levels) - 1, short_action=0)

    def test_dust_between_positions_does_not_age_the_next_one(self, env, mock_trader):
        """A residual left between two positions must not carry the old age into the new one.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_mode="isolated", liquidation_price=45000.0,
            )}

        with patch.object(env, "_wait_for_next_timestamp"):
            mock_trader.get_status = MagicMock(return_value=status(0.01))
            env.reset()
            long_idx = len(env.action_levels) - 1     # index 1 is a half SHORT in these fixtures
            long = TensorDict({"action": torch.tensor(long_idx)}, batch_size=())

            for _ in range(5):                       # age a real position
                env.step(long)

            mock_trader.get_status = MagicMock(return_value=status(1e-12))   # closed -> dust
            env.step(long)

            mock_trader.get_status = MagicMock(return_value=status(0.01))    # a NEW position
            td = env.step(long)

        holding_time = td["next"]["account_state"][3].item()
        assert holding_time == 1.0, (
            f"a brand-new position is reported as {holding_time} bars old -- the dust bar "
            f"between the two did not reset the counter"
        )

    def test_reset_clears_the_holding_time_of_the_previous_episode(self, env, mock_trader):
        """Reset must zero hold_counter, or episode 2 inherits episode 1's age.

        Asserting it on a fresh env proves nothing (PositionState defaults it to 0), so the
        counter is aged first. Also pins that an OPEN position looks open.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=0.01, notional_value=500.0, entry_price=47500.0, unrealized_pnl=26.3,
            unrealized_pnl_pct=0.0526, mark_price=50000.0, leverage=20,  # NOT the config's 5
            margin_mode="isolated", liquidation_price=45000.0,
        )})

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            long_idx = len(env.action_levels) - 1     # index 1 is a half SHORT in these fixtures
            for _ in range(5):
                env.step(TensorDict({"action": torch.tensor(long_idx)}, batch_size=()))
            assert env.position.hold_counter > 0      # genuinely aged

            aged = env.position.hold_counter
            td = env.reset()                         # position still open on the exchange

        # 1, not 0: _reset zeroes the counter and then takes an observation, which legitimately
        # counts the still-open position as bar ONE of the new episode. The bug is it reading
        # `aged + 1` -- the previous episode's age carried across the reset.
        assert env.position.hold_counter == 1, f"reset carried {aged} bars into the new episode"

        # An OPEN position must look OPEN. Every other account_state assertion on this branch
        # checks that a FLAT account reads flat; the inverse was unpinned here, so corrupting
        # any of these while genuinely open shipped with the suite green. The position's
        # leverage (20) deliberately differs from the config's (5) -- with both at 5 the
        # assertion could not tell the open branch from the flat one.
        exposure, direction, pnl, holding_time, leverage, dist_to_liq = td["account_state"].tolist()
        assert direction == 1.0
        assert exposure == 0.5                        # 500 notional / 1000 balance
        assert pnl == pytest.approx(0.0526)
        assert leverage == 20.0                       # the POSITION's, not the config's 5
        assert dist_to_liq == pytest.approx(0.1)      # (50000 - 45000) / 50000
        assert holding_time == 1.0

    def test_reentry_after_external_close_starts_a_fresh_holding_time(self, env, mock_trader):
        """A re-entry made in the SAME step as an external close must not inherit its age.

        The sync detects the close and lets the guard re-enter -- but if it does not discard
        hold_counter, the policy is handed a brand-new position as N+1 bars old.
        """
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_mode="isolated", liquidation_price=45000.0,
            )} if qty else {"position_status": None}

        with patch.object(env, "_wait_for_next_timestamp"):
            long_idx = len(env.action_levels) - 1
            long = TensorDict({"action": torch.tensor(long_idx)}, batch_size=())

            mock_trader.get_status = MagicMock(return_value=status(0.01))
            env.reset()
            for _ in range(5):
                env.step(long)
            aged = env.position.hold_counter
            assert aged > 1

            mock_trader.get_status = MagicMock(return_value=status(None))   # liquidated
            td = env.step(long)                                          # same-step re-entry

        assert td["next"]["account_state"][3].item() <= 1.0, (
            f"a position opened after a liquidation inherited the dead position's age ({aged})"
        )


class TestBitgetFractionalPositionResizing:
    """Tests for fractional position resizing (regression for #155)."""

    @pytest.fixture
    def mock_observer(self):
        observer = MagicMock()
        observer.get_keys = MagicMock(return_value=["1m_10", "5m_10"])
        def mock_observations(return_base_ohlc=False):
            obs = {
                "1m_10": np.random.randn(10, 4).astype(np.float32),
                "5m_10": np.random.randn(10, 4).astype(np.float32),
            }
            if return_base_ohlc:
                obs["base_features"] = np.random.randn(10, 4).astype(np.float32)
                obs["base_timestamps"] = np.arange(10)
            return obs
        observer.get_observations = MagicMock(side_effect=mock_observations)
        observer.intervals = ["1m", "5m"]
        observer.window_sizes = [10, 10]
        return observer

    @pytest.fixture
    def mock_trader(self):
        trader = MagicMock()
        trader.cancel_open_orders = MagicMock(return_value=True)
        trader.close_position = MagicMock(return_value=True)
        trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })
        trader.get_mark_price = MagicMock(return_value=50000.0)
        trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        trader._round_amount = MagicMock(side_effect=lambda amount: amount)
        trader.get_status = MagicMock(return_value={"position_status": None})
        trader.trade = MagicMock(return_value=True)
        return trader

    @pytest.fixture
    def env(self, mock_observer, mock_trader):
        from torchtrade.envs.live.bitget.env import (
            BitgetFuturesTorchTradingEnv,
            BitgetFuturesTradingEnvConfig,
        )
        config = BitgetFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m", "5m"],
            window_sizes=[10, 10],
            execute_on="1m",
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        with patch("time.sleep"), \
             patch("torchtrade.envs.live.bitget.env.BitgetFuturesTorchTradingEnv._wait_for_next_timestamp"):
            return BitgetFuturesTorchTradingEnv(
                config=config, observer=mock_observer, trader=mock_trader,
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


class TestBitgetFuturesTorchTradingEnvIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection and credentials")
    def test_live_environment(self):
        """Test environment with live Bitget testnet."""
        import os
        from torchtrade.envs.live.bitget.env import (
            BitgetFuturesTorchTradingEnv,
            BitgetFuturesTradingEnvConfig,
        )

        config = BitgetFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            leverage=5,
        )

        env = BitgetFuturesTorchTradingEnv(
            config=config,
            api_key=os.getenv("BITGET_API_KEY"),
            api_secret=os.getenv("BITGET_SECRET"),
            api_passphrase=os.getenv("BITGET_PASSPHRASE"),
        )

        td = env.reset()
        assert "account_state" in td.keys()

        action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # 0.0
        next_td = env.step(action_td)
        assert "reward" in next_td["next"].keys()


class TestBitgetInitCleanup:
    """Test that __init__ flattens by default and respects close_position_on_init."""

    @pytest.fixture
    def mock_observer(self):
        observer = MagicMock()
        observer.get_keys = MagicMock(return_value=["1m_10"])
        observer.get_observations = MagicMock(return_value={
            "1m_10": np.random.randn(10, 4).astype(np.float32),
        })
        return observer

    @pytest.fixture
    def mock_trader(self):
        trader = MagicMock()
        trader.cancel_open_orders = MagicMock(return_value=True)
        trader.close_position = MagicMock(return_value=True)
        trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })
        trader.get_mark_price = MagicMock(return_value=50000.0)
        trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
        trader._round_amount = MagicMock(side_effect=lambda amount: amount)
        trader.get_status = MagicMock(return_value={"position_status": None})
        return trader

    @pytest.mark.parametrize("close_on_init,expect_close", [
        (True, True),
        (False, False),
    ])
    def test_init_close_position_configurable(
        self, mock_observer, mock_trader, close_on_init, expect_close
    ):
        """close_position_on_init controls whether positions are closed on startup."""
        from torchtrade.envs.live.bitget.env import (
            BitgetFuturesTorchTradingEnv,
            BitgetFuturesTradingEnvConfig,
        )

        config = BitgetFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            close_position_on_init=close_on_init,
        )

        with patch("time.sleep"), \
             patch.object(BitgetFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            BitgetFuturesTorchTradingEnv(
                config=config, observer=mock_observer, trader=mock_trader,
            )

        mock_trader.cancel_open_orders.assert_called_once()
        if expect_close:
            mock_trader.close_position.assert_called_once()
        else:
            mock_trader.close_position.assert_not_called()


class TestWithReplayData:
    """Integration tests using ReplayObserver + ReplayOrderExecutor with real price data."""

    @pytest.fixture
    def replay_df(self):
        """Create realistic OHLCV test data."""
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
        from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = BitgetFuturesTradingEnvConfig(
            symbol="BTC/USDT:USDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            leverage=5,
            demo=True,
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
             patch.object(BitgetFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BitgetFuturesTorchTradingEnv(
                config=config, observer=observer, trader=executor,
            )

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

            for i in range(20):
                # Cycle through action levels (hold, long, hold, short, hold, close...)
                action_idx = i % len(env.action_levels)
                action_td = td.clone()
                action_td["action"] = torch.tensor(action_idx)
                result = env.step(action_td)
                td = result["next"]

                assert "reward" in td.keys()
                assert "done" in td.keys()
                assert td["account_state"].shape == (6,)

                if td["done"].item():
                    break

            assert executor.current_price > 0
