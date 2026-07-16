"""Tests for BinanceFuturesTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict


class TestBinanceFuturesTorchTradingEnv:
    """Tests for BinanceFuturesTorchTradingEnv."""

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
        trader.close_all_positions = MagicMock(return_value={})

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
        from torchtrade.envs.live.binance.env import BinanceFuturesTradingEnvConfig

        return BinanceFuturesTradingEnvConfig(
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
        from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv

        # Patch time.sleep to avoid waiting
        with patch("time.sleep"):
            with patch("torchtrade.envs.live.binance.env.BinanceFuturesTorchTradingEnv._wait_for_next_timestamp"):
                env = BinanceFuturesTorchTradingEnv(
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
        """Test action spec uses fractional mode with proper ordering."""
        # Should have fractional action levels (not exact count check)
        assert env.action_spec.n >= 3, "Should have at least 3 actions"

        # Verify action levels are fractional (floats between -1 and 1)
        action_levels = env.action_levels
        assert all(isinstance(level, (int, float)) for level in action_levels), \
            "Action levels should be numeric"
        assert all(-1 <= level <= 1 for level in action_levels), \
            f"Action levels should be in [-1, 1], got {action_levels}"

        # Verify proper ordering: negative values, then 0, then positive values
        # Should be sorted in ascending order: [-1, -0.5, 0, 0.5, 1] ✓
        # Should NOT be: [1, 0.5, 0, -0.5, -1] ✗
        assert action_levels == sorted(action_levels), \
            f"Action levels should be sorted ascending, got {action_levels}"

        # Verify no improper mixing (e.g., [-1, 0.5, 0, 0.5, 1] would be wrong)
        negatives = [x for x in action_levels if x < 0]
        positives = [x for x in action_levels if x > 0]
        zeros = [x for x in action_levels if x == 0]

        # If we have negatives and positives, zero should be between them
        if negatives and positives:
            assert len(zeros) > 0, "Should have 0 between negative and positive values"
            neg_indices = [i for i, x in enumerate(action_levels) if x < 0]
            pos_indices = [i for i, x in enumerate(action_levels) if x > 0]
            zero_indices = [i for i, x in enumerate(action_levels) if x == 0]

            # All negatives should come before zero, zero before positives
            assert max(neg_indices) < min(zero_indices), \
                "Negative values should come before zero"
            assert max(zero_indices) < min(pos_indices), \
                "Zero should come before positive values"

    def test_base_features_declared_in_observation_spec(self, env_config, mock_observer, mock_trader):
        """include_base_features=True must DECLARE base_features in observation_spec, not just
        emit it -- else spec and observation disagree and a collector pre-allocating from the
        spec silently drops it (#61)."""
        import dataclasses
        from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv

        config = dataclasses.replace(env_config, include_base_features=True)
        with patch("time.sleep"), patch.object(BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BinanceFuturesTorchTradingEnv(config=config, observer=mock_observer, trader=mock_trader)
            td = env.reset()

        assert "base_features" in env.observation_spec.keys()   # the fix (was missing on binance)
        assert "base_features" in td.keys()                     # emitted -> spec & obs consistent
        # shape must agree too: a collector pre-allocates buffers BY SHAPE from the spec
        assert env.observation_spec["base_features"].shape == td["base_features"].shape

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys."""
        obs_spec = env.observation_spec

        assert "account_state" in obs_spec.keys()
        assert "market_data_1m_10" in obs_spec.keys()
        assert "market_data_5m_10" in obs_spec.keys()
        assert "base_features" not in obs_spec.keys()   # off by default (mirror of #61)

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

            action_td = TensorDict({"action": torch.tensor(1)}, batch_size=())  # Hold
            next_td = env.step(action_td)

            # TorchRL step returns results under "next" key
            assert "reward" in next_td["next"].keys()
            assert "done" in next_td["next"].keys()
            assert "account_state" in next_td["next"].keys()

    def test_step_long_action(self, env, mock_trader):
        """Test step with long action."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            # Get the index for max positive action (e.g., 1.0)
            max_long_idx = len(env.action_levels) - 1
            action_td = TensorDict({"action": torch.tensor(max_long_idx)}, batch_size=())
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

    def test_close_method(self, env, mock_trader):
        """Test environment close method."""
        env.close()
        mock_trader.cancel_open_orders.assert_called()

    def test_reenters_after_external_position_close(self, env, mock_trader):
        """A position closed on the exchange must not leave the guard refusing to re-enter.

        Both halves matter: the guard must still suppress a genuinely redundant re-command,
        or a fix that resynced on every step would pass too.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

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
                margin_type="isolated", liquidation_price=45000.0,
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
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=1e-12, notional_value=500.0, entry_price=47500.0, unrealized_pnl=26.3,
            unrealized_pnl_pct=0.0526, mark_price=50000.0, leverage=20,
            margin_type="isolated", liquidation_price=48000.0,
        )})

        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

        assert env.position.current_position == 0

        # EVERY field on the residual is hostile on purpose. Earlier versions of this test
        # passed 0.0 for notional / pnl / liquidation_price -- the exact values that make the
        # position branch produce a flat-looking vector whatever the code does, so the bug it
        # was guarding could be deleted with the suite green.
        exposure, direction, pnl, _holding_time, leverage, dist_to_liq = td["account_state"].tolist()
        assert exposure == 0.0        # 500 notional attached to a position that is not there
        assert direction == 0.0
        assert pnl == 0.0             # a position that does not exist cannot be up 5.26%
        # NOT asserting holding_time: binance manages hold_counter in _step, so at reset it is 0
        # in BOTH branches whatever the dust rule does -- the assertion would be dead.
        assert leverage == 5.0        # the CONFIG leverage, not the 20 on the residual
        assert dist_to_liq == 1.0     # no position -> no liquidation to be near

    def test_a_direct_flip_does_not_age_the_new_position(self, env, mock_trader):
        from torchtrade.envs.live.binance.order_executor import PositionStatus
        from tests.envs.base_exchange_tests import (
            assert_a_direct_flip_does_not_age_the_new_position as assert_flip,
        )
        assert_flip(env, mock_trader, PositionStatus,
                    long_action=len(env.action_levels) - 1, short_action=0)

    def test_reset_clears_the_holding_time_of_the_previous_episode(self, env, mock_trader):
        """Reset must zero hold_counter, or episode 2 inherits episode 1's age.

        Asserting it on a fresh env proves nothing (PositionState defaults it to 0), so the
        counter is aged first. Also pins that an OPEN position looks open.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=0.01, notional_value=500.0, entry_price=47500.0, unrealized_pnl=26.3,
            unrealized_pnl_pct=0.0526, mark_price=50000.0, leverage=20,  # NOT the config's 5
            margin_type="isolated", liquidation_price=45000.0,
        )})

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            long_idx = len(env.action_levels) - 1     # index 1 is FLAT here ([-1, 0, 1])
            for _ in range(5):
                env.step(TensorDict({"action": torch.tensor(long_idx)}, batch_size=()))
            assert env.position.hold_counter > 0      # genuinely aged

            aged = env.position.hold_counter
            td = env.reset()                         # position still open on the exchange

        # 0: _get_observation() only READS hold_counter now (advance_hold_counter runs exactly
        # once per _step(), never from _reset(), which passes advance_hold=False), so a reset --
        # even one that finds a position still open on the exchange -- never itself counts a bar.
        assert env.position.hold_counter == 0, f"reset carried {aged} bars into the new episode"

        # An OPEN position must look OPEN. Every other account_state assertion on this branch
        # checks that a FLAT account reads flat -- the inverse was unpinned, and forcing
        # position_direction to 0 (so the flat branch is always taken) left the whole suite
        # green while handing the policy a healthy long as flat, unlevered and far from
        # liquidation. Values measured, not computed.
        exposure, direction, pnl, holding_time, leverage, dist_to_liq = td["account_state"].tolist()
        assert direction == 1.0
        assert pnl == pytest.approx(0.0526)          # was unpacked and never asserted
        assert exposure == 0.5                       # 500 notional / 1000 balance
        # 20, the POSITION's leverage -- not the config's 5. They are deliberately different:
        # with both set to 5 this assertion could not tell "reports the open position's
        # leverage" from a regression to "always reports the config's".
        assert leverage == 20.0
        assert dist_to_liq == pytest.approx(0.1)     # (50000 - 45000) / 50000
        assert holding_time == 0.0

    @pytest.mark.parametrize("qty,liq_price,expected_dtl", [
        (0.001, 45000.0, pytest.approx(0.1018, rel=1e-2)),   # long normal: (50100-45000)/50100
        (0.001, 0.0, 1.0),                                   # long zero-liq -> unknown -> 1.0
        (-0.001, 55000.0, pytest.approx(0.0978, rel=1e-2)),  # short normal: (55000-50100)/50100
        (-0.001, 0.0, 1.0),                                  # short zero-liq -> unknown -> 1.0 (the fix)
    ], ids=["long-normal", "long-zero-liq", "short-normal", "short-zero-liq"])
    def test_distance_to_liquidation(self, env, mock_trader, qty, liq_price, expected_dtl):
        """distance_to_liquidation (account_state[5]) across long/short x normal/zero-liq.

        A zero/absent liq price (cross-margin, or .get(..., 0)) must read 1.0 -- for a short the
        unguarded (0 - price)/price = 0.0 would falsely signal AT liquidation. Matches bybit/okx.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=qty, notional_value=50.1, entry_price=50000.0, unrealized_pnl=0.1,
            unrealized_pnl_pct=0.002, mark_price=50100.0, leverage=10,
            margin_type="isolated", liquidation_price=liq_price,
        )})
        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

        # direction proves a genuine OPEN position, so dtl==1.0 can't pass via the flat branch
        assert td["account_state"][1].item() == (1.0 if qty > 0 else -1.0)
        assert td["account_state"][5].item() == expected_dtl

    def test_exposure_pct_uses_equity_not_wallet_balance(self, env, mock_trader):
        """exposure_pct (account_state[0]) must divide by total_margin_balance (equity incl.
        uPnL), not total_wallet_balance. Binance's total_wallet_balance excludes uPnL, so using
        it read a different exposure than the other exchanges for the same position (#60)."""
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0,    # Binance: excludes unrealized PnL
            "available_balance": 900.0,
            "total_unrealized_profit": 100.0,
            "total_margin_balance": 1100.0,    # equity = wallet + uPnL
        })
        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=0.011, notional_value=550.0, entry_price=50000.0, unrealized_pnl=100.0,
            unrealized_pnl_pct=0.02, mark_price=50000.0, leverage=10,
            margin_type="isolated", liquidation_price=45000.0,
        )})
        with patch.object(env, "_wait_for_next_timestamp"):
            td = env.reset()

        assert td["account_state"][1].item() == 1.0                 # a genuine open long
        # equity: 550 / 1100 = 0.5. Wallet would give 550 / 1000 = 0.55 (excludes the +100 uPnL).
        assert td["account_state"][0].item() == pytest.approx(0.5)

    def test_closing_a_position_does_not_age_the_next_one(self, env, mock_trader):
        """A closed position's age must not carry into the next one.

        The exchange must report what the env holds -- the bare fixture says "no position"
        while the env believes it is long, and the sync (correctly) resets the counter on that
        disagreement every step, so it could never age.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_type="isolated", liquidation_price=45000.0,
            )} if qty else {"position_status": None}

        with patch.object(env, "_wait_for_next_timestamp"):
            long = TensorDict({"action": torch.tensor(2)}, batch_size=())   # levels [-1, 0, 1]
            flat = TensorDict({"action": torch.tensor(1)}, batch_size=())

            mock_trader.get_status = MagicMock(return_value=status(0.01))
            env.reset()
            for _ in range(5):
                env._step(long)
            assert env.position.hold_counter == 5           # genuinely aged

            mock_trader.get_status = MagicMock(return_value=status(None))   # closed
            env._step(flat)
            assert env.position.hold_counter == 0           # the age goes with it

            mock_trader.get_status = MagicMock(return_value=status(0.01))   # a BRAND NEW one
            env._step(long)

        assert env.position.hold_counter == 1, "a new position starts older than 1 bar"
    def test_reentry_after_external_close_starts_a_fresh_holding_time(self, env, mock_trader):
        """A re-entry made in the SAME step as an external close must not inherit its age.

        The sync detects the close and lets the guard re-enter -- but if it does not discard
        hold_counter, the policy is handed a brand-new position as N+1 bars old.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_type="isolated", liquidation_price=45000.0,
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


class TestBinanceFuturesTradingEnvConfig:
    """Tests for BinanceFuturesTradingEnvConfig."""

    def test_custom_config(self):
        """Test custom configuration."""
        from torchtrade.envs.live.binance.env import BinanceFuturesTradingEnvConfig
        from torchtrade.envs.live.binance.order_executor import MarginType

        config = BinanceFuturesTradingEnvConfig(
            symbol="ETHUSDT",
            leverage=10,
            margin_type=MarginType.CROSSED,
            demo=False,
        )

        assert config.symbol == "ETHUSDT"
        assert config.leverage == 10
        assert config.margin_type == MarginType.CROSSED
        assert config.demo is False


class TestBinanceFractionalPositionResizing:
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
        trader.close_all_positions = MagicMock(return_value={})
        trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 1000.0, "available_balance": 900.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 1000.0,
        })
        trader.get_mark_price = MagicMock(return_value=50000.0)
        trader.get_status = MagicMock(return_value={"position_status": None})
        trader.trade = MagicMock(return_value=True)
        return trader

    @pytest.fixture
    def env(self, mock_observer, mock_trader):
        from torchtrade.envs.live.binance.env import (
            BinanceFuturesTorchTradingEnv,
            BinanceFuturesTradingEnvConfig,
        )
        config = BinanceFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            time_frames=["1m", "5m"],
            window_sizes=[10, 10],
            execute_on="1m",
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        with patch("time.sleep"), \
             patch("torchtrade.envs.live.binance.env.BinanceFuturesTorchTradingEnv._wait_for_next_timestamp"):
            return BinanceFuturesTorchTradingEnv(
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
        trade_executed = {"executed": True, "amount": 0.01, "side": "BUY",
                         "success": True, "closed_position": False}

        with patch.object(env, '_execute_fractional_action', return_value=trade_executed) as mock_exec:
            env.position.current_action_level = first_action
            result = env._execute_trade_if_needed(second_action)

            if should_execute:
                mock_exec.assert_called_once_with(second_action)
            else:
                mock_exec.assert_not_called()
                assert result["executed"] is False


class TestMultipleSteps:
    """Test multiple environment steps."""

    @pytest.fixture
    def env_with_mocks(self):
        """Create environment for multi-step testing."""
        from torchtrade.envs.live.binance.env import (
            BinanceFuturesTorchTradingEnv,
            BinanceFuturesTradingEnvConfig,
        )

        mock_observer = MagicMock()
        mock_observer.get_keys = MagicMock(return_value=["1m_10"])
        mock_observer.get_observations = MagicMock(return_value={
            "1m_10": np.random.randn(10, 4).astype(np.float32),
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

        config = BinanceFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
        )

        with patch("time.sleep"):
            with patch.object(BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BinanceFuturesTorchTradingEnv(
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
                action = torch.randint(0, 3, (1,)).item()
                action_td = TensorDict({"action": torch.tensor(action)}, batch_size=())
                next_td = env_with_mocks.step(action_td)

                # TorchRL step returns results under "next" key
                assert "reward" in next_td["next"].keys()
                assert "done" in next_td["next"].keys()

    def test_rollout(self, env_with_mocks):
        """Test environment rollout."""
        with patch.object(env_with_mocks, "_wait_for_next_timestamp"):
            env_with_mocks.reset()

            rewards = []
            for _ in range(5):
                action = torch.randint(0, 3, (1,)).item()
                action_td = TensorDict({"action": torch.tensor(action)}, batch_size=())
                next_td = env_with_mocks.step(action_td)
                rewards.append(next_td["next", "reward"].item())

            assert len(rewards) == 5


class TestBinanceInitCleanup:
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
        from torchtrade.envs.live.binance.env import (
            BinanceFuturesTorchTradingEnv,
            BinanceFuturesTradingEnvConfig,
        )

        config = BinanceFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            close_position_on_init=close_on_init,
        )

        with patch("time.sleep"), \
             patch.object(BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            BinanceFuturesTorchTradingEnv(
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
        from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = BinanceFuturesTradingEnvConfig(
            symbol="BTCUSDT",
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
             patch.object(BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BinanceFuturesTorchTradingEnv(
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
