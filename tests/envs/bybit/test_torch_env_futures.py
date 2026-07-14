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

    def test_reset_reads_dust_as_flat(self, env, mock_trader):
        """A dust residual (1e-12) left behind a close must read as FLAT, not as a position.

        The fixture is hostile in every field on purpose: a zeroed one made every element
        read 0 whatever the code did.
        """
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

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

    def test_dust_between_positions_does_not_age_the_next_one(self, env, mock_trader):
        """A residual left between two positions must not carry the old age into the new one.
        """
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

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

    def test_a_direct_flip_does_not_age_the_new_position(self, env, mock_trader):
        from torchtrade.envs.live.bybit.order_executor import PositionStatus
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
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

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
        from torchtrade.envs.live.bybit.order_executor import PositionStatus

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


class TestBybitInitCleanup:
    """Test that __init__ flattens by default and respects close_position_on_init."""

    @pytest.mark.parametrize("close_on_init,expect_close", [
        (True, True),    # Default: close positions on startup
        (False, False),  # Opt-out: keep existing positions
    ])
    def test_init_close_position_configurable(
        self, mock_env_observer, mock_env_trader, close_on_init, expect_close
    ):
        """close_position_on_init controls whether positions are closed on startup."""
        from torchtrade.envs.live.bybit.env import (
            BybitFuturesTorchTradingEnv,
            BybitFuturesTradingEnvConfig,
        )

        config = BybitFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            time_frames=["1m"],
            window_sizes=[10],
            execute_on="1m",
            close_position_on_init=close_on_init,
        )

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            BybitFuturesTorchTradingEnv(
                config=config, observer=mock_env_observer, trader=mock_env_trader,
            )

        # cancel_open_orders always runs on init
        mock_env_trader.cancel_open_orders.assert_called_once()
        if expect_close:
            mock_env_trader.close_position.assert_called_once()
        else:
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

    @pytest.mark.parametrize("cancel_ok,close_ok", [
        (False, True),   # cancel fails
        (True, False),   # close fails
        (False, False),  # both fail
    ], ids=["cancel-fails", "close-fails", "both-fail"])
    def test_reset_logs_warning_on_cleanup_failure(self, mock_env_observer, mock_env_trader, cancel_ok, close_ok):
        """reset() must warn but not raise when cleanup calls return False."""
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

        mock_env_trader.cancel_open_orders = MagicMock(return_value=cancel_ok)
        mock_env_trader.close_position = MagicMock(return_value=close_ok)

        # Must not raise — reset proceeds despite cleanup failures
        obs = env.reset()
        assert obs is not None

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

    @pytest.mark.parametrize("first_action,second_action", [
        (0.5, 1.0),    # Scale up long
        (-0.5, -1.0),  # Scale up short
        (1.0, 0.5),    # Scale down long
        (1.0, 1.0),    # Same level: delegates to fractional (exchange-based)
        (0.0, 0.0),    # Both flat: delegates to fractional (exchange-based)
    ])
    def test_fractional_resizing_always_delegates(self, env, first_action, second_action):
        """_execute_trade_if_needed always delegates to exchange-based fractional sizing."""
        trade_result = {"executed": True, "amount": 0.01, "side": "buy",
                        "success": True, "closed_position": False}

        with patch.object(env, '_execute_fractional_action', return_value=trade_result) as mock_exec:
            env.position.current_action_level = first_action
            env._execute_trade_if_needed(second_action)
            mock_exec.assert_called_once_with(second_action)

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
        from torchtrade.envs.live.bybit.env import BybitFuturesTorchTradingEnv, BybitFuturesTradingEnvConfig
        from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

        config = BybitFuturesTradingEnvConfig(
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
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
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
