"""Tests for OKXFuturesTorchTradingEnv."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict


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
    def env_config(self):
        from torchtrade.envs.live.okx.env import OKXFuturesTradingEnvConfig
        return OKXFuturesTradingEnvConfig(
            symbol="BTC-USDT-SWAP", demo=True,
            time_frames=["1m", "5m"], window_sizes=[10, 10], execute_on="1m", leverage=5,
        )

    @pytest.fixture
    def env(self, env_config, mock_observer, mock_env_trader):
        from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv

        with patch("time.sleep"), \
             patch.object(OKXFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            return OKXFuturesTorchTradingEnv(
                config=env_config, observer=mock_observer, trader=mock_env_trader,
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

    def test_reset(self, env, mock_env_trader):
        """Test environment reset returns expected keys and shapes."""
        td = env.reset()
        assert "account_state" in td.keys()
        assert td["account_state"].shape == (6,)
        assert td["market_data_1Minute_10"].shape == (10, 4)
        mock_env_trader.cancel_open_orders.assert_called()

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
    def test_step_trade_action(self, env, mock_env_trader, action_idx, label):
        """Test step with long/short action calls trade."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            env.step(TensorDict({"action": torch.tensor(action_idx)}, batch_size=()))
            mock_env_trader.trade.assert_called()

    def test_reward_and_done_tensor_shapes(self, env):
        """Test that reward and done flags have correct tensor shapes."""
        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            next_td = env.step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
            assert next_td["next"]["reward"].shape == (1,)
            assert next_td["next"]["done"].shape == (1,)
            assert next_td["next"]["terminated"].shape == (1,)
            assert next_td["next"]["truncated"].shape == (1,)

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
            next_td = env.step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
            assert next_td["next"]["done"].item() is expected_done

    def test_reset_reads_dust_as_flat(self, env, mock_env_trader):
        """A dust residual (1e-12) left behind a close must read as FLAT, not as a position.

        The fixture is hostile in every field on purpose: a zeroed one made every element
        read 0 whatever the code did.
        """
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
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

    def test_a_direct_flip_does_not_age_the_new_position(self, env, mock_env_trader):
        from torchtrade.envs.live.okx.order_executor import PositionStatus
        from tests.envs.base_exchange_tests import (
            assert_a_direct_flip_does_not_age_the_new_position as assert_flip,
        )
        assert_flip(env, mock_env_trader, PositionStatus,
                    long_action=len(env.action_levels) - 1, short_action=0)

    def test_dust_between_positions_does_not_age_the_next_one(self, env, mock_env_trader):
        """A residual left between two positions must not carry the old age into the new one.
        """
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_mode="isolated", liquidation_price=45000.0,
            )}

        with patch.object(env, "_wait_for_next_timestamp"):
            mock_env_trader.get_status = MagicMock(return_value=status(0.01))
            env.reset()
            long_idx = len(env.action_levels) - 1     # index 1 is a half SHORT in these fixtures
            long = TensorDict({"action": torch.tensor(long_idx)}, batch_size=())

            for _ in range(5):                       # age a real position
                env.step(long)

            mock_env_trader.get_status = MagicMock(return_value=status(1e-12))   # closed -> dust
            env.step(long)

            mock_env_trader.get_status = MagicMock(return_value=status(0.01))    # a NEW position
            td = env.step(long)

        holding_time = td["next"]["account_state"][3].item()
        assert holding_time == 1.0, (
            f"a brand-new position is reported as {holding_time} bars old -- the dust bar "
            f"between the two did not reset the counter"
        )

    def test_reset_clears_the_holding_time_of_the_previous_episode(self, env, mock_env_trader):
        """Reset must zero hold_counter, or episode 2 inherits episode 1's age.

        Asserting it on a fresh env proves nothing (PositionState defaults it to 0), so the
        counter is aged first. Also pins that an OPEN position looks open.
        """
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        mock_env_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
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

        # 0: _get_observation() only READS hold_counter now (advance_hold_counter is called
        # exactly once per _step(), never from _reset()), so a reset -- even one that finds a
        # position still open on the exchange -- never itself counts a bar. The bug this test
        # used to pin was `_get_observation()` also advancing the counter when reset called it,
        # which read `aged + 1` (or, coincidentally with hold_counter zeroed first, always 1).
        assert env.position.hold_counter == 0, f"reset carried {aged} bars into the new episode"

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
        assert holding_time == 0.0

    def test_reentry_after_external_close_starts_a_fresh_holding_time(self, env, mock_env_trader):
        """A re-entry made in the SAME step as an external close must not inherit its age.

        The sync detects the close and lets the guard re-enter -- but if it does not discard
        hold_counter, the policy is handed a brand-new position as N+1 bars old.
        """
        from torchtrade.envs.live.okx.order_executor import PositionStatus

        def status(qty):
            return {"position_status": PositionStatus(
                qty=qty, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
                margin_mode="isolated", liquidation_price=45000.0,
            )} if qty else {"position_status": None}

        with patch.object(env, "_wait_for_next_timestamp"):
            long_idx = len(env.action_levels) - 1
            long = TensorDict({"action": torch.tensor(long_idx)}, batch_size=())

            mock_env_trader.get_status = MagicMock(return_value=status(0.01))
            env.reset()
            for _ in range(5):
                env.step(long)
            aged = env.position.hold_counter
            assert aged > 1

            mock_env_trader.get_status = MagicMock(return_value=status(None))   # liquidated
            td = env.step(long)                                          # same-step re-entry

        assert td["next"]["account_state"][3].item() <= 1.0, (
            f"a position opened after a liquidation inherited the dead position's age ({aged})"
        )


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


class TestWithReplayData:
    """Integration tests using ReplayObserver + ReplayOrderExecutor."""

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
