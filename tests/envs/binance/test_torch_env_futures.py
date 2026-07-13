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

        Regression: current_position/current_action_level are written only by the env's OWN
        trades, so a liquidation (or a manual close in the exchange UI) left them stale. The
        duplicate-action guard then silently no-op'd an agent that re-requested the level it
        used to hold -- and kept refusing for the REST of the episode.

        Both halves matter. The guard must still suppress a redundant trade while the
        position is genuinely held, or a fix that just resyncs on every step would pass.
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
        """A dust residual on reset is flat -- internally AND in what the agent sees.

        An exchange can leave a float residue (1e-12) behind a full close. Read as an open
        position it poisons the vector the policy consumes: a direction and a holding_time and
        a distance-to-liquidation for a position that does not exist, at zero exposure. The
        policy never saw that combination in training.

        Behavioural on purpose: the structural guard that every _reset uses the shared rule
        can be dodged by moving the derivation into a helper. This cannot.
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
        exposure, direction, pnl, holding_time, leverage, dist_to_liq = td["account_state"].tolist()
        assert exposure == 0.0        # 500 notional attached to a position that is not there
        assert direction == 0.0
        assert pnl == 0.0             # a position that does not exist cannot be up 5.26%
        assert holding_time == 0.0    # nor can it have been held for a bar
        assert leverage == 5.0        # the CONFIG leverage, not the 20 on the residual
        assert dist_to_liq == 1.0     # no position -> no liquidation to be near

    def test_reset_clears_the_holding_time_of_the_previous_episode(self, env, mock_trader):
        """Reset must zero hold_counter, or episode 2 inherits episode 1's age.

        Asserting it on a FRESH env proves nothing -- PositionState already defaults it to 0.
        The counter has to be aged first. Without this, an agent opens the next episode seeing
        a position it has "held" for bars it never traded.
        """
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={"position_status": PositionStatus(
            qty=0.01, notional_value=500.0, entry_price=50000.0, unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0, mark_price=50000.0, leverage=5,
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

        # 0 here, unlike bitget/bybit/okx: binance increments hold_counter in _step, not in
        # _get_observation, so the reset observation does not count a bar. The bug either way
        # is the previous episode's age surviving the reset.
        assert env.position.hold_counter == 0, f"reset carried {aged} bars into the new episode"
        assert td["account_state"][3].item() == 0.0


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
