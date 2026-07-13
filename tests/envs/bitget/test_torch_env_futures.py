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

        Regression: current_position/current_action_level are written only by the env's OWN
        trades, so a liquidation (or a manual close in the exchange UI) left them stale. The
        duplicate-action guard then silently no-op'd an agent that re-requested the level it
        used to hold -- and kept refusing for the REST of the episode.

        Both halves matter. The guard must still suppress a redundant trade while the
        position is genuinely held, or a fix that just resyncs on every step would pass.
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
