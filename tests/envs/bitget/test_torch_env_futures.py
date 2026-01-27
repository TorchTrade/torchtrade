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

    def test_no_bankruptcy_when_disabled(self, env_config, mock_observer, mock_trader):
        """Test that bankruptcy check can be disabled."""
        from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv

        env_config.done_on_bankruptcy = False

        with patch("time.sleep"):
            with patch.object(BitgetFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesTorchTradingEnv(
                    config=env_config,
                    observer=mock_observer,
                    trader=mock_trader,
                )

        # Mock very low balance
        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 10.0,
            "available_balance": 10.0,
            "total_unrealized_profit": 0.0,
            "total_margin_balance": 10.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()

            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # 0.0
            next_td = env.step(action_td)

            done = next_td["next"]["done"]
            assert done.item() is False  # Should not terminate

    def test_close_position_action(self, env, mock_trader):
        """Test that action 1 (hold/close) closes existing position."""
        from torchtrade.envs.live.bitget.order_executor import PositionStatus

        # Mock existing long position
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

            # Action index 2 maps to level 0.0, which should close position
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            env._step(action_td)  # Use _step to test internal logic

            # Fractional action levels: [0=-1.0, 1=-0.5, 2=0.0, 3=0.5, 4=1.0]
            # Action 0.0 means close position, so if qty != 0, it should close

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
