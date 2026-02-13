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
        assert "market_data_1m_10" in obs_spec.keys()
        assert "market_data_5m_10" in obs_spec.keys()
        assert obs_spec["account_state"].shape == (6,)

    def test_reset(self, env, mock_trader):
        """Test environment reset returns expected keys and shapes."""
        td = env.reset()

        assert "account_state" in td.keys()
        assert "market_data_1m_10" in td.keys()
        assert "market_data_5m_10" in td.keys()
        assert td["account_state"].shape == (6,)
        assert td["market_data_1m_10"].shape == (10, 4)
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

    def test_no_bankruptcy_when_disabled(self, env_config, mock_observer, mock_trader):
        """Test that bankruptcy check can be disabled."""
        from torchtrade.envs.live.bybit.env import BybitFuturesTorchTradingEnv

        env_config.done_on_bankruptcy = False

        with patch("time.sleep"), \
             patch.object(BybitFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
            env = BybitFuturesTorchTradingEnv(
                config=env_config, observer=mock_observer, trader=mock_trader,
            )

        mock_trader.get_account_balance = MagicMock(return_value={
            "total_wallet_balance": 10.0, "available_balance": 10.0,
            "total_unrealized_profit": 0.0, "total_margin_balance": 10.0,
        })

        with patch.object(env, "_wait_for_next_timestamp"):
            env.reset()
            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())
            next_td = env.step(action_td)
            assert next_td["next"]["done"].item() is False

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
