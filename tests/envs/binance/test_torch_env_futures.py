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
        from torchtrade.envs.binance.torch_env_futures import BinanceFuturesTradingEnvConfig

        return BinanceFuturesTradingEnvConfig(
            symbol="BTCUSDT",
            demo=True,
            intervals=["1m", "5m"],
            window_sizes=[10, 10],
            execute_on="1m",
            leverage=5,
        )

    @pytest.fixture
    def env(self, env_config, mock_observer, mock_trader):
        """Create environment with mocks."""
        from torchtrade.envs.binance.torch_env_futures import BinanceFuturesTorchTradingEnv

        # Patch time.sleep to avoid waiting
        with patch("time.sleep"):
            with patch("torchtrade.envs.binance.torch_env_futures.BinanceFuturesTorchTradingEnv._wait_for_next_timestamp"):
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
        """Test action spec is correctly defined."""
        assert env.action_spec.n == 3  # short, hold, long

    def test_observation_spec(self, env):
        """Test observation spec contains expected keys."""
        obs_spec = env.observation_spec

        assert "account_state" in obs_spec.keys()
        assert "market_data_1m_10" in obs_spec.keys()
        assert "market_data_5m_10" in obs_spec.keys()

    def test_account_state_shape(self, env):
        """Test account state has correct shape (10 elements for futures)."""
        obs_spec = env.observation_spec
        assert obs_spec["account_state"].shape == (10,)

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

        assert td["account_state"].shape == (10,)
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

            action_td = TensorDict({"action": torch.tensor(2)}, batch_size=())  # Long
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

    def test_account_state_with_position(self, env, mock_trader):
        """Test account state when there's an open position."""
        from torchtrade.envs.binance.futures_order_executor import PositionStatus

        # Mock a position
        mock_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.001,
                notional_value=50.0,
                entry_price=50000.0,
                unrealized_pnl=0.5,
                unrealized_pnl_pct=0.01,
                mark_price=50500.0,
                leverage=5,
                margin_type="isolated",
                liquidation_price=45000.0,
            )
        })

        td = env._get_observation()

        account_state = td["account_state"]
        assert account_state[1].item() == pytest.approx(0.001, rel=1e-3)  # position_size
        assert account_state[4].item() == pytest.approx(50500.0, rel=1e-3)  # current_price (mark_price)
        assert account_state[6].item() == pytest.approx(5.0, rel=1e-3)  # leverage

    def test_account_state_no_position(self, env, mock_trader):
        """Test account state when there's no position."""
        mock_trader.get_status = MagicMock(return_value={
            "position_status": None
        })

        td = env._get_observation()

        account_state = td["account_state"]
        assert account_state[1].item() == 0.0  # position_size
        assert account_state[9].item() == 0.0  # holding_time

    def test_done_on_bankruptcy(self, env, mock_trader):
        """Test termination on bankruptcy."""
        env.initial_portfolio_value = 1000.0
        env.config.bankrupt_threshold = 0.1

        # Set balance to below threshold
        mock_trader.get_account_balance = MagicMock(return_value={
            "total_margin_balance": 50.0,  # Below 10% of 1000
        })

        done = env._check_termination(50.0)
        assert done is True

    def test_no_termination_above_threshold(self, env, mock_trader):
        """Test no termination when above bankruptcy threshold."""
        env.initial_portfolio_value = 1000.0
        env.config.bankrupt_threshold = 0.1

        done = env._check_termination(500.0)
        assert done is False

    def test_close_method(self, env, mock_trader):
        """Test environment close method."""
        env.close()
        mock_trader.cancel_open_orders.assert_called()

    def test_reward_calculation_close_position(self, env):
        """Test reward calculation when closing position."""
        trade_info = {"executed": True, "closed_position": True}

        reward = env._calculate_reward(
            old_portfolio_value=1000.0,
            new_portfolio_value=1050.0,
            action=0,  # Close action
            trade_info=trade_info,
        )

        assert reward > 0  # Positive reward for profit

    def test_reward_calculation_invalid_action(self, env):
        """Test reward calculation for invalid action."""
        trade_info = {"executed": False, "closed_position": False}

        reward = env._calculate_reward(
            old_portfolio_value=1000.0,
            new_portfolio_value=1000.0,
            action=1,  # Long action that wasn't executed
            trade_info=trade_info,
        )

        assert reward < 0  # Negative reward for invalid action


class TestBinanceFuturesTradingEnvConfig:
    """Tests for BinanceFuturesTradingEnvConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from torchtrade.envs.binance.torch_env_futures import BinanceFuturesTradingEnvConfig

        config = BinanceFuturesTradingEnvConfig()

        assert config.symbol == "BTCUSDT"
        assert config.demo is True
        assert config.leverage == 1
        assert config.action_levels == [-1.0, 0.0, 1.0]

    def test_custom_config(self):
        """Test custom configuration."""
        from torchtrade.envs.binance.torch_env_futures import BinanceFuturesTradingEnvConfig
        from torchtrade.envs.binance.futures_order_executor import MarginType

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


class TestMultipleSteps:
    """Test multiple environment steps."""

    @pytest.fixture
    def env_with_mocks(self):
        """Create environment for multi-step testing."""
        from torchtrade.envs.binance.torch_env_futures import (
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
            intervals=["1m"],
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
