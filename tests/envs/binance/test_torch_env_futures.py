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
