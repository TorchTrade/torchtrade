"""Tests for BinanceFuturesTorchTradingEnv.

Inherits common tests from BaseEnvTests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict
from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
from tests.envs.base_exchange_tests import BaseEnvTests


@pytest.fixture
def mock_observer():
    """Create a mock observer."""
    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1m_10"])
    observer.get_observations = MagicMock(return_value={
        "1m_10": np.random.randn(10, 4).astype(np.float32),
    })
    observer.intervals = ["1m"]
    observer.window_sizes = [10]
    return observer


@pytest.fixture
def mock_trader():
    """Create a mock trader."""
    trader = MagicMock()
    trader.cancel_open_orders = MagicMock(return_value=True)
    trader.close_position = MagicMock(return_value=True)
    trader.get_account_balance = MagicMock(return_value={
        "total_wallet_balance": 1000.0,
        "available_balance": 900.0,
        "total_margin_balance": 1000.0,
    })
    trader.get_mark_price = MagicMock(return_value=50000.0)
    trader.get_status = MagicMock(return_value={"position_status": None})
    trader.trade = MagicMock(return_value=True)
    return trader


class TestBinanceFuturesTorchTradingEnv(BaseEnvTests):
    """Tests for BinanceFuturesTorchTradingEnv - inherits common tests from base."""

    def create_env(self, config, observer, trader):
        """Create a BinanceFuturesTorchTradingEnv instance."""
        with patch("time.sleep"):
            with patch.object(BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BinanceFuturesTorchTradingEnv(
                    config=config,
                    observer=observer,
                    trader=trader,
                )
                return env

    def create_config(self, **kwargs):
        """Create a BinanceFuturesTradingEnvConfig instance."""
        return BinanceFuturesTradingEnvConfig(
            symbol=kwargs.get('symbol', 'BTCUSDT'),
            demo=kwargs.get('demo', True),
            time_frames=kwargs.get('time_frames', ["1m"]),
            window_sizes=kwargs.get('window_sizes', [10]),
            leverage=kwargs.get('leverage', 5),
            done_on_bankruptcy=kwargs.get('done_on_bankruptcy', False),
            bankrupt_threshold=kwargs.get('bankrupt_threshold', 0.1),
        )

    # Binance-specific tests

    def test_account_state_has_10_elements(self, mock_observer, mock_trader):
        """Test that Binance futures account state has 10 elements."""
        config = self.create_config()
        env = self.create_env(config, mock_observer, mock_trader)

        td = env.reset()
        assert td["account_state"].shape == (10,)

    def test_action_spec_fractional(self, mock_observer, mock_trader):
        """Test action spec uses fractional action levels."""
        config = self.create_config()
        env = self.create_env(config, mock_observer, mock_trader)

        assert env.action_spec.n >= 3
        action_levels = env.action_levels
        assert all(-1 <= level <= 1 for level in action_levels)
        assert action_levels == sorted(action_levels)

    def test_leverage_in_account_state(self, mock_observer, mock_trader):
        """Test that leverage is included in account state."""
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        mock_trader.get_status = MagicMock(return_value={
            "position_status": PositionStatus(
                qty=0.001, notional_value=50.0, entry_price=50000.0,
                unrealized_pnl=0.5, unrealized_pnl_pct=0.01, mark_price=50500.0,
                leverage=5, margin_type="isolated", liquidation_price=45000.0,
            )
        })

        config = self.create_config(leverage=5)
        env = self.create_env(config, mock_observer, mock_trader)

        td = env._get_observation()
        account_state = td["account_state"]
        assert account_state[6].item() == pytest.approx(5.0, rel=1e-3)
