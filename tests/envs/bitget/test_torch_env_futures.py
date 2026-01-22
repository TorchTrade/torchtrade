"""Tests for BitgetFuturesTorchTradingEnv with CCXT.

Inherits common tests from BaseEnvTests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torchtrade.envs.bitget.torch_env_futures import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig
from tests.envs.base_exchange_tests import BaseEnvTests
from tests.mocks.bitget import mock_ccxt_client


@pytest.fixture
def mock_observer():
    """Create a mock observer."""
    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1m_10"])
    observer.get_observations = MagicMock(return_value={
        "1m_10": np.random.randn(10, 4).astype(np.float32),
    })
    return observer


@pytest.fixture
def mock_trader():
    """Create a mock trader."""
    trader = MagicMock()
    trader.cancel_open_orders = MagicMock(return_value=True)
    trader.close_position = MagicMock(return_value=True)
    trader.get_account_balance = MagicMock(return_value={"total_margin_balance": 1000.0})
    trader.get_mark_price = MagicMock(return_value=50000.0)
    trader.get_status = MagicMock(return_value={"position_status": None})
    trader.trade = MagicMock(return_value=True)
    return trader


class TestBitgetFuturesTorchTradingEnv(BaseEnvTests):
    """Tests for BitgetFuturesTorchTradingEnv - inherits common tests from base."""

    def create_env(self, config, observer, trader):
        """Create a BitgetFuturesTorchTradingEnv instance."""
        with patch("time.sleep"):
            with patch.object(BitgetFuturesTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesTorchTradingEnv(
                    config=config,
                    observer=observer,
                    trader=trader,
                )
                return env

    def create_config(self, **kwargs):
        """Create a BitgetFuturesTradingEnvConfig instance."""
        return BitgetFuturesTradingEnvConfig(
            symbol=kwargs.get('symbol', 'BTCUSDT:USDT'),
            time_frames=kwargs.get('time_frames', ["1m"]),
            window_sizes=kwargs.get('window_sizes', [10]),
            leverage=kwargs.get('leverage', 5),
            done_on_bankruptcy=kwargs.get('done_on_bankruptcy', False),
            bankrupt_threshold=kwargs.get('bankrupt_threshold', 0.1),
        )

    # Bitget-specific tests

    def test_account_state_has_10_elements(self, mock_observer, mock_trader):
        """Test that Bitget futures account state has 10 elements."""
        config = self.create_config()
        env = self.create_env(config, mock_observer, mock_trader)

        td = env.reset()
        assert td["account_state"].shape == (10,)
