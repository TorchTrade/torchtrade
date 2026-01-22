"""Tests for BitgetFuturesSLTPTorchTradingEnv with CCXT.

Inherits common tests from BaseSLTPTests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torchtrade.envs.live.bitget.env_sltp import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig,
)
from tests.envs.base_exchange_tests import BaseSLTPTests
from tests.mocks.bitget import mock_ccxt_client


@pytest.fixture
def mock_observer():
    """Create a mock observer."""
    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1m_10"])

    def mock_observations(return_base_ohlc=False):
        obs = {"1m_10": np.random.randn(10, 4).astype(np.float32)}
        if return_base_ohlc:
            obs["base_features"] = np.array([[50000, 50100, 49900, 50050]] * 10, dtype=np.float32)
        return obs

    observer.get_observations = MagicMock(side_effect=mock_observations)
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


class TestBitgetFuturesSLTPTorchTradingEnv(BaseSLTPTests):
    """Tests for BitgetFuturesSLTPTorchTradingEnv - inherits common tests from base."""

    def create_sltp_env(self, config, observer, trader):
        """Create a BitgetFuturesSLTPTorchTradingEnv instance."""
        with patch("time.sleep"):
            with patch.object(BitgetFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BitgetFuturesSLTPTorchTradingEnv(
                    config=config,
                    observer=observer,
                    trader=trader,
                )
                return env

    def create_sltp_config(self, **kwargs):
        """Create a BitgetFuturesSLTPTradingEnvConfig instance."""
        return BitgetFuturesSLTPTradingEnvConfig(
            symbol=kwargs.get('symbol', 'BTCUSDT:USDT'),
            time_frames=kwargs.get('time_frames', ["1m"]),
            window_sizes=kwargs.get('window_sizes', [10]),
            leverage=kwargs.get('leverage', 5),
            stoploss_levels=kwargs.get('stoploss_levels', (-0.02, -0.05)),
            takeprofit_levels=kwargs.get('takeprofit_levels', (0.03, 0.06)),
            include_short_positions=kwargs.get('include_short_positions', True),
            done_on_bankruptcy=kwargs.get('done_on_bankruptcy', False),
            bankrupt_threshold=kwargs.get('bankrupt_threshold', 0.1),
        )

    # Bitget-specific SL/TP tests

    def test_account_state_has_10_elements(self, mock_observer, mock_trader):
        """Test that Bitget account state has 10 elements."""
        config = self.create_sltp_config()
        env = self.create_sltp_env(config, mock_observer, mock_trader)

        td = env.reset()
        assert td["account_state"].shape == (10,)
