"""Tests for BinanceFuturesSLTPTorchTradingEnv.

Inherits common tests from BaseSLTPTests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from tensordict import TensorDict
from torchtrade.envs.live.binance.env_sltp import (
    BinanceFuturesSLTPTorchTradingEnv,
    BinanceFuturesSLTPTradingEnvConfig,
)
from tests.envs.base_exchange_tests import BaseSLTPTests


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


class TestBinanceFuturesSLTPTorchTradingEnv(BaseSLTPTests):
    """Tests for BinanceFuturesSLTPTorchTradingEnv - inherits common tests from base."""

    def create_sltp_env(self, config, observer, trader):
        """Create a BinanceFuturesSLTPTorchTradingEnv instance."""
        with patch("time.sleep"):
            with patch.object(BinanceFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
                env = BinanceFuturesSLTPTorchTradingEnv(
                    config=config,
                    observer=observer,
                    trader=trader,
                )
                return env

    def create_sltp_config(self, **kwargs):
        """Create a BinanceFuturesSLTPTradingEnvConfig instance."""
        return BinanceFuturesSLTPTradingEnvConfig(
            symbol=kwargs.get('symbol', 'BTCUSDT'),
            demo=kwargs.get('demo', True),
            time_frames=kwargs.get('time_frames', ["1m"]),
            window_sizes=kwargs.get('window_sizes', [10]),
            leverage=kwargs.get('leverage', 5),
            stoploss_levels=kwargs.get('stoploss_levels', (-0.02, -0.05)),
            takeprofit_levels=kwargs.get('takeprofit_levels', (0.03, 0.06)),
            include_short_positions=kwargs.get('include_short_positions', True),
            done_on_bankruptcy=kwargs.get('done_on_bankruptcy', False),
            bankrupt_threshold=kwargs.get('bankrupt_threshold', 0.1),
        )

    # Binance-specific SL/TP tests

    def test_action_map_with_shorts(self, mock_observer, mock_trader):
        """Test action map includes both longs and shorts."""
        config = self.create_sltp_config(
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
            include_short_positions=True,
        )

        env = self.create_sltp_env(config, mock_observer, mock_trader)

        # 1 HOLD + 4 LONG (2x2) + 4 SHORT (2x2) = 9
        assert len(env.action_map) == 9
        assert env.action_map[0] == (None, None, None)

    def test_action_map_long_only(self, mock_observer, mock_trader):
        """Test action map with shorts disabled."""
        config = self.create_sltp_config(
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
            include_short_positions=False,
        )

        env = self.create_sltp_env(config, mock_observer, mock_trader)

        # 1 HOLD + 4 LONG (2x2) = 5
        assert len(env.action_map) == 5

    def test_short_actions_have_flipped_sltp(self, mock_observer, mock_trader):
        """Test that short actions have flipped SL/TP signs."""
        config = self.create_sltp_config(include_short_positions=True)
        env = self.create_sltp_env(config, mock_observer, mock_trader)

        # Check a short action (actions 5-8)
        for i in range(5, min(9, len(env.action_map))):
            side, sl, tp = env.action_map[i]
            if side == "short":
                assert sl > 0  # SL above entry for shorts
                assert tp < 0  # TP below entry for shorts

    def test_account_state_has_10_elements(self, mock_observer, mock_trader):
        """Test that Binance account state has 10 elements."""
        config = self.create_sltp_config()
        env = self.create_sltp_env(config, mock_observer, mock_trader)

        td = env.reset()
        assert td["account_state"].shape == (10,)
