"""Shared test fixtures for OKX tests."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_okx_trade_client():
    """Create a mock OKX Trade client."""
    client = MagicMock()

    # Mock order placement
    client.place_order = MagicMock(return_value={
        "code": "0",
        "msg": "",
        "data": [{
            "ordId": "12345",
            "clOrdId": "",
            "sCode": "0",
            "sMsg": "",
        }],
    })

    return client


@pytest.fixture
def mock_okx_account_client():
    """Create a mock OKX Account client."""
    client = MagicMock()

    # Mock account configuration
    client.set_position_mode = MagicMock(return_value={"code": "0", "msg": ""})
    client.set_leverage = MagicMock(return_value={"code": "0", "msg": "", "data": [{}]})

    # Mock position information
    client.get_positions = MagicMock(return_value={
        "code": "0",
        "msg": "",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "pos": "0.001",
            "posSide": "net",
            "avgPx": "50000.0",
            "markPx": "50100.0",
            "upl": "0.1",
            "lever": "10",
            "mgnMode": "isolated",
            "liqPx": "45000.0",
            "notionalUsd": "50.1",
        }],
    })

    # Mock account balance
    client.get_account_balance = MagicMock(return_value={
        "code": "0",
        "msg": "",
        "data": [{
            "totalEq": "1000.0",
            "upl": "0.1",
            "details": [{
                "ccy": "USDT",
                "availBal": "900.0",
            }],
        }],
    })

    return client


@pytest.fixture
def mock_okx_public_client():
    """Create a mock OKX PublicData client."""
    client = MagicMock()

    # Mock mark price
    client.get_mark_price = MagicMock(return_value={
        "code": "0",
        "msg": "",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "instType": "SWAP",
            "markPx": "50100.0",
        }],
    })

    # Mock instrument info (lot size + price precision)
    client.get_instruments = MagicMock(return_value={
        "code": "0",
        "msg": "",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "tickSz": "0.01",
            "minSz": "0.001",
            "lotSz": "0.001",
        }],
    })

    return client


@pytest.fixture
def mock_okx_market_client():
    """Create a mock OKX MarketData client for observation tests."""
    client = MagicMock()

    def mock_get_candlesticks(instId, bar, limit="200"):
        """Generate mock candle data (reverse chronological order like OKX)."""
        n = int(limit)
        candles = []
        base_time = 1700000000000
        for i in range(n - 1, -1, -1):  # Reverse order
            candles.append([
                str(base_time + i * 60000),  # timestamp (string)
                "50000.0",  # open
                "50100.0",  # high
                "49900.0",  # low
                "50050.0",  # close
                "100.0",    # volume
                "5005000.0",  # vol_ccy
                "5005000.0",  # vol_ccy_quote
                "1",        # confirm
            ])
        return {
            "code": "0",
            "msg": "",
            "data": candles,
        }

    client.get_candlesticks = MagicMock(side_effect=mock_get_candlesticks)

    return client


@pytest.fixture
def mock_env_observer():
    """Create a mock observer for env tests (single timeframe)."""
    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1Minute_10"])

    def mock_observations(return_base_ohlc=False):
        obs = {"1Minute_10": np.random.randn(10, 4).astype(np.float32)}
        if return_base_ohlc:
            obs["base_features"] = np.array(
                [[50000, 50100, 49900, 50050]] * 10, dtype=np.float32
            )
        return obs

    observer.get_observations = MagicMock(side_effect=mock_observations)
    observer.get_features = MagicMock(return_value={
        "observation_features": ["feature_close", "feature_open", "feature_high", "feature_low"],
        "original_features": ["open", "high", "low", "close", "volume"],
    })
    return observer


@pytest.fixture
def mock_env_trader():
    """Create a mock trader for env tests."""
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
    trader.get_lot_size = MagicMock(return_value={"min_qty": 0.001, "qty_step": 0.001})
    return trader


@pytest.fixture
def replay_df():
    """Create realistic OHLCV test data for replay integration tests."""
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
