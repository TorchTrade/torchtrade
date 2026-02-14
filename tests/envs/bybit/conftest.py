"""Shared test fixtures for Bybit tests."""

import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_pybit_client():
    """Create a mock pybit HTTP client with common methods."""
    client = MagicMock()

    # Mock account configuration methods
    client.set_leverage = MagicMock(return_value={"retCode": 0})
    client.switch_margin_mode = MagicMock(return_value={"retCode": 0})
    client.switch_position_mode = MagicMock(return_value={"retCode": 0})

    # Mock order placement
    client.place_order = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "orderId": "12345",
            "orderLinkId": "",
        },
    })

    # Mock position information
    client.get_positions = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "list": [{
                "symbol": "BTCUSDT",
                "size": "0.001",
                "side": "Buy",
                "avgPrice": "50000.0",
                "markPrice": "50100.0",
                "unrealisedPnl": "0.1",
                "leverage": "10",
                "tradeMode": "1",
                "liqPrice": "45000.0",
                "positionValue": "50.1",
            }]
        },
    })

    # Mock account balance
    client.get_wallet_balance = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "list": [{
                "totalEquity": "1000.0",
                "totalAvailableBalance": "900.0",
                "totalPerpUPL": "0.1",
                "totalMarginBalance": "1000.1",
            }]
        },
    })

    # Mock market data
    client.get_tickers = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "list": [{
                "symbol": "BTCUSDT",
                "markPrice": "50100.0",
                "lastPrice": "50100.0",
            }]
        },
    })

    # Mock kline data
    def mock_get_kline(category, symbol, interval, limit=200):
        """Generate mock candle data (reverse chronological order like Bybit)."""
        candles = []
        base_time = 1700000000000
        for i in range(limit - 1, -1, -1):  # Reverse order
            candles.append([
                str(base_time + i * 60000),  # timestamp (string)
                "50000.0",  # open
                "50100.0",  # high
                "49900.0",  # low
                "50050.0",  # close
                "100.0",    # volume
                "5005000.0",  # turnover
            ])
        return {
            "retCode": 0,
            "result": {
                "list": candles,
            },
        }

    client.get_kline = MagicMock(side_effect=mock_get_kline)

    # Mock order management
    client.get_open_orders = MagicMock(return_value={
        "retCode": 0,
        "result": {"list": []},
    })
    client.cancel_all_orders = MagicMock(return_value={"retCode": 0})

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
    return trader


@pytest.fixture
def mock_empty_position(mock_pybit_client):
    """Configure mock client with no open position."""
    mock_pybit_client.get_positions = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "list": [{
                "symbol": "BTCUSDT",
                "size": "0",
                "side": "Buy",
            }]
        },
    })
    return mock_pybit_client


@pytest.fixture
def mock_short_position(mock_pybit_client):
    """Configure mock client with a short position."""
    mock_pybit_client.get_positions = MagicMock(return_value={
        "retCode": 0,
        "result": {
            "list": [{
                "symbol": "BTCUSDT",
                "size": "0.001",
                "side": "Sell",
                "avgPrice": "50000.0",
                "markPrice": "49900.0",
                "unrealisedPnl": "0.1",
                "leverage": "10",
                "tradeMode": "1",
                "liqPrice": "55000.0",
                "positionValue": "49.9",
            }]
        },
    })
    return mock_pybit_client
