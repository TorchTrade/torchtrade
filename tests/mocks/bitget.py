"""Shared test fixtures for Bitget tests."""

import pytest
from unittest.mock import MagicMock


__all__ = ['mock_ccxt_client', 'mock_empty_position', 'mock_short_position']


@pytest.fixture
def mock_ccxt_client():
    """Create a mock CCXT Bitget client with common methods."""
    client = MagicMock()

    # Mock account configuration methods
    client.set_leverage = MagicMock(return_value={"leverage": 10})
    client.set_margin_mode = MagicMock(return_value={"marginMode": "isolated"})
    client.set_position_mode = MagicMock(return_value={"posMode": "one_way_mode"})

    # Mock order placement
    client.create_order = MagicMock(return_value={
        "id": "12345",
        "symbol": "BTC/USDT:USDT",
        "status": "closed",
        "side": "buy",
        "type": "market",
        "filled": 0.001,
        "average": 50000.0,
    })

    # Mock bracket orders
    client.create_order_with_take_profit_and_stop_loss = MagicMock(return_value={
        "id": "12345",
        "symbol": "BTC/USDT:USDT",
        "status": "closed",
    })

    # Mock stop market order
    client.create_stop_market_order = MagicMock(return_value={
        "id": "12346",
        "symbol": "BTC/USDT:USDT",
        "type": "stop_market",
    })

    # Mock position information
    client.fetch_positions = MagicMock(return_value=[{
        "symbol": "BTC/USDT:USDT",
        "contracts": 0.001,
        "side": "long",
        "entryPrice": 50000.0,
        "markPrice": 50100.0,
        "unrealizedPnl": 0.1,
        "leverage": 10,
        "marginMode": "isolated",
        "liquidationPrice": 45000.0,
        "notional": 50.1,
    }])

    # Mock account balance
    client.fetch_balance = MagicMock(return_value={
        "USDT": {
            "total": 1000.0,
            "free": 900.0,
            "used": 100.0,
        },
        "info": {
            "totalEquity": "1000.0",
            "available": "900.0",
            "totalUnrealizedProfit": "0.1",
            "totalMarginBalance": "1000.1",
        },
    })

    # Mock market data
    client.fetch_ticker = MagicMock(return_value={
        "symbol": "BTC/USDT:USDT",
        "info": {"markPrice": "50100.0"},
        "last": 50100.0,
    })

    # Mock OHLCV data
    def mock_fetch_ohlcv(symbol, timeframe, limit=200):
        """Generate mock candle data."""
        candles = []
        base_time = 1700000000000
        for i in range(limit):
            candles.append([
                base_time + i * 60000,  # timestamp
                50000.0,  # open
                50100.0,  # high
                49900.0,  # low
                50050.0,  # close
                100.0,    # volume
            ])
        return candles

    client.fetch_ohlcv = MagicMock(side_effect=mock_fetch_ohlcv)

    # Mock order management
    client.fetch_open_orders = MagicMock(return_value=[])
    client.cancel_order = MagicMock(return_value={"id": "12345", "status": "canceled"})
    client.fetch_order = MagicMock(return_value={
        "id": "12345",
        "status": "closed",
        "side": "buy",
        "type": "market",
        "filled": 0.001,
        "average": 50000.0,
    })

    return client


@pytest.fixture
def mock_empty_position(mock_ccxt_client):
    """Configure mock client with no open position."""
    mock_ccxt_client.fetch_positions = MagicMock(return_value=[{
        "symbol": "BTC/USDT:USDT",
        "contracts": 0,
        "side": "long",
    }])
    return mock_ccxt_client


@pytest.fixture
def mock_short_position(mock_ccxt_client):
    """Configure mock client with a short position."""
    mock_ccxt_client.fetch_positions = MagicMock(return_value=[{
        "symbol": "BTC/USDT:USDT",
        "contracts": 0.001,
        "side": "short",
        "entryPrice": 50000.0,
        "markPrice": 49900.0,
        "unrealizedPnl": 0.1,
        "leverage": 10,
        "marginMode": "isolated",
        "liquidationPrice": 55000.0,
        "notional": 49.9,
    }])
    return mock_ccxt_client
