"""Mock Binance futures client for testing."""

from unittest.mock import MagicMock


__all__ = ['mock_binance_client']


def mock_binance_client():
    """Create a mock Binance futures client."""
    client = MagicMock()

    # Mock futures methods
    client.futures_change_leverage = MagicMock(return_value={"leverage": 10})
    client.futures_change_margin_type = MagicMock(return_value={})

    client.futures_create_order = MagicMock(return_value={
        "orderId": 12345,
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "side": "BUY",
        "type": "MARKET",
        "executedQty": "0.001",
        "avgPrice": "50000.0",
    })

    client.futures_get_order = MagicMock(return_value={
        "orderId": 12345,
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "side": "BUY",
        "type": "MARKET",
        "executedQty": "0.001",
        "avgPrice": "50000.0",
    })

    client.futures_position_information = MagicMock(return_value=[{
        "symbol": "BTCUSDT",
        "positionAmt": "0.001",
        "entryPrice": "50000.0",
        "markPrice": "50100.0",
        "unRealizedProfit": "0.1",
        "notional": "50.1",
        "leverage": "10",
        "marginType": "isolated",
        "liquidationPrice": "45000.0",
    }])

    client.futures_account = MagicMock(return_value={
        "totalWalletBalance": "1000.0",
        "availableBalance": "900.0",
        "totalUnrealizedProfit": "0.1",
        "totalMarginBalance": "1000.1",
    })

    client.futures_mark_price = MagicMock(return_value={
        "markPrice": "50100.0",
    })

    client.futures_get_open_orders = MagicMock(return_value=[])
    client.futures_cancel_all_open_orders = MagicMock(return_value={})

    return client
