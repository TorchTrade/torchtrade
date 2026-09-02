"""Shared test fixtures for OKX tests."""

from tests.envs.base_exchange_tests import a_mock_futures_trader, a_mock_observer
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
    # Echoes `lever` AND `posSide`, as OKX documents: an entry without lever confirms
    # nothing (#277), and one echoing the wrong side confirmed the wrong leg (#363).
    client.set_leverage = MagicMock(
        side_effect=lambda instId=None, lever=None, posSide=None, **k: {
            "code": "0", "msg": "",
            "data": [{"lever": lever, **({} if posSide is None else {"posSide": posSide})}]})

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
    return a_mock_observer(["1Minute_10"], base=(50000, 50100, 49900, 50050))


@pytest.fixture
def mock_env_trader():
    return a_mock_futures_trader()
