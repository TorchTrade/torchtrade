"""Shared mock fixtures for exchange testing."""

import pytest
from tests.mocks import binance, bitget

# Re-export Binance fixtures
@pytest.fixture
def mock_binance_client():
    """Create mock Binance futures client."""
    return binance.mock_binance_client()

# Re-export Bitget fixtures
@pytest.fixture
def mock_ccxt_client():
    """Create mock CCXT Bitget client."""
    return bitget.mock_ccxt_client()

@pytest.fixture
def mock_empty_position(mock_ccxt_client):
    """Configure mock client with no open position."""
    return bitget.mock_empty_position(mock_ccxt_client)

@pytest.fixture
def mock_short_position(mock_ccxt_client):
    """Configure mock client with a short position."""
    return bitget.mock_short_position(mock_ccxt_client)
