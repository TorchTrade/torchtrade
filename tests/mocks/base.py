"""
Base documentation for TorchTrade exchange test mocks.

This directory contains centralized mock implementations for testing exchange integrations:
- alpaca.py: Mock Alpaca trading client (dataclass-based mocks)
- binance.py: Mock Binance futures client (MagicMock-based)
- bitget.py: Mock Bitget CCXT client (MagicMock-based)

Mock Patterns:
- Alpaca: Uses custom dataclasses (MockOrder, MockPosition, etc.) with a MockTradingClient
- Binance: Uses pytest fixtures returning configured MagicMock objects for Binance futures API
- Bitget: Uses pytest fixtures returning configured MagicMock objects for CCXT API

All mocks are available through fixtures defined in tests/mocks/conftest.py.
"""
