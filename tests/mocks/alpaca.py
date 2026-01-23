"""
Mock classes for testing Alpaca trading environments.

These mocks simulate the behavior of Alpaca API clients without making real API calls.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import uuid

import numpy as np
import pandas as pd


class MockOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class MockOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class MockOrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass
class MockOrder:
    """Mock Alpaca order object."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = "BTCUSD"
    side: MockOrderSide = MockOrderSide.BUY
    type: MockOrderType = MockOrderType.MARKET
    status: MockOrderStatus = MockOrderStatus.FILLED
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    notional: Optional[float] = None
    qty: Optional[float] = None


@dataclass
class MockPosition:
    """Mock Alpaca position object."""
    symbol: str = "BTCUSD"
    qty: float = 0.001
    market_value: float = 100.0
    avg_entry_price: float = 95000.0
    unrealized_pl: float = 5.0
    unrealized_plpc: float = 0.05
    current_price: float = 100000.0


@dataclass
class MockAccount:
    """Mock Alpaca account object."""
    cash: float = 10000.0
    buying_power: float = 10000.0
    portfolio_value: float = 10000.0


@dataclass
class MockClock:
    """Mock Alpaca clock object."""
    is_open: bool = True
    next_close: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("America/New_York")) + timedelta(hours=2))
    next_open: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("America/New_York")) + timedelta(hours=14))
    timestamp: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("America/New_York")))


class MockTradingClient:
    """
    Mock TradingClient for testing AlpacaOrderClass.

    Simulates order submission, position tracking, and account management.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        current_price: float = 100000.0,
        simulate_failures: bool = False,
    ):
        """
        Initialize the mock trading client.

        Args:
            initial_cash: Starting cash balance
            current_price: Current price of the asset
            simulate_failures: If True, some operations will fail for testing error handling
        """
        self.cash = initial_cash
        self.current_price = current_price
        self.simulate_failures = simulate_failures
        self.positions: Dict[str, MockPosition] = {}
        self.orders: Dict[str, MockOrder] = {}
        self.order_history: List[MockOrder] = []

    def submit_order(self, request: Any) -> MockOrder:
        """Submit a mock order."""
        if self.simulate_failures:
            raise Exception("Simulated API failure")

        symbol = getattr(request, 'symbol', 'BTCUSD')
        side = getattr(request, 'side', None)
        notional = getattr(request, 'notional', None)
        qty = getattr(request, 'qty', None)

        # Calculate quantity from notional if needed
        if notional is not None:
            qty = notional / self.current_price

        # Determine order side - handle various representations
        # Handle enums with .value attribute and string representations
        if hasattr(side, 'value'):
            side_str = str(side.value).lower()
        else:
            side_str = str(side).lower()
        is_buy = 'buy' in side_str

        order = MockOrder(
            symbol=symbol,
            side=MockOrderSide.BUY if is_buy else MockOrderSide.SELL,
            status=MockOrderStatus.FILLED,
            filled_qty=qty or 0.0,
            filled_avg_price=self.current_price,
            notional=notional,
            qty=qty,
        )

        self.orders[order.id] = order
        self.order_history.append(order)

        # Update position and cash
        if order.side == MockOrderSide.BUY:
            self._add_position(symbol, order.filled_qty, order.filled_avg_price)
            self.cash -= (order.filled_qty * order.filled_avg_price)
        else:
            self._reduce_position(symbol, order.filled_qty, order.filled_avg_price)
            self.cash += (order.filled_qty * order.filled_avg_price)

        return order

    def _add_position(self, symbol: str, qty: float, price: float):
        """Add to position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.qty + qty
            pos.avg_entry_price = (pos.avg_entry_price * pos.qty + price * qty) / total_qty
            pos.qty = total_qty
            pos.market_value = pos.qty * self.current_price
        else:
            self.positions[symbol] = MockPosition(
                symbol=symbol,
                qty=qty,
                market_value=qty * self.current_price,
                avg_entry_price=price,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
                current_price=self.current_price,
            )

    def _reduce_position(self, symbol: str, qty: float, price: float):
        """Reduce position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.unrealized_pl = (price - pos.avg_entry_price) * qty
            pos.unrealized_plpc = pos.unrealized_pl / (pos.avg_entry_price * qty)
            pos.qty -= qty
            if pos.qty <= 0:
                del self.positions[symbol]
            else:
                pos.market_value = pos.qty * self.current_price

    def get_order_by_id(self, order_id: str) -> MockOrder:
        """Get order by ID."""
        if order_id in self.orders:
            return self.orders[order_id]
        raise Exception(f"Order not found: {order_id}")

    def get_open_position(self, symbol_or_asset_id: str) -> MockPosition:
        """Get open position for symbol."""
        if symbol_or_asset_id in self.positions:
            return self.positions[symbol_or_asset_id]
        raise Exception(f"No position found for {symbol_or_asset_id}")

    def get_all_positions(self) -> List[MockPosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_orders(self, request: Any) -> List[MockOrder]:
        """Get orders based on request filters."""
        return [o for o in self.orders.values() if o.status == MockOrderStatus.NEW]

    def cancel_orders(self):
        """Cancel all open orders."""
        for order_id, order in list(self.orders.items()):
            if order.status == MockOrderStatus.NEW:
                order.status = MockOrderStatus.CANCELED

    def close_position(self, symbol_or_asset_id: str, close_options: Any = None):
        """Close a position."""
        if self.simulate_failures:
            raise Exception("Simulated close position failure")

        if symbol_or_asset_id in self.positions:
            pos = self.positions[symbol_or_asset_id]
            qty_to_close = pos.qty

            if close_options is not None and hasattr(close_options, 'qty'):
                qty_to_close = float(close_options.qty)

            self.cash += qty_to_close * self.current_price

            if qty_to_close >= pos.qty:
                del self.positions[symbol_or_asset_id]
            else:
                pos.qty -= qty_to_close
                pos.market_value = pos.qty * self.current_price

    def get_account(self) -> MockAccount:
        """Get account information."""
        total_position_value = sum(p.market_value for p in self.positions.values())
        return MockAccount(
            cash=self.cash,
            buying_power=self.cash,
            portfolio_value=self.cash + total_position_value,
        )

    def get_clock(self) -> MockClock:
        """Get market clock."""
        return MockClock()


class MockBarsResult:
    """Mock result from get_crypto_bars."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df


class MockObserver:
    """
    Mock AlpacaObservationClass for testing environments.

    Provides synthetic observation data without needing a real data client.
    """

    def __init__(
        self,
        symbol: str = "BTC/USD",
        num_features: int = 4,
        window_sizes: List[int] = None,
        keys: List[str] = None,
    ):
        self.symbol = symbol
        self.num_features = num_features
        self.window_sizes = window_sizes or [10]
        self._keys = keys or [f"1Minute_{ws}" for ws in self.window_sizes]
        self._call_count = 0

    def get_keys(self) -> List[str]:
        return self._keys

    def get_observations(self, return_base_ohlc: bool = False) -> Dict[str, np.ndarray]:
        self._call_count += 1
        np.random.seed(42 + self._call_count)

        observations = {}
        for key, ws in zip(self._keys, self.window_sizes):
            observations[key] = np.random.randn(ws, self.num_features).astype(np.float32)

        if return_base_ohlc:
            observations["base_features"] = np.random.randn(self.window_sizes[0], 4).astype(np.float32)
            observations["base_timestamps"] = np.array([
                datetime.now(ZoneInfo("America/New_York")) - timedelta(minutes=i)
                for i in range(self.window_sizes[0] - 1, -1, -1)
            ])

        return observations

    def get_features(self) -> Dict[str, List[str]]:
        return {
            "observation_features": ["feature_close", "feature_open", "feature_high", "feature_low"],
            "original_features": ["timestamp", "open", "high", "low", "close", "volume"],
        }


class MockTrader:
    """
    Mock AlpacaOrderClass for testing environments.

    Simulates trading operations without real API calls.
    """

    def __init__(
        self,
        symbol: str = "BTCUSD",
        initial_cash: float = 10000.0,
        current_price: float = 100000.0,
    ):
        self.symbol = symbol
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.current_price = current_price
        self.position_qty = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.last_order_id = None
        self.trade_mode = "notional"
        self.client = self  # Self-reference for client.get_account() calls

    def trade(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        **kwargs
    ) -> bool:
        if side.lower() == "buy":
            # Check if already holding a position (can't buy more in simple mode)
            if self.position_qty > 0:
                return False  # Already holding

            qty = amount / self.current_price
            self.cash -= amount
            self.avg_entry_price = self.current_price
            self.position_qty += qty
            self.position_value = self.position_qty * self.current_price
        else:  # sell
            # Check if no position to sell
            if self.position_qty <= 0:
                return False  # No position to sell

            sell_value = self.position_qty * self.current_price
            self.cash += sell_value
            self.position_qty = 0.0
            self.position_value = 0.0
            self.avg_entry_price = 0.0

        self.last_order_id = str(uuid.uuid4())
        return True

    def get_status(self) -> Dict:
        status = {}
        if self.position_qty > 0:
            unrealized_pl = (self.current_price - self.avg_entry_price) * self.position_qty
            unrealized_plpc = unrealized_pl / (self.avg_entry_price * self.position_qty) if self.avg_entry_price > 0 else 0
            status["position_status"] = PositionStatus(
                qty=self.position_qty,
                market_value=self.position_value,
                avg_entry_price=self.avg_entry_price,
                unrealized_pl=unrealized_pl,
                unrealized_plpc=unrealized_plpc,
                current_price=self.current_price,
            )
        else:
            status["position_status"] = None
        return status

    def get_account(self) -> MockAccount:
        return MockAccount(
            cash=self.cash,
            buying_power=self.cash,
            portfolio_value=self.cash + self.position_value,
        )

    def cancel_open_orders(self) -> bool:
        return True

    def close_position(self, qty: Optional[float] = None) -> bool:
        if qty is not None:
            sell_value = qty * self.current_price
            self.cash += sell_value
            self.position_qty -= qty
            self.position_value = self.position_qty * self.current_price
        else:
            sell_value = self.position_qty * self.current_price
            self.cash += sell_value
            self.position_qty = 0.0
            self.position_value = 0.0
        return True

    def close_all_positions(self) -> Dict[str, bool]:
        self.close_position()
        return {self.symbol: True}

    def get_clock(self) -> MockClock:
        return MockClock()


@dataclass
class PositionStatus:
    """Position status dataclass for MockTrader."""
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float


class MockCryptoHistoricalDataClient:
    """
    Mock CryptoHistoricalDataClient for testing AlpacaObservationClass.

    Returns synthetic OHLCV data without making real API calls.
    """

    def __init__(
        self,
        initial_price: float = 100000.0,
        volatility: float = 0.001,
        num_bars: int = 100,
        trend: float = 0.0,
    ):
        """
        Initialize the mock crypto data client.

        Args:
            initial_price: Starting price for synthetic data
            volatility: Standard deviation of returns
            num_bars: Number of bars to generate
            trend: Trend component (positive = uptrend, negative = downtrend)
        """
        self.initial_price = initial_price
        self.volatility = volatility
        self.num_bars = num_bars
        self.trend = trend
        self._call_count = 0

    def get_crypto_bars(self, request: Any) -> MockBarsResult:
        """Generate synthetic OHLCV bars."""
        self._call_count += 1

        symbol = request.symbol_or_symbols
        if isinstance(symbol, list):
            symbol = symbol[0]

        # Generate synthetic price data
        np.random.seed(42 + self._call_count)  # Different data each call but reproducible
        returns = np.random.normal(self.trend, self.volatility, self.num_bars)
        close_prices = self.initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, self.num_bars)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, self.num_bars)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = self.initial_price

        # Ensure OHLC consistency
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))

        # Generate volume
        volume = np.random.lognormal(10, 1, self.num_bars)

        # Create timestamps based on request timeframe
        now = datetime.now(ZoneInfo("America/New_York"))
        timeframe = request.timeframe
        if hasattr(timeframe, 'amount') and hasattr(timeframe, 'unit'):
            unit = str(timeframe.unit)
            amount = timeframe.amount
            if 'Minute' in unit:
                delta = timedelta(minutes=amount)
            elif 'Hour' in unit:
                delta = timedelta(hours=amount)
            elif 'Day' in unit:
                delta = timedelta(days=amount)
            else:
                delta = timedelta(minutes=1)
        else:
            delta = timedelta(minutes=1)

        timestamps = [now - delta * (self.num_bars - i) for i in range(self.num_bars)]

        # Create DataFrame with proper index
        df = pd.DataFrame({
            'symbol': [symbol] * self.num_bars,
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'trade_count': np.random.randint(100, 1000, self.num_bars),
            'vwap': close_prices * (1 + np.random.normal(0, 0.001, self.num_bars)),
        })

        df = df.set_index(['symbol', 'timestamp'])

        return MockBarsResult(df)
