import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
)
from dotenv import load_dotenv


class TradeMode(Enum):
    NOTIONAL = "notional"
    QUANTITY = "quantity"


@dataclass
class OrderStatus:
    is_open: bool
    order_id: Optional[str]
    filled_qty: Optional[float]
    filled_avg_price: Optional[float]
    status: str
    side: str
    order_type: str


@dataclass
class PositionStatus:
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float


class AlpacaOrderClass:
    def __init__(
        self,
        symbol: str,
        trade_mode: TradeMode,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ):
        """
        Initialize the AlpacaOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTC/USD")
            trade_mode: TradeMode.NOTIONAL for dollar-based orders or TradeMode.QUANTITY for unit-based orders
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Whether to use paper trading (default: True)
        """
        self.symbol = symbol
        # TODO how to get asset_id?
        self.asset_id = "64bbff51-59d6-4b3c-9351-13ad85e3c752"
        self.trade_mode = trade_mode
        self.client = TradingClient(api_key, secret_key=api_secret, paper=paper)
        self.last_order_id = None

    def trade(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "gtc",
    ) -> bool:
        """
        Execute a trade with the specified parameters.

        Args:
            side: "buy" or "sell"
            amount: Amount to trade (in USD if trade_mode is NOTIONAL, in units if QUANTITY)
            order_type: "market", "limit", or "stop_limit"
            limit_price: Required for limit and stop_limit orders
            stop_price: Required for stop_limit orders
            time_in_force: Time in force for the order (default: "gtc")

        Returns:
            bool: True if order was submitted successfully, False otherwise
        """
        try:
            # Convert parameters to Alpaca enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            time_in_force = (
                TimeInForce.GTC if time_in_force.lower() == "gtc" else TimeInForce.IOC
            )

            # Prepare base order parameters
            order_params = {
                "symbol": self.symbol,
                "side": order_side,
                "time_in_force": time_in_force,
            }

            # Add amount based on trade mode
            if self.trade_mode == TradeMode.NOTIONAL:
                order_params["notional"] = amount
            else:
                order_params["qty"] = amount

            # Create appropriate order request based on order type
            if order_type.lower() == "market":
                order_params["type"] = OrderType.MARKET
                request = MarketOrderRequest(**order_params)

            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                order_params["type"] = OrderType.LIMIT
                order_params["limit_price"] = limit_price
                request = LimitOrderRequest(**order_params)

            elif order_type.lower() == "stop_limit":
                if limit_price is None or stop_price is None:
                    raise ValueError(
                        "limit_price and stop_price are required for stop_limit orders"
                    )
                order_params["limit_price"] = limit_price
                order_params["stop_price"] = stop_price
                request = StopLimitOrderRequest(**order_params)

            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Submit the order
            response = self.client.submit_order(request)
            self.last_order_id = response.id
            return True

        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Union[OrderStatus, PositionStatus]]:
        """
        Get current order and position status.

        Returns:
            Dictionary containing order_status and position_status
        """
        try:
            status = {}

            # Get order status if we have a last order
            if self.last_order_id:
                order = self.client.get_order_by_id(self.last_order_id)
                status["order_status"] = OrderStatus(
                    is_open=order.status not in ["filled", "canceled", "expired"],
                    order_id=order.id,
                    filled_qty=order.filled_qty,
                    filled_avg_price=order.filled_avg_price,
                    status=order.status,
                    side=order.side.value,
                    order_type=order.type.value,
                )

            # Get position status
            try:
                position = self.client.get_open_position(symbol_or_asset_id=self.asset_id)
                status["position_status"] = PositionStatus(
                    qty=float(position.qty),
                    market_value=float(position.market_value),
                    avg_entry_price=float(position.avg_entry_price),
                    unrealized_pl=float(position.unrealized_pl),
                    unrealized_plpc=float(position.unrealized_plpc),
                    current_price=float(position.current_price),
                )
            except Exception:
                status["position_status"] = None  # No open position

            return status

        except Exception as e:
            print(f"Error getting status: {str(e)}")
            return {}

    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders for the symbol.

        Returns:
            List of open orders
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[self.symbol]
            )
            return self.client.get_orders(request)
        except Exception as e:
            print(f"Error getting open orders: {str(e)}")
            return []
        
    def cancel_open_orders(self) -> bool:
        """
        Cancel all open orders for the symbol.

        Returns:
            bool: True if orders were cancelled successfully, False otherwise
        """
        try:
            open_orders = self.get_open_orders()
            if not open_orders:
                print("No open orders to cancel")
                return True
            else:
                self.client.cancel_orders()
                print("Open orders cancelled")
                return True
        except Exception as e:
            print(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self, qty: Optional[float] = None) -> bool:
        """
        Close the current position, either partially or fully.

        Args:
            qty: Optional quantity to close. If None, closes entire position.

        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        try:
            if qty is not None:
                # Close specific quantity
                self.client.close_position(
                    symbol_or_asset_id=self.symbol,
                    close_options=ClosePositionRequest(qty=str(qty)),
                )
            else:
                # Close entire position
                self.client.close_position(symbol_or_asset_id=self.asset_id)
            return True
        except Exception as e:
            print(f"Error closing position: {str(e)}")
            return False

    def close_all_positions(self) -> Dict[str, bool]:
        """
        Close all open positions for the account.

        Returns:
            Dict with position symbols as keys and success status as values
        """
        try:
            results = {}
            positions = self.client.get_all_positions()

            for position in positions:
                try:
                    self.client.close_position(symbol_or_asset_id=position.symbol)
                    results[position.symbol] = True
                except Exception as e:
                    print(f"Error closing position for {position.symbol}: {str(e)}")
                    results[position.symbol] = False

            return results
        except Exception as e:
            print(f"Error getting positions: {str(e)}")
            return {}


# Example usage:
if __name__ == "__main__":
    # Load the environment variables from the .env file
    load_dotenv()
    key = os.getenv("API_KEY")
    # Initialize the order class
    order_manager = AlpacaOrderClass(
        symbol="BTC/USD",
        trade_mode=TradeMode.NOTIONAL,
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY"),
        paper=True,
    )

    # Example market buy order
    success = order_manager.trade(
        side="buy", amount=100, order_type="market"  # $100 worth of BTC
    )
    print(f"Market order submitted successfully: {success}")

    # Get status
    status = order_manager.get_status()
    print("\nCurrent Status:")
    if "order_status" in status:
        print(f"Order Status: {status['order_status']}")
    if "position_status" in status:
        print(f"Position Status: {status['position_status']}")

    # Example limit order
    success = order_manager.trade(
        side="buy", amount=100, order_type="limit", limit_price=50000
    )
    print(f"\nLimit order submitted successfully: {success}")

    # Get open orders
    open_orders = order_manager.get_open_orders()
    print(f"\nOpen Orders: {open_orders}")

    # Cancel open orders
    success = order_manager.cancel_open_orders()
    print(f"Open orders cancelled: {success}")

    # Example: Close entire position for the symbol
    success = order_manager.close_all_positions()
    print(f"Full position close successful: {success}")

    # Final status
    status = order_manager.get_status()
    print("\nFinal Status:")
    if "order_status" in status:
        print(f"Order Status: {status['order_status']}")
    if "position_status" in status:
        print(f"Position Status: {status['position_status']}")
    
