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
    TakeProfitRequest,
    StopLossRequest,
)
from dotenv import load_dotenv


# Common Time in Force Options:
# GTC (Good Till Canceled) – The order remains open until it is executed or manually canceled by the trader. It does not expire at the end of the trading day.
# DAY (Day Order) – The order is only valid for the current trading day and expires if not executed by the market close.
# FOK (Fill or Kill) – The order must be fully executed immediately, or it is canceled.
# IOC (Immediate or Cancel) – The order must be executed immediately (fully or partially); any remaining portion is canceled.
# GTD (Good Till Date/Time) – The order stays active until a specified date and time, unless it is executed or canceled.
# AON (All or None) – The order must be fully executed at once or not at all.
# For Alpaca:
"""
Represents the various time in force options for an Order.

The Time-In-Force values supported by Alpaca vary based on the order's security type. Here is a breakdown of the supported TIFs for each specific security type:
- Equity trading: day, gtc, opg, cls, ioc, fok.
- Options trading: day.
- Crypto trading: gtc, ioc.
Below are the descriptions of each TIF:
- day: A day order is eligible for execution only on the day it is live. By default, the order is only valid during Regular Trading Hours (9:30am - 4:00pm ET). If unfilled after the closing auction, it is automatically canceled. If submitted after the close, it is queued and submitted the following trading day. However, if marked as eligible for extended hours, the order can also execute during supported extended hours.
- gtc: The order is good until canceled. Non-marketable GTC limit orders are subject to price adjustments to offset corporate actions affecting the issue. We do not currently support Do Not Reduce(DNR) orders to opt out of such price adjustments.
- opg: Use this TIF with a market/limit order type to submit “market on open” (MOO) and “limit on open” (LOO) orders. This order is eligible to execute only in the market opening auction. Any unfilled orders after the open will be cancelled. OPG orders submitted after 9:28am but before 7:00pm ET will be rejected. OPG orders submitted after 7:00pm will be queued and routed to the following day’s opening auction. On open/on close orders are routed to the primary exchange. Such orders do not necessarily execute exactly at 9:30am / 4:00pm ET but execute per the exchange’s auction rules.
- cls: Use this TIF with a market/limit order type to submit “market on close” (MOC) and “limit on close” (LOC) orders. This order is eligible to execute only in the market closing auction. Any unfilled orders after the close will be cancelled. CLS orders submitted after 3:50pm but before 7:00pm ET will be rejected. CLS orders submitted after 7:00pm will be queued and routed to the following day’s closing auction. Only available with API v2.
- ioc: An Immediate Or Cancel (IOC) order requires all or part of the order to be executed immediately. Any unfilled portion of the order is canceled. Only available with API v2. Most market makers who receive IOC orders will attempt to fill the order on a principal basis only, and cancel any unfilled balance. On occasion, this can result in the entire order being cancelled if the market maker does not have any existing inventory of the security in question.
- fok: A Fill or Kill (FOK) order is only executed if the entire order quantity can be filled, otherwise the order is canceled. Only available with API v2.
"""

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
        assert "/" not in symbol, "Symbol should not contain '/'; e.g. use BTCUSD instead of BTC/USD"
        self.symbol = symbol
        self.trade_mode = trade_mode
        self.client = TradingClient(api_key, secret_key=api_secret, paper=paper)
        self.last_order_id = None
        self.transaction_fee = 0.025

    def trade(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "ioc", # gtc
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
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
                "take_profit": TakeProfitRequest(
                    limit_price=take_profit,
                ) if take_profit is not None else None,
                "stop_loss": StopLossRequest(
                    limit_price=stop_loss,
                ) if stop_loss is not None else None,
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
            print("Order response: ", response)
            self.last_order_id = response.id
            return True

        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return False
        
    def get_clock(self) -> Dict:
        """
        Get the current market clock.

        Returns:
            Dict[str, Union[Clock]]: The current market clock.

        Example:
            >>> clock = order_manager.get_clock()
            >>> print(clock)
            {'is_open': True,
            'next_close': datetime.datetime(2025, 8, 27, 16, 0, tzinfo=TzInfo(-04:00)),
            'next_open': datetime.datetime(2025, 8, 28, 9, 30, tzinfo=TzInfo(-04:00)),
            'timestamp': datetime.datetime(2025, 8, 27, 14, 12, 49, 663325, tzinfo=TzInfo(-04:00))}
        """
        return self.client.get_clock()

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
                position = self.client.get_open_position(symbol_or_asset_id=self.symbol)
                status["position_status"] = PositionStatus(
                    qty=float(position.qty),
                    market_value=float(position.market_value),
                    avg_entry_price=float(position.avg_entry_price),
                    unrealized_pl=float(position.unrealized_pl),
                    unrealized_plpc=float(position.unrealized_plpc),
                    current_price=float(position.current_price),
                )
            except Exception as e:
                print(f"Error getting position status: {str(e)}")
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
                self.client.close_position(symbol_or_asset_id=self.symbol)
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
    
