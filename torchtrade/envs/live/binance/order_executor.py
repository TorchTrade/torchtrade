import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import warnings
import os
from dotenv import load_dotenv

from torchtrade.envs.core.common import TradeMode

load_dotenv()

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """
    Position side for Binance Futures.

    Binance Futures supports two position modes:
    - One-way mode: Single position per symbol. Use BOTH.
    - Hedge mode: Separate long/short positions simultaneously. Use LONG/SHORT.
    """
    LONG = "LONG"    # Hedge mode: explicit long position
    SHORT = "SHORT"  # Hedge mode: explicit short position
    BOTH = "BOTH"    # One-way mode: single net position (default)


class MarginType(Enum):
    """
    Margin type for Binance Futures positions.

    - ISOLATED: Margin is isolated per position. Losses are limited to
      that position's margin. Lower risk but requires more capital.
    - CROSSED: Margin is shared across all positions. Entire account
      balance can be used to prevent liquidation. Higher risk.
    """
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


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
    qty: float  # Positive for long, negative for short
    notional_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    mark_price: float
    leverage: int
    margin_type: str
    liquidation_price: float


class BinanceFuturesOrderClass:
    """
    Order executor for Binance Futures trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Market, limit, stop-market, and take-profit orders
    - OCO-style bracket orders
    - Demo (mock) and testnet modes for paper trading
    """

    # API endpoints
    PRODUCTION_URL = "https://fapi.binance.com"
    DEMO_URL = "https://testnet.binancefuture.com"  # Demo/Mock trading

    def __init__(
        self,
        symbol: str,
        trade_mode: TradeMode = TradeMode.QUANTITY,
        api_key: str = "",
        api_secret: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_type: MarginType = MarginType.ISOLATED,
        client: Optional[object] = None,
    ):
        """
        Initialize the BinanceFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            trade_mode: TradeMode.QUANTITY for unit-based orders
            api_key: Binance API key
            api_secret: Binance API secret
            demo: Whether to use demo trading (default: True for safety)
            leverage: Leverage to use (1-125, default: 1)
            margin_type: ISOLATED (margin per position, limits loss) or
                        CROSSED (shared margin, higher liquidation risk)
            client: Optional pre-configured Client for dependency injection
        """
        # Normalize symbol
        if "/" in symbol:
            warnings.warn(
                f"Symbol {symbol} contains '/'; will be changed to {symbol.replace('/', '')}."
            )
            symbol = symbol.replace("/", "")
        self.symbol = symbol

        self.trade_mode = trade_mode
        self.demo = demo
        self.leverage = leverage
        self.margin_type = margin_type
        self.last_order_id = None

        # Initialize client
        if client is not None:
            self.client = client
        else:
            try:
                from binance.client import Client
                self.client = Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=demo  # Use testnet for demo mode
                )
            except ImportError:
                raise ImportError("python-binance is required. Install with: pip install python-binance")

        # Setup futures account
        self._setup_futures_account()

    def _setup_futures_account(self):
        """Configure futures account settings."""
        try:
            # Set leverage
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )

            # Set margin type
            try:
                self.client.futures_change_margin_type(
                    symbol=self.symbol,
                    marginType=self.margin_type.value
                )
            except Exception as e:
                # May fail if already set to this margin type
                if "No need to change margin type" not in str(e):
                    logger.warning(f"Could not set margin type: {e}")

        except Exception as e:
            logger.warning(f"Could not setup futures account: {e}")

    def trade(
        self,
        side: str,
        quantity: float,
        order_type: str = "market",
        position_side: str = "BOTH",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> bool:
        """
        Execute a futures trade.

        Args:
            side: "BUY" or "SELL"
            quantity: Amount to trade in base asset units
            order_type: "market", "limit", "stop_market", "take_profit_market"
            position_side: "LONG", "SHORT", or "BOTH" (for one-way mode)
            limit_price: Required for limit orders
            stop_price: Required for stop orders
            take_profit: Take profit price (creates separate TP order)
            stop_loss: Stop loss price (creates separate SL order)
            reduce_only: If True, only reduce position (no new positions)
            time_in_force: Time in force ("GTC", "IOC", "FOK")

        Returns:
            bool: True if order was submitted successfully
        """
        try:
            side = side.upper()
            order_type_map = {
                "market": "MARKET",
                "limit": "LIMIT",
                "stop_market": "STOP_MARKET",
                "take_profit_market": "TAKE_PROFIT_MARKET",
            }
            binance_order_type = order_type_map.get(order_type.lower(), "MARKET")

            # Base order parameters
            order_params = {
                "symbol": self.symbol,
                "side": side,
                "type": binance_order_type,
                "quantity": round(quantity, 3),
            }

            # Add position side for hedge mode
            if position_side != "BOTH":
                order_params["positionSide"] = position_side

            # Add reduce only flag
            if reduce_only:
                order_params["reduceOnly"] = "true"

            # Add price parameters based on order type
            if binance_order_type == "LIMIT":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                order_params["price"] = limit_price
                order_params["timeInForce"] = time_in_force

            elif binance_order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                if stop_price is None:
                    raise ValueError("stop_price is required for stop orders")
                order_params["stopPrice"] = stop_price

            # Submit main order
            response = self.client.futures_create_order(**order_params)
            self.last_order_id = response.get("orderId")
            logger.info(f"Order executed: {response}")

            # Create take profit order if specified
            if take_profit is not None and not reduce_only:
                tp_params = {
                    "symbol": self.symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": take_profit,
                    "quantity": round(quantity, 3),
                    "reduceOnly": "true",
                }
                if position_side != "BOTH":
                    tp_params["positionSide"] = position_side
                self.client.futures_create_order(**tp_params)

            # Create stop loss order if specified
            if stop_loss is not None and not reduce_only:
                sl_params = {
                    "symbol": self.symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "STOP_MARKET",
                    "stopPrice": stop_loss,
                    "quantity": round(quantity, 3),
                    "reduceOnly": "true",
                }
                if position_side != "BOTH":
                    sl_params["positionSide"] = position_side
                self.client.futures_create_order(**sl_params)

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Union[OrderStatus, PositionStatus, None]]:
        """
        Get current order and position status.

        Returns:
            Dictionary containing order_status and position_status
        """
        status = {}

        try:
            # Get order status if we have a last order
            if self.last_order_id:
                order = self.client.futures_get_order(
                    symbol=self.symbol,
                    orderId=self.last_order_id
                )
                status["order_status"] = OrderStatus(
                    is_open=order["status"] not in ["FILLED", "CANCELED", "EXPIRED", "REJECTED"],
                    order_id=str(order["orderId"]),
                    filled_qty=float(order.get("executedQty", 0)),
                    filled_avg_price=float(order.get("avgPrice", 0)),
                    status=order["status"],
                    side=order["side"],
                    order_type=order["type"],
                )

            # Get position status
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                qty = float(pos["positionAmt"])
                if qty != 0:
                    entry_price = float(pos["entryPrice"])
                    mark_price = float(pos["markPrice"])
                    unrealized_pnl = float(pos["unRealizedProfit"])

                    # Calculate unrealized PnL percentage
                    if entry_price > 0:
                        if qty > 0:  # Long
                            unrealized_pnl_pct = (mark_price - entry_price) / entry_price
                        else:  # Short
                            unrealized_pnl_pct = (entry_price - mark_price) / entry_price
                    else:
                        unrealized_pnl_pct = 0.0

                    status["position_status"] = PositionStatus(
                        qty=qty,
                        notional_value=float(pos.get("notional", 0)),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        mark_price=mark_price,
                        leverage=int(pos.get("leverage", self.leverage)),
                        margin_type=pos.get("marginType", self.margin_type.value),
                        liquidation_price=float(pos.get("liquidationPrice", 0)),
                    )
                    break
            else:
                status["position_status"] = None

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            status["position_status"] = None

        return status

    def get_account_balance(self) -> Dict[str, float]:
        """
        Get futures account balance.

        Returns:
            Dictionary with balance information

        Raises:
            RuntimeError: If balance cannot be retrieved
        """
        try:
            account = self.client.futures_account()
            return {
                "total_wallet_balance": float(account["totalWalletBalance"]),
                "available_balance": float(account["availableBalance"]),
                "total_unrealized_profit": float(account["totalUnrealizedProfit"]),
                "total_margin_balance": float(account["totalMarginBalance"]),
            }
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            raise RuntimeError(f"Failed to get account balance: {e}") from e

    def get_mark_price(self) -> float:
        """
        Get current mark price for the symbol.

        Returns:
            Current mark price

        Raises:
            RuntimeError: If mark price cannot be retrieved
        """
        try:
            ticker = self.client.futures_mark_price(symbol=self.symbol)
            return float(ticker["markPrice"])
        except Exception as e:
            logger.error(f"Error getting mark price: {str(e)}")
            raise RuntimeError(f"Failed to get mark price: {e}") from e

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for the symbol."""
        try:
            return self.client.futures_get_open_orders(symbol=self.symbol)
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []

    def cancel_open_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info("Open orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self, position_side: str = "BOTH") -> bool:
        """
        Close the current position.

        Args:
            position_side: "LONG", "SHORT", or "BOTH"

        Returns:
            bool: True if position was closed successfully
        """
        try:
            status = self.get_status()
            position = status.get("position_status")

            if position is None or position.qty == 0:
                logger.debug("No position to close")
                return True

            # Determine side to close
            qty = abs(position.qty)
            side = "SELL" if position.qty > 0 else "BUY"

            order_params = {
                "symbol": self.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": round(qty, 3),
                "reduceOnly": "true",
            }

            if position_side != "BOTH":
                order_params["positionSide"] = position_side

            self.client.futures_create_order(**order_params)
            logger.info(f"Position closed: {qty} {side}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    def close_all_positions(self) -> Dict[str, bool]:
        """Close all open positions."""
        try:
            results = {}
            positions = self.client.futures_position_information()

            for pos in positions:
                qty = float(pos["positionAmt"])
                if qty != 0:
                    symbol = pos["symbol"]
                    side = "SELL" if qty > 0 else "BUY"
                    try:
                        self.client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type="MARKET",
                            quantity=round(abs(qty), 3),
                            reduceOnly="true",
                        )
                        results[symbol] = True
                        logger.info(f"Closed position for {symbol}")
                    except Exception as e:
                        logger.error(f"Error closing position for {symbol}: {str(e)}")
                        results[symbol] = False

            return results

        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}

    def set_leverage(self, leverage: int) -> bool:
        """
        Change leverage for the symbol.

        Args:
            leverage: New leverage value (1-125)

        Returns:
            bool: True if successful
        """
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=leverage
            )
            self.leverage = leverage
            logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize with demo mode
    order_manager = BinanceFuturesOrderClass(
        symbol="BTCUSDT",
        trade_mode=TradeMode.QUANTITY,
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_SECRET", ""),
        demo=True,
        leverage=5,
    )

    # Get account balance
    balance = order_manager.get_account_balance()
    print(f"Account balance: {balance}")

    # Get mark price
    price = order_manager.get_mark_price()
    print(f"Mark price: {price}")

    # Get status
    status = order_manager.get_status()
    print(f"Status: {status}")
