import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import warnings

logger = logging.getLogger(__name__)


class TradeMode(Enum):
    NOTIONAL = "notional"  # Dollar-based orders (not directly supported, calculated from qty)
    QUANTITY = "quantity"  # Unit-based orders


class PositionMode(Enum):
    """
    Position mode for Bitget Futures.

    Bitget Futures supports two position modes:
    - One-way mode: Single net position per symbol.
    - Hedge mode: Separate long/short positions simultaneously.
    """
    ONE_WAY = "one_way_mode"      # Single net position (default)
    HEDGE = "hedge_mode"          # Separate long/short


class MarginMode(Enum):
    """
    Margin mode for Bitget Futures positions.

    - ISOLATED: Margin is isolated per position. Losses are limited to
      that position's margin. Lower risk but requires more capital.
    - CROSSED: Margin is shared across all positions. Entire account
      balance can be used to prevent liquidation. Higher risk.
    """
    ISOLATED = "isolated"
    CROSSED = "crossed"


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
    margin_mode: str
    liquidation_price: float


class BitgetFuturesOrderClass:
    """
    Order executor for Bitget Futures trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Market, limit, and stop orders
    - Bracket orders with stop-loss and take-profit
    - Demo (testnet) and production modes
    """

    def __init__(
        self,
        symbol: str,
        product_type: str = "SUMCBL",  # SUMCBL=testnet, UMCBL=production
        trade_mode: TradeMode = TradeMode.QUANTITY,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        client: Optional[object] = None,
    ):
        """
        Initialize the BitgetFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            product_type: Product type (SUMCBL=testnet USDT, UMCBL=prod USDT)
            trade_mode: TradeMode.QUANTITY for unit-based orders
            api_key: Bitget API key
            api_secret: Bitget API secret
            passphrase: Bitget API passphrase (required!)
            demo: Whether to use demo/testnet (default: True for safety)
            leverage: Leverage to use (1-125, default: 1)
            margin_mode: ISOLATED (margin per position) or CROSSED (shared margin)
            client: Optional pre-configured Client for dependency injection
        """
        # Normalize symbol
        if "/" in symbol:
            warnings.warn(
                f"Symbol {symbol} contains '/'; will be changed to {symbol.replace('/', '')}."
            )
            symbol = symbol.replace("/", "")
        self.symbol = symbol

        self.product_type = "SUMCBL" if demo else product_type  # Force testnet for demo
        self.trade_mode = trade_mode
        self.demo = demo
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.last_order_id = None

        # Initialize client
        if client is not None:
            self.client = client
        else:
            try:
                from pybitget import Client
                self.client = Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    use_server_time=False
                )
            except ImportError:
                raise ImportError("python-bitget is required. Install with: pip install python-bitget")

        # Setup futures account
        self._setup_futures_account()

    def _setup_futures_account(self):
        """Configure futures account settings."""
        try:
            # Set leverage
            self.client.mix_adjust_leverage(
                symbol=self.symbol,
                productType=self.product_type,
                leverage=str(self.leverage),
                marginCoin="USDT"
            )

            # Set margin mode
            self.client.mix_change_margin_mode(
                symbol=self.symbol,
                productType=self.product_type,
                marginMode=self.margin_mode.value
            )

        except Exception as e:
            # May fail if settings already configured
            logger.warning(f"Could not setup futures account: {e}")

    def trade(
        self,
        side: str,
        quantity: float,
        order_type: str = "market",
        position_side: Optional[str] = None,
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
            side: "buy" or "sell" (lowercase for Bitget)
            quantity: Amount to trade in base asset units
            order_type: "market", "limit", "stop_market"
            position_side: Optional position side for hedge mode
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
            side = side.lower()
            order_type_lower = order_type.lower()

            # Bitget order type mapping
            order_type_map = {
                "market": "market",
                "limit": "limit",
                "stop_market": "stop_market",
            }
            bitget_order_type = order_type_map.get(order_type_lower, "market")

            # Build order parameters
            order_params = {
                "symbol": self.symbol,
                "productType": self.product_type,
                "marginCoin": "USDT",
                "side": side,
                "orderType": bitget_order_type,
                "size": str(round(quantity, 3)),
                "timeInForceValue": time_in_force,
            }

            # Add price for limit orders
            if bitget_order_type == "limit":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                order_params["price"] = str(limit_price)

            # Add trigger price for stop orders
            if bitget_order_type == "stop_market":
                if stop_price is None:
                    raise ValueError("stop_price is required for stop orders")
                order_params["triggerPrice"] = str(stop_price)

            # Submit main order
            response = self.client.mix_place_order(**order_params)

            # Extract order ID from response
            if isinstance(response, dict) and "data" in response:
                self.last_order_id = response["data"].get("orderId")
            logger.info(f"Order executed: {response}")

            # Create take profit order if specified
            if take_profit is not None and not reduce_only:
                tp_side = "sell" if side == "buy" else "buy"
                tp_params = {
                    "symbol": self.symbol,
                    "productType": self.product_type,
                    "marginCoin": "USDT",
                    "side": tp_side,
                    "orderType": "limit",
                    "price": str(take_profit),
                    "size": str(round(quantity, 3)),
                    "reduceOnly": "YES",
                }
                self.client.mix_place_order(**tp_params)

            # Create stop loss order if specified
            if stop_loss is not None and not reduce_only:
                sl_side = "sell" if side == "buy" else "buy"
                sl_params = {
                    "symbol": self.symbol,
                    "productType": self.product_type,
                    "marginCoin": "USDT",
                    "side": sl_side,
                    "orderType": "stop_market",
                    "triggerPrice": str(stop_loss),
                    "size": str(round(quantity, 3)),
                    "reduceOnly": "YES",
                }
                self.client.mix_place_order(**sl_params)

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
                order_response = self.client.mix_get_order_details(
                    symbol=self.symbol,
                    productType=self.product_type,
                    orderId=self.last_order_id
                )
                if isinstance(order_response, dict) and "data" in order_response:
                    order = order_response["data"]
                    status["order_status"] = OrderStatus(
                        is_open=order.get("status") not in ["filled", "cancelled", "expired"],
                        order_id=str(order.get("orderId", "")),
                        filled_qty=float(order.get("filledQty", 0)),
                        filled_avg_price=float(order.get("priceAvg", 0)),
                        status=order.get("status", "unknown"),
                        side=order.get("side", "unknown"),
                        order_type=order.get("orderType", "unknown"),
                    )

            # Get position status
            position_response = self.client.mix_get_single_position(
                symbol=self.symbol,
                productType=self.product_type,
                marginCoin="USDT"
            )

            if isinstance(position_response, dict) and "data" in position_response:
                positions = position_response["data"]
                if isinstance(positions, list) and len(positions) > 0:
                    pos = positions[0]
                    qty = float(pos.get("total", 0))

                    # Bitget uses 'long' and 'short' strings for holdSide
                    hold_side = pos.get("holdSide", "long")
                    if hold_side == "short":
                        qty = -abs(qty)  # Make negative for short

                    if qty != 0:
                        entry_price = float(pos.get("averageOpenPrice", 0))
                        mark_price = float(pos.get("markPrice", entry_price))
                        unrealized_pnl = float(pos.get("unrealizedPL", 0))

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
                            notional_value=abs(float(pos.get("total", 0)) * mark_price),
                            entry_price=entry_price,
                            unrealized_pnl=unrealized_pnl,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            mark_price=mark_price,
                            leverage=int(pos.get("leverage", self.leverage)),
                            margin_mode=pos.get("marginMode", self.margin_mode.value),
                            liquidation_price=float(pos.get("liquidationPrice", 0)),
                        )
                    else:
                        status["position_status"] = None
                else:
                    status["position_status"] = None
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
            accounts_response = self.client.mix_get_accounts(productType=self.product_type)

            if isinstance(accounts_response, dict) and "data" in accounts_response:
                accounts = accounts_response["data"]
                if isinstance(accounts, list) and len(accounts) > 0:
                    account = accounts[0]
                    return {
                        "total_wallet_balance": float(account.get("equity", 0)),
                        "available_balance": float(account.get("available", 0)),
                        "total_unrealized_profit": float(account.get("unrealizedPL", 0)),
                        "total_margin_balance": float(account.get("equity", 0)),
                    }

            raise ValueError("No account data returned from Bitget API")

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
            ticker_response = self.client.mix_get_market_price(
                symbol=self.symbol,
                productType=self.product_type
            )

            if isinstance(ticker_response, dict) and "data" in ticker_response:
                data = ticker_response["data"]
                if isinstance(data, dict):
                    return float(data.get("markPrice", 0))
                elif isinstance(data, list) and len(data) > 0:
                    return float(data[0].get("markPrice", 0))

            raise ValueError("No mark price data returned from Bitget API")

        except Exception as e:
            logger.error(f"Error getting mark price: {str(e)}")
            raise RuntimeError(f"Failed to get mark price: {e}") from e

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for the symbol."""
        try:
            response = self.client.mix_get_open_orders(
                symbol=self.symbol,
                productType=self.product_type
            )
            if isinstance(response, dict) and "data" in response:
                return response["data"]
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []

    def cancel_open_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            self.client.mix_cancel_all_orders(
                symbol=self.symbol,
                productType=self.product_type,
                marginCoin="USDT"
            )
            logger.info("Open orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self, position_side: Optional[str] = None) -> bool:
        """
        Close the current position.

        Args:
            position_side: Optional position side for hedge mode

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
            side = "sell" if position.qty > 0 else "buy"

            order_params = {
                "symbol": self.symbol,
                "productType": self.product_type,
                "marginCoin": "USDT",
                "side": side,
                "orderType": "market",
                "size": str(round(qty, 3)),
                "reduceOnly": "YES",
            }

            self.client.mix_place_order(**order_params)
            logger.info(f"Position closed: {qty} {side}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    def set_leverage(self, leverage: int) -> bool:
        """
        Change leverage for the symbol.

        Args:
            leverage: New leverage value (1-125)

        Returns:
            bool: True if successful
        """
        try:
            self.client.mix_adjust_leverage(
                symbol=self.symbol,
                productType=self.product_type,
                leverage=str(leverage),
                marginCoin="USDT"
            )
            self.leverage = leverage
            logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            return False

    def set_margin_mode(self, mode: MarginMode) -> bool:
        """
        Change margin mode for the symbol.

        Args:
            mode: New margin mode (ISOLATED or CROSSED)

        Returns:
            bool: True if successful
        """
        try:
            self.client.mix_change_margin_mode(
                symbol=self.symbol,
                productType=self.product_type,
                marginMode=mode.value
            )
            self.margin_mode = mode
            logger.info(f"Margin mode set to {mode.value} for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting margin mode: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import os

    # Initialize with demo mode
    print("Initializing BitgetFuturesOrderClass...")

    try:
        order_manager = BitgetFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            api_key=os.getenv("BITGET_API_KEY", ""),
            api_secret=os.getenv("BITGET_SECRET", ""),
            passphrase=os.getenv("BITGET_PASSPHRASE", ""),
            demo=True,
            leverage=5,
        )

        print("✓ Initialized successfully")

        # Get account balance
        print("\nGetting account balance...")
        balance = order_manager.get_account_balance()
        print(f"Account balance: {balance}")

        # Get mark price
        print("\nGetting mark price...")
        price = order_manager.get_mark_price()
        print(f"Mark price: {price}")

        # Get status
        print("\nGetting status...")
        status = order_manager.get_status()
        print(f"Status: {status}")

        print("\n✅ All operations completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
