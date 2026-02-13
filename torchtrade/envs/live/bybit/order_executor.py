"""Order executor for Bybit Futures trading using pybit."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from torchtrade.envs.live.bybit.utils import normalize_symbol
from torchtrade.envs.core.common import TradeMode

logger = logging.getLogger(__name__)


class PositionMode(Enum):
    """
    Position mode for Bybit Futures.

    - ONE_WAY: Single net position per symbol (positionIdx=0).
    - HEDGE: Separate long/short positions simultaneously.
    """
    ONE_WAY = "one_way"
    HEDGE = "hedge"


class MarginMode(Enum):
    """
    Margin mode for Bybit Futures positions.

    - ISOLATED: Margin is isolated per position. tradeMode=1 in pybit.
    - CROSSED: Margin is shared across all positions. tradeMode=0 in pybit.
    """
    ISOLATED = "isolated"
    CROSSED = "crossed"

    def to_pybit(self) -> int:
        """Convert to pybit tradeMode integer.

        Returns:
            1 for ISOLATED, 0 for CROSSED
        """
        return 1 if self == MarginMode.ISOLATED else 0


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


class BybitFuturesOrderClass:
    """
    Order executor for Bybit Futures trading using pybit.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-100x)
    - Market orders
    - Bracket orders with stop-loss and take-profit (native pybit support)
    - Demo (testnet) and production modes
    """

    def __init__(
        self,
        symbol: str,
        trade_mode: TradeMode = "quantity",
        api_key: str = "",
        api_secret: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        position_mode: PositionMode = PositionMode.ONE_WAY,
        client: Optional[object] = None,
    ):
        """
        Initialize the BybitFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            trade_mode: "quantity" for unit-based orders
            api_key: Bybit API key
            api_secret: Bybit API secret
            demo: Whether to use demo/testnet (default: True for safety)
            leverage: Leverage to use (1-100, default: 1)
            margin_mode: ISOLATED or CROSSED
            position_mode: ONE_WAY or HEDGE
            client: Optional pre-configured pybit HTTP client for dependency injection
        """
        self.symbol = normalize_symbol(symbol)
        self.trade_mode = trade_mode
        self.demo = demo
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.position_mode = position_mode
        self.last_order_id = None

        # Initialize pybit client
        if client is not None:
            self.client = client
        else:
            from pybit.unified_trading import HTTP

            self.client = HTTP(
                testnet=demo,
                api_key=api_key,
                api_secret=api_secret,
            )

        # Setup futures account
        self._setup_futures_account()

    def _calculate_unrealized_pnl_pct(self, qty: float, entry_price: float, mark_price: float) -> float:
        """Calculate unrealized PnL percentage."""
        if entry_price <= 0:
            return 0.0
        if qty > 0:
            return (mark_price - entry_price) / entry_price
        else:
            return (entry_price - mark_price) / entry_price

    def _setup_futures_account(self):
        """Configure futures account settings."""
        try:
            # Set position mode
            try:
                if self.position_mode == PositionMode.ONE_WAY:
                    mode = 0  # MergedSingle
                else:
                    mode = 3  # BothSides
                self.client.switch_position_mode(
                    category="linear",
                    symbol=self.symbol,
                    mode=mode,
                )
                logger.info(f"Position mode set to {self.position_mode.value}")
            except Exception as e:
                error_str = str(e).lower()
                if "position" in error_str:
                    logger.warning("Could not set position mode: close open positions before switching modes.")
                else:
                    logger.warning(f"Could not set position mode (may already be configured): {e}")

            # Set leverage
            self.client.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )

            # Set margin mode
            try:
                self.client.switch_margin_mode(
                    category="linear",
                    symbol=self.symbol,
                    tradeMode=self.margin_mode.to_pybit(),
                    buyLeverage=str(self.leverage),
                    sellLeverage=str(self.leverage),
                )
                logger.info(f"Margin mode set to {self.margin_mode.value}")
            except Exception as e:
                logger.warning(f"Could not set margin mode (may already be configured): {e}")

        except Exception as e:
            logger.warning(f"Could not setup futures account (may already be configured): {e}")

    def trade(
        self,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
    ) -> bool:
        """
        Execute a futures trade using pybit.

        Args:
            side: "buy" or "sell"
            quantity: Amount to trade in base asset units
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            take_profit: Take profit price (set directly on order)
            stop_loss: Stop loss price (set directly on order)
            reduce_only: If True, only reduce position

        Returns:
            bool: True if order was submitted successfully
        """
        try:
            side_upper = side.capitalize()  # Bybit uses "Buy" / "Sell"
            order_type_title = order_type.capitalize()  # "Market" / "Limit"

            params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side_upper,
                "orderType": order_type_title,
                "qty": str(quantity),
            }

            if limit_price is not None:
                params["price"] = str(limit_price)

            if take_profit is not None:
                params["takeProfit"] = str(take_profit)

            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)

            if reduce_only:
                params["reduceOnly"] = True

            # Position index for hedge mode
            if self.position_mode == PositionMode.HEDGE:
                if reduce_only:
                    # Closing: Buy closes short (positionIdx=2), Sell closes long (positionIdx=1)
                    params["positionIdx"] = 2 if side_upper == "Buy" else 1
                else:
                    # Opening: Buy opens long (positionIdx=1), Sell opens short (positionIdx=2)
                    params["positionIdx"] = 1 if side_upper == "Buy" else 2
            else:
                params["positionIdx"] = 0  # One-way mode

            response = self.client.place_order(**params)

            # Extract order ID
            result = response.get("result", {})
            if isinstance(result, dict) and "orderId" in result:
                self.last_order_id = result["orderId"]
                logger.info(f"Order executed: {side} {quantity} @ {order_type} (ID: {self.last_order_id})")
            else:
                logger.info(f"Order executed: {side} {quantity} @ {order_type}")

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Optional[PositionStatus]]:
        """
        Get current order and position status.

        Returns:
            Dictionary containing order_status and position_status
        """
        status = {}

        try:
            # Get position status
            response = self.client.get_positions(
                category="linear",
                symbol=self.symbol,
            )

            positions = response.get("result", {}).get("list", [])

            if positions:
                pos = positions[0]
                size = float(pos.get("size", 0))

                if size != 0:
                    side = pos.get("side", "Buy")
                    qty = size if side == "Buy" else -size

                    entry_price = float(pos.get("avgPrice", 0))
                    mark_price = float(pos.get("markPrice", entry_price))
                    unrealized_pnl = float(pos.get("unrealisedPnl", 0))
                    unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(qty, entry_price, mark_price)

                    liq_price_str = pos.get("liqPrice", "0")
                    liq_price = float(liq_price_str) if liq_price_str and liq_price_str != "" else 0.0

                    status["position_status"] = PositionStatus(
                        qty=qty,
                        notional_value=float(pos.get("positionValue", abs(size * mark_price))),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        mark_price=mark_price,
                        leverage=int(float(pos.get("leverage", self.leverage))),
                        margin_mode=pos.get("tradeMode", str(self.margin_mode.to_pybit())),
                        liquidation_price=liq_price,
                    )
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
        Get futures account balance using pybit.

        Returns:
            Dictionary with balance information

        Raises:
            RuntimeError: If balance cannot be retrieved
        """
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")

            accounts = response.get("result", {}).get("list", [])
            if not accounts:
                raise RuntimeError("No account data returned")

            account = accounts[0]
            total_equity = float(account.get("totalEquity", 0))
            available = float(account.get("totalAvailableBalance", 0))
            total_pnl = float(account.get("totalPerpUPL", 0))
            margin_balance = float(account.get("totalMarginBalance", total_equity))

            result = {
                "total_wallet_balance": total_equity,
                "available_balance": available,
                "total_unrealized_profit": total_pnl,
                "total_margin_balance": margin_balance,
            }

            logger.debug(f"Account balance: total={total_equity:.2f}, available={available:.2f}, pnl={total_pnl:.4f}")

            if self.demo and total_equity == 0:
                logger.warning(
                    "Demo account balance is 0 USDT! "
                    "Please fund your Bybit demo account at: "
                    "https://testnet.bybit.com"
                )

            return result

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
            response = self.client.get_tickers(
                category="linear",
                symbol=self.symbol,
            )

            tickers = response.get("result", {}).get("list", [])
            if tickers:
                mark_price = tickers[0].get("markPrice")
                if mark_price:
                    return float(mark_price)

                # Fallback to last price
                last_price = tickers[0].get("lastPrice")
                if last_price:
                    return float(last_price)

            raise RuntimeError(f"No ticker data for {self.symbol}")

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            logger.error(f"Error getting mark price: {str(e)}")
            raise RuntimeError(f"Failed to get mark price: {e}") from e

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for the symbol."""
        try:
            response = self.client.get_open_orders(
                category="linear",
                symbol=self.symbol,
            )
            return response.get("result", {}).get("list", [])
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []

    def cancel_open_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            self.client.cancel_all_orders(
                category="linear",
                symbol=self.symbol,
            )
            logger.debug(f"Cancelled all open orders for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self) -> bool:
        """
        Close the current position by placing an opposite market order.

        Returns:
            bool: True if position was closed successfully
        """
        try:
            status = self.get_status()
            position = status.get("position_status")

            if position is None or position.qty == 0:
                logger.debug("No open position to close")
                return True

            qty = abs(position.qty)
            side = "Sell" if position.qty > 0 else "Buy"

            params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "reduceOnly": True,
            }

            if self.position_mode == PositionMode.HEDGE:
                # Sell closes long (positionIdx=1), Buy closes short (positionIdx=2)
                params["positionIdx"] = 1 if side == "Sell" else 2
            else:
                params["positionIdx"] = 0

            self.client.place_order(**params)
            logger.info(f"Position closed: {qty} {side}")
            return True

        except Exception as e:
            error_msg = str(e)
            # No position to close is not an error
            if "position" in error_msg.lower() and ("not" in error_msg.lower() or "no" in error_msg.lower()):
                logger.debug("No open position to close")
                return True
            logger.error(f"Error closing position: {error_msg}")
            return False

    def set_leverage(self, leverage: int) -> bool:
        """
        Change leverage for the symbol.

        Args:
            leverage: New leverage value (1-100)

        Returns:
            bool: True if successful
        """
        try:
            self.client.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            self.leverage = leverage
            logger.debug(f"Leverage set to {leverage}x for {self.symbol}")
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
            self.client.switch_margin_mode(
                category="linear",
                symbol=self.symbol,
                tradeMode=mode.to_pybit(),
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )
            self.margin_mode = mode
            logger.info(f"Margin mode set to {mode.value} for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting margin mode: {str(e)}")
            return False
