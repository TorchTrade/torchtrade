"""Order executor for Bybit Futures trading using pybit."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from torchtrade.envs.live.bybit.utils import normalize_symbol
from torchtrade.envs.core.common import TradeMode

logger = logging.getLogger(__name__)

_DEFAULT_LOT_SIZE = {"min_qty": 0.001, "qty_step": 0.001}


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
        self._lot_size_cache: Optional[Dict[str, float]] = None

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
        mode = 0 if self.position_mode == PositionMode.ONE_WAY else 3
        leverage_str = str(self.leverage)

        try:
            self.client.switch_position_mode(
                category="linear", symbol=self.symbol, mode=mode,
            )
            logger.info(f"Position mode set to {self.position_mode.value}")
        except Exception as e:
            logger.warning(f"Could not set position mode (may already be configured): {e}")

        try:
            self.client.set_leverage(
                category="linear", symbol=self.symbol,
                buyLeverage=leverage_str, sellLeverage=leverage_str,
            )
        except Exception as e:
            logger.warning(f"Could not set leverage (may already be configured): {e}")

        try:
            self.client.switch_margin_mode(
                category="linear", symbol=self.symbol,
                tradeMode=self.margin_mode.to_pybit(),
                buyLeverage=leverage_str, sellLeverage=leverage_str,
            )
            logger.info(f"Margin mode set to {self.margin_mode.value}")
        except Exception as e:
            logger.warning(f"Could not set margin mode (may already be configured): {e}")

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
        if order_type.lower() == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders")

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

            # Validate API response
            ret_code = response.get("retCode")
            if ret_code is not None and int(ret_code) != 0:
                ret_msg = response.get("retMsg", "unknown error")
                logger.error(f"Order rejected (retCode={ret_code}): {ret_msg}")
                return False

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

            ret_code = response.get("retCode")
            if ret_code is not None and int(ret_code) != 0:
                ret_msg = response.get("retMsg", "unknown error")
                logger.error(f"get_positions failed (retCode={ret_code}): {ret_msg}")
                status["position_status"] = None
                return status

            positions = response.get("result", {}).get("list", [])

            # Find non-zero positions (handles hedge mode)
            non_zero = [p for p in positions if float(p.get("size", 0)) != 0]
            if len(non_zero) > 1:
                logger.warning("Multiple open positions detected (hedge mode); using first non-zero")
            pos = non_zero[0] if non_zero else None

            if pos is not None:
                size = float(pos.get("size", 0))
                side = pos.get("side", "Buy")
                qty = size if side == "Buy" else -size

                entry_price = float(pos.get("avgPrice", 0))
                mark_price = float(pos.get("markPrice", entry_price))
                unrealized_pnl = float(pos.get("unrealisedPnl", 0))
                unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(qty, entry_price, mark_price)

                liq_price = float(pos.get("liqPrice") or "0")

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
            for account_type in ("UNIFIED", "CONTRACT"):
                response = self.client.get_wallet_balance(accountType=account_type)
                accounts = response.get("result", {}).get("list", [])
                if accounts:
                    break
            else:
                raise RuntimeError("No account data returned from UNIFIED or CONTRACT account types")

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
        response = self.client.get_tickers(
            category="linear",
            symbol=self.symbol,
        )

        ret_code = response.get("retCode")
        if ret_code is not None and int(ret_code) != 0:
            ret_msg = response.get("retMsg", "unknown error")
            raise RuntimeError(f"get_tickers failed (retCode={ret_code}): {ret_msg}")

        tickers = response.get("result", {}).get("list", [])
        if tickers:
            mark_price = tickers[0].get("markPrice")
            if mark_price:
                return float(mark_price)

            last_price = tickers[0].get("lastPrice")
            if last_price:
                return float(last_price)

        raise RuntimeError(f"No ticker data for {self.symbol}")

    def get_lot_size(self) -> Dict[str, float]:
        """
        Get and cache lot size constraints for the symbol.

        Returns:
            Dictionary with 'min_qty' and 'qty_step' for the symbol.
        """
        if self._lot_size_cache is not None:
            return self._lot_size_cache

        try:
            response = self.client.get_instruments_info(
                category="linear", symbol=self.symbol,
            )
            ret_code = response.get("retCode")
            if ret_code is not None and int(ret_code) != 0:
                ret_msg = response.get("retMsg", "unknown error")
                logger.warning(f"get_instruments_info failed (retCode={ret_code}): {ret_msg}, using defaults")
                self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()
                return self._lot_size_cache
            instruments = response.get("result", {}).get("list", [])
            if instruments:
                lot_filter = instruments[0].get("lotSizeFilter", {})
                self._lot_size_cache = {
                    "min_qty": float(lot_filter.get("minOrderQty", 0.001)),
                    "qty_step": float(lot_filter.get("qtyStep", 0.001)),
                }
            else:
                logger.warning(f"No instrument info for {self.symbol}, using defaults")
                self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()
        except Exception as e:
            logger.warning(f"Failed to fetch lot size for {self.symbol}: {e}, using defaults")
            self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()

        return self._lot_size_cache

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
            response = self.client.cancel_all_orders(
                category="linear",
                symbol=self.symbol,
            )
            ret_code = response.get("retCode") if isinstance(response, dict) else None
            if ret_code is not None and int(ret_code) != 0:
                ret_msg = response.get("retMsg", "unknown error")
                logger.error(f"Cancel orders rejected (retCode={ret_code}): {ret_msg}")
                return False
            logger.debug(f"Cancelled all open orders for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self) -> bool:
        """
        Close all open positions for the symbol.

        In hedge mode, closes both long and short sides if both are open.

        Returns:
            bool: True if all positions were closed successfully
        """
        try:
            response = self.client.get_positions(
                category="linear", symbol=self.symbol,
            )
            ret_code = response.get("retCode")
            if ret_code is not None and int(ret_code) != 0:
                ret_msg = response.get("retMsg", "unknown error")
                logger.error(f"get_positions failed in close_position (retCode={ret_code}): {ret_msg}")
                return False
            positions = response.get("result", {}).get("list", [])
            non_zero = [p for p in positions if float(p.get("size", 0)) != 0]

            if not non_zero:
                logger.debug("No open position to close")
                return True

            all_closed = True
            for pos in non_zero:
                size = float(pos.get("size", 0))
                pos_side = pos.get("side", "Buy")
                close_side = "Sell" if pos_side == "Buy" else "Buy"

                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "side": close_side,
                    "orderType": "Market",
                    "qty": str(size),
                    "reduceOnly": True,
                }

                if self.position_mode == PositionMode.HEDGE:
                    # Sell closes long (positionIdx=1), Buy closes short (positionIdx=2)
                    params["positionIdx"] = 1 if close_side == "Sell" else 2
                else:
                    params["positionIdx"] = 0

                response = self.client.place_order(**params)
                ret_code = response.get("retCode")
                if ret_code is not None and int(ret_code) != 0:
                    ret_msg = response.get("retMsg", "unknown error")
                    logger.error(f"Close order rejected (retCode={ret_code}): {ret_msg}")
                    all_closed = False
                else:
                    logger.info(f"Position closed: {size} {close_side}")

            return all_closed

        except Exception as e:
            logger.warning(f"close_position order failed: {e}; re-querying position")
            try:
                status = self.get_status()
                pos = status.get("position_status")
                if pos is None or pos.qty == 0:
                    logger.debug("Position confirmed closed after failed order")
                    return True
            except Exception:
                pass
            logger.error(f"Error closing position: {e}")
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
