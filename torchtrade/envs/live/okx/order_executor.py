"""Order executor for OKX Futures trading using python-okx."""
import logging
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Dict, List, Optional

from torchtrade.envs.live.okx.utils import normalize_symbol
from torchtrade.envs.core.common import TradeMode

logger = logging.getLogger(__name__)

_DEFAULT_LOT_SIZE = {"min_qty": 0.001, "qty_step": 0.001}


class PositionMode(Enum):
    """
    Position mode for OKX Futures.

    - NET: Net mode — single position per instrument (mgnMode determines margin).
    - LONG_SHORT: Long/short mode — separate long and short positions simultaneously.
    """
    NET = "net_mode"
    LONG_SHORT = "long_short_mode"


class MarginMode(Enum):
    """
    Margin mode for OKX Futures positions.

    - ISOLATED: Margin is isolated per position. tdMode="isolated" in OKX API.
    - CROSS: Margin is shared across all positions. tdMode="cross" in OKX API.
    """
    ISOLATED = "isolated"
    CROSS = "cross"

    def to_okx(self) -> str:
        """Convert to OKX tdMode string.

        Returns:
            "isolated" or "cross"
        """
        return self.value


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


class OKXFuturesOrderClass:
    """
    Order executor for OKX Futures trading using python-okx.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Market orders
    - Bracket orders with stop-loss and take-profit (via attachAlgoOrds)
    - Demo and production modes
    """

    def __init__(
        self,
        symbol: str,
        trade_mode: TradeMode = "quantity",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        position_mode: PositionMode = PositionMode.NET,
        client=None,
        account_client=None,
        public_client=None,
    ):
        """
        Initialize the OKXFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTC-USDT-SWAP")
            trade_mode: "quantity" for unit-based orders
            api_key: OKX API key
            api_secret: OKX API secret key
            passphrase: OKX API passphrase
            demo: Whether to use demo trading (default: True for safety)
            leverage: Leverage to use (1-125, default: 1)
            margin_mode: ISOLATED or CROSS
            position_mode: NET or LONG_SHORT
            client: Optional pre-configured Trade client for dependency injection
            account_client: Optional pre-configured Account client
            public_client: Optional pre-configured PublicData client
        """
        self.symbol = normalize_symbol(symbol)
        self.trade_mode = trade_mode
        self.demo = demo
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.position_mode = position_mode
        self.last_order_id = None
        self._lot_size_cache: Optional[Dict[str, float]] = None
        self._tick_size: Optional[float] = None
        self._tick_decimals: int = 0

        flag = "1" if demo else "0"

        # Initialize OKX clients
        if client is not None:
            self.client = client
        else:
            import okx.Trade as Trade
            self.client = Trade.TradeAPI(api_key, api_secret, passphrase, False, flag)

        if account_client is not None:
            self.account_client = account_client
        else:
            import okx.Account as Account
            self.account_client = Account.AccountAPI(api_key, api_secret, passphrase, False, flag)

        if public_client is not None:
            self.public_client = public_client
        else:
            import okx.PublicData as PublicData
            self.public_client = PublicData.PublicAPI(flag=flag)

        # Setup futures account and fetch price precision
        self._setup_futures_account()
        self._fetch_price_precision()

    def _fetch_price_precision(self):
        """Fetch and cache tick size (and lot size) from OKX instruments info.

        Populates both _tick_size and _lot_size_cache from a single API call.
        """
        try:
            response = self.public_client.get_instruments(
                instType="SWAP", instId=self.symbol,
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                logger.warning("get_instruments failed, prices will not be rounded")
                return
            instruments = response.get("data", [])
            if instruments:
                instrument = instruments[0]
                # Tick size for price quantization
                tick_str = instrument.get("tickSz", "0")
                tick_size = float(tick_str)
                if tick_size > 0:
                    self._tick_size = tick_size
                    if '.' in tick_str:
                        decimal_part = tick_str.rstrip('0').split('.')[1]
                        self._tick_decimals = len(decimal_part) if decimal_part else 0
                    logger.info(f"Tick size for {self.symbol}: {self._tick_size} ({self._tick_decimals} decimals)")

                # Cache lot size
                self._lot_size_cache = {
                    "min_qty": float(instrument.get("minSz", 0.001)),
                    "qty_step": float(instrument.get("lotSz", 0.001)),
                }
        except Exception as e:
            logger.warning(f"Could not fetch tick size for {self.symbol}: {e}")

    def _round_price(self, price: float) -> float:
        """Round a price to the nearest tick size."""
        if self._tick_size is not None:
            rounded = round(price / self._tick_size) * self._tick_size
            return round(rounded, self._tick_decimals)
        return price

    def _format_price(self, price: float) -> str:
        """Round price to tick size and format as deterministic string."""
        rounded = self._round_price(price)
        if self._tick_size is not None:
            return f"{rounded:.{self._tick_decimals}f}"
        return str(rounded)

    def _format_size(self, qty: float) -> str:
        """Quantize quantity to lot size constraints and format as string.

        Rounds down to nearest lot step and validates against min size.

        Args:
            qty: Raw quantity to quantize

        Returns:
            Formatted quantity string

        Raises:
            ValueError: If quantized quantity is below min_qty
        """
        lot = self.get_lot_size()
        step = Decimal(str(lot["qty_step"]))
        min_qty = Decimal(str(lot["min_qty"]))
        q = Decimal(str(qty))
        q = (q / step).to_integral_value(rounding=ROUND_DOWN) * step
        if q < min_qty:
            raise ValueError(f"qty {qty} below min_qty {min_qty} after quantizing to step {step}")
        return format(q, "f")

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
        # Set position mode
        pos_mode = self.position_mode.value
        try:
            res = self.account_client.set_position_mode(posMode=pos_mode)
            if str(res.get("code", "-1")) == "0":
                logger.info(f"Position mode set to {pos_mode}")
            else:
                logger.warning(f"Failed to set position mode: code={res.get('code')} msg={res.get('msg')}")
        except Exception as e:
            logger.warning(f"Could not set position mode (may already be configured): {e}")

        # Set leverage
        try:
            res = self.account_client.set_leverage(
                instId=self.symbol,
                lever=str(self.leverage),
                mgnMode=self.margin_mode.to_okx(),
            )
            if str(res.get("code", "-1")) != "0":
                logger.warning(f"Failed to set leverage: code={res.get('code')} msg={res.get('msg')}")
        except Exception as e:
            logger.warning(f"Could not set leverage (may already be configured): {e}")

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
        Execute a futures trade using OKX API.

        Args:
            side: "buy" or "sell"
            quantity: Amount to trade in contracts/base asset units
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            take_profit: Take profit price
            stop_loss: Stop loss price
            reduce_only: If True, only reduce position

        Returns:
            bool: True if order was submitted successfully
        """
        if order_type.lower() == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders")

        try:
            params = {
                "instId": self.symbol,
                "tdMode": self.margin_mode.to_okx(),
                "side": side.lower(),
                "ordType": order_type.lower(),
                "sz": self._format_size(quantity),
            }

            if limit_price is not None:
                params["px"] = self._format_price(limit_price)

            if reduce_only:
                params["reduceOnly"] = True

            # Position side for long/short mode
            if self.position_mode == PositionMode.LONG_SHORT:
                if reduce_only:
                    params["posSide"] = "short" if side.lower() == "buy" else "long"
                else:
                    params["posSide"] = "long" if side.lower() == "buy" else "short"

            # Attach SL/TP as algo orders
            if take_profit is not None or stop_loss is not None:
                algo_ord = {}
                if take_profit is not None:
                    algo_ord["tpTriggerPx"] = self._format_price(take_profit)
                    algo_ord["tpOrdPx"] = "-1"  # Market price
                if stop_loss is not None:
                    algo_ord["slTriggerPx"] = self._format_price(stop_loss)
                    algo_ord["slOrdPx"] = "-1"
                params["attachAlgoOrds"] = [algo_ord]

            response = self.client.place_order(**params)

            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"Order rejected (code={code}): {msg}")
                return False

            # Extract order ID
            data = response.get("data", [])
            if data and isinstance(data[0], dict):
                self.last_order_id = data[0].get("ordId")
            order_id_str = f" (ID: {self.last_order_id})" if self.last_order_id else ""
            logger.info(f"Order executed: {side} {quantity} @ {order_type}{order_id_str}")

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Optional[PositionStatus]]:
        """
        Get current position status.

        Returns:
            Dictionary containing position_status
        """
        status = {}

        try:
            response = self.account_client.get_positions(
                instType="SWAP",
                instId=self.symbol,
            )

            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"get_positions failed (code={code}): {msg}")
                status["position_status"] = None
                return status

            positions = response.get("data", [])

            # Find non-zero positions
            non_zero = [p for p in positions if float(p.get("pos", 0)) != 0]
            if len(non_zero) > 1:
                logger.error("Multiple open positions in LONG_SHORT mode are not supported by this env")
                status["position_status"] = None
                return status
            pos = non_zero[0] if non_zero else None

            if pos is not None:
                raw_pos = float(pos.get("pos", 0))
                pos_side = pos.get("posSide", "net")

                # Determine signed quantity
                if pos_side == "short":
                    qty = -abs(raw_pos)
                elif pos_side == "long":
                    qty = abs(raw_pos)
                else:
                    # Net mode: sign from pos field
                    qty = raw_pos

                entry_price = float(pos.get("avgPx") or "0")
                mark_price = float(pos.get("markPx") or str(entry_price))
                unrealized_pnl = float(pos.get("upl") or "0")
                unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(qty, entry_price, mark_price)
                notional_value = float(pos.get("notionalUsd") or str(abs(qty * mark_price)))
                liq_price = float(pos.get("liqPx") or "0")

                status["position_status"] = PositionStatus(
                    qty=qty,
                    notional_value=notional_value,
                    entry_price=entry_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    mark_price=mark_price,
                    leverage=int(float(pos.get("lever") or str(self.leverage))),
                    margin_mode=pos.get("mgnMode", self.margin_mode.to_okx()),
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
        Get futures account balance using OKX API.

        Returns:
            Dictionary with balance information

        Raises:
            RuntimeError: If balance cannot be retrieved
        """
        try:
            response = self.account_client.get_account_balance()
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                raise RuntimeError(f"get_account_balance failed (code={code}): {msg}")

            data = response.get("data", [])
            if not data:
                raise RuntimeError("No account data returned from OKX")

            account = data[0]
            total_equity = float(account.get("totalEq") or "0")
            # Available equity for trading
            details = account.get("details", [])
            available = 0.0
            for detail in details:
                if detail.get("ccy") == "USDT":
                    available = float(detail.get("availBal") or "0")
                    break

            total_pnl = float(account.get("upl") or "0")

            result = {
                "total_wallet_balance": total_equity,
                "available_balance": available,
                "total_unrealized_profit": total_pnl,
                "total_margin_balance": total_equity,
            }

            logger.debug(f"Account balance: total={total_equity:.2f}, available={available:.2f}, pnl={total_pnl:.4f}")

            if self.demo and total_equity == 0:
                logger.warning(
                    "Demo account balance is 0 USDT! "
                    "Please fund your OKX demo account at: "
                    "https://www.okx.com (Demo Trading)"
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
        response = self.public_client.get_mark_price(
            instType="SWAP",
            instId=self.symbol,
        )

        code = response.get("code", "-1")
        if str(code) != "0":
            msg = response.get("msg", "unknown error")
            raise RuntimeError(f"get_mark_price failed (code={code}): {msg}")

        data = response.get("data", [])
        if data:
            mark_price = data[0].get("markPx")
            if mark_price:
                return float(mark_price)

        raise RuntimeError(f"No mark price data for {self.symbol}")

    def get_lot_size(self) -> Dict[str, float]:
        """
        Get and cache lot size constraints for the symbol.

        Returns:
            Dictionary with 'min_qty' and 'qty_step' for the symbol.
        """
        if self._lot_size_cache is not None:
            return self._lot_size_cache

        try:
            response = self.public_client.get_instruments(
                instType="SWAP", instId=self.symbol,
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.warning(f"get_instruments failed (code={code}): {msg}, using defaults")
                self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()
                return self._lot_size_cache
            instruments = response.get("data", [])
            if instruments:
                instrument = instruments[0]
                self._lot_size_cache = {
                    "min_qty": float(instrument.get("minSz", 0.001)),
                    "qty_step": float(instrument.get("lotSz", 0.001)),
                }
            else:
                logger.warning(f"No instrument info for {self.symbol}, using defaults")
                self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()
        except Exception as e:
            logger.warning(f"Failed to fetch lot size for {self.symbol}: {e}, using defaults")
            self._lot_size_cache = _DEFAULT_LOT_SIZE.copy()

        return self._lot_size_cache

    def get_open_orders(self) -> Optional[List[Dict]]:
        """Get all open orders for the symbol.

        Returns:
            List of open orders, or None if the API call failed.
        """
        try:
            response = self.client.get_order_list(
                instType="SWAP",
                instId=self.symbol,
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"get_order_list failed (code={code}): {msg}")
                return None
            return response.get("data", [])
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return None

    def cancel_open_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            orders = self.get_open_orders()
            if orders is None:
                return False
            if not orders:
                logger.debug(f"No open orders to cancel for {self.symbol}")
                return True

            for order in orders:
                order_id = order.get("ordId")
                if order_id:
                    response = self.client.cancel_order(
                        instId=self.symbol,
                        ordId=order_id,
                    )
                    code = response.get("code", "-1")
                    if str(code) != "0":
                        msg = response.get("msg", "unknown error")
                        logger.error(f"Cancel order {order_id} rejected (code={code}): {msg}")
                        return False

            logger.debug(f"Cancelled all open orders for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling open orders: {str(e)}")
            return False

    def close_position(self) -> bool:
        """
        Close all open positions for the symbol.

        Returns:
            bool: True if all positions were closed successfully
        """
        try:
            response = self.account_client.get_positions(
                instType="SWAP", instId=self.symbol,
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"get_positions failed in close_position (code={code}): {msg}")
                return False
            positions = response.get("data", [])
            non_zero = [p for p in positions if float(p.get("pos", 0)) != 0]

            if not non_zero:
                logger.debug("No open position to close")
                return True

            all_closed = True
            for pos in non_zero:
                raw_pos = float(pos.get("pos", 0))
                pos_side = pos.get("posSide", "net")

                if pos_side in ("long", "net") and raw_pos > 0:
                    close_side = "sell"
                else:
                    close_side = "buy"

                params = {
                    "instId": self.symbol,
                    "tdMode": self.margin_mode.to_okx(),
                    "side": close_side,
                    "ordType": "market",
                    "sz": self._format_size(abs(raw_pos)),
                    "reduceOnly": True,
                }

                if self.position_mode == PositionMode.LONG_SHORT:
                    params["posSide"] = pos_side

                response = self.client.place_order(**params)
                code = response.get("code", "-1")
                if str(code) != "0":
                    msg = response.get("msg", "unknown error")
                    logger.error(f"Close order rejected (code={code}): {msg}")
                    all_closed = False
                else:
                    logger.info(f"Position closed: {abs(raw_pos)} {close_side}")

            return all_closed

        except Exception as e:
            logger.warning(f"close_position order failed: {e}; re-querying position")
            try:
                resp = self.account_client.get_positions(instType="SWAP", instId=self.symbol)
                if str(resp.get("code", "-1")) == "0":
                    still_open = any(float(p.get("pos", 0)) != 0 for p in resp.get("data", []))
                    if not still_open:
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
            leverage: New leverage value (1-125)

        Returns:
            bool: True if successful
        """
        try:
            response = self.account_client.set_leverage(
                instId=self.symbol,
                lever=str(leverage),
                mgnMode=self.margin_mode.to_okx(),
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"set_leverage rejected (code={code}): {msg}")
                return False
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
            mode: New margin mode (ISOLATED or CROSS)

        Returns:
            bool: True if successful
        """
        try:
            response = self.account_client.set_leverage(
                instId=self.symbol,
                lever=str(self.leverage),
                mgnMode=mode.to_okx(),
            )
            code = response.get("code", "-1")
            if str(code) != "0":
                msg = response.get("msg", "unknown error")
                logger.error(f"set_margin_mode rejected (code={code}): {msg}")
                return False
            self.margin_mode = mode
            logger.info(f"Margin mode set to {mode.value} for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting margin mode: {str(e)}")
            return False
