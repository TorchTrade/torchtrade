"""Order executor for OKX Futures trading using python-okx."""
import logging

from torchtrade.envs.live.shared.executor_helpers import ExecutorHelpersMixin

from torchtrade.envs.utils.precision import decimals_for_step
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from torchtrade.envs.live.okx.utils import normalize_symbol
from torchtrade.envs.core.common import TradeMode
from torchtrade.envs.core.state import POSITION_UNKNOWN
from torchtrade.envs.utils.leverage import (
    leverage_already_set,
    require_dict_response,
    require_leverage_applied,
)

logger = logging.getLogger(__name__)

# Venue taker rate, read by env.py and env_sltp.py alike (#278).
TAKER_FEE = 0.0005

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


@dataclass
class PositionStatus:
    qty: float  # Positive for long, negative for short
    notional_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    mark_price: float
    leverage: float  # float, not int: int() truncated 1.5x to 1x, which then took
    # the no-liquidation branch and reported a levered position as safe (#277).
    margin_mode: str
    liquidation_price: float


class OKXFuturesOrderClass(ExecutorHelpersMixin):
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
        self._lot_size_decimals: int = 3  # Default; updated from instrument data
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
                    # Both, or neither: assigning the size before computing the decimals
                    # left a half-configured executor formatting prices at 0 dp.
                    tick_decimals = decimals_for_step(tick_size)
                    self._tick_size, self._tick_decimals = tick_size, tick_decimals
                    logger.info(f"Tick size for {self.symbol}: {self._tick_size} ({self._tick_decimals} decimals)")

                # Cache lot size and derive decimal places from the step
                # float() first: on a blank lotSz this raises into the handler below
                # WITHOUT having already zeroed _lot_size_decimals, whose declared
                # default is 3. Zeroing it re-created the #278 rounding bug on the
                # degraded path -- _format_size(0.977065) -> '1'.
                lot_sz_str = instrument.get("lotSz", "0.001")
                # Everything parsed BEFORE anything is assigned: a blank minSz raising
                # between the two left decimals overwritten and the cache None, which
                # falls back to a 0.001 step and re-creates the #278 rounding bug.
                qty_step = float(lot_sz_str)
                min_qty = float(instrument.get("minSz", 0.001))
                lot_decimals = decimals_for_step(lot_sz_str)
                self._lot_size_decimals = lot_decimals
                self._lot_size_cache = {"min_qty": min_qty, "qty_step": qty_step}
        except Exception as e:
            logger.warning(f"Could not fetch tick size for {self.symbol}: {e}")

    def _format_size(self, qty: float) -> str:
        """Quantize quantity to lot size step, enforce minimum, and format as string."""
        lot = self.get_lot_size()
        step = lot["qty_step"]
        min_qty = lot["min_qty"]
        quantized = math.floor(qty / step) * step
        if quantized < min_qty:
            quantized = min_qty
        # Use _lot_size_decimals derived from instrument data, not str(float) which
        # can produce scientific notation for small steps (e.g. 1e-05 → decimals=0)
        return f"{quantized:.{self._lot_size_decimals}f}"


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

        # OKX's docs: "posSide is only required when margin mode is isolated in
        # long/short position mode". Its leverage-scope table gives the reason -- that
        # combination is the ONLY one stored per side; every other is per instrument
        # family, so one call covers it. Omitting the per-side calls left the env sizing
        # against leverage the account never had, which is #277 on this one config, and
        # became a hard construction failure once #277 stopped swallowing rejections
        # (#363). Narrow because posSide is not APPLICABLE elsewhere -- not because
        # sending it there is refused, which is undocumented either way.
        per_side = (
            self.margin_mode is MarginMode.ISOLATED
            and self.position_mode is PositionMode.LONG_SHORT
        )
        for pos_side in (("long", "short") if per_side else (None,)):
            self._apply_leverage(pos_side)

    def _apply_leverage(self, pos_side: Optional[str] = None):
        """Set and verify the leverage for one side, or for the net position."""
        # Not tolerated like the position mode above: leverage sizes every position (#277).
        request = dict(
            instId=self.symbol,
            lever=str(self.leverage),
            mgnMode=self.margin_mode.value,
        )
        if pos_side is not None:
            request["posSide"] = pos_side
        try:
            res = self.account_client.set_leverage(**request)
        except Exception as e:
            if not leverage_already_set(e):
                raise
        else:
            require_dict_response(self.symbol, self.leverage, res)
            # "-1" default, matching every other code check in this file: a response
            # with no code is an adapter surprise, not a confirmation.
            code = str(res.get("code", "-1"))
            if code != "0":
                # sMsg carries the real reason; top-level msg is often "All operations
                # failed" or empty, which points the operator at nothing.
                entries = res.get("data") or []
                msg = (entries[0].get("sMsg") if entries else None) or res.get(
                    "msg", "unknown error"
                )
                raise ValueError(
                    f"okx refused {self.leverage}x leverage for {self.symbol} "
                    f"(code={code}): {msg}"
                )

            # An empty list is not a pass -- okx does not legitimately return one on
            # success, and treating it as "nothing to check" is how this check goes
            # inert. With posSide sent the response is a single entry echoing it; the
            # loop predates the per-side calls and costs nothing.
            entries = res.get("data") or []
            if not entries:
                raise ValueError(
                    f"okx confirmed no leverage for {self.symbol}: the set-leverage "
                    f"response carried no data to check {self.leverage}x against."
                )
            for entry in entries:
                # The echoed side too: a response carrying posSide="long" for the short
                # call passed verification, because only `lever` was ever compared.
                if pos_side is not None and entry.get("posSide") != pos_side:
                    raise ValueError(
                        f"okx confirmed leverage for posSide={entry.get('posSide')!r} "
                        f"when {pos_side!r} was requested for {self.symbol}"
                    )
                require_leverage_applied(self.symbol, self.leverage, entry, "lever")

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
                "tdMode": self.margin_mode.value,
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
                status["position_status"] = POSITION_UNKNOWN
                return status

            positions = response.get("data", [])

            # Find non-zero positions
            non_zero = [p for p in positions if float(p.get("pos", 0)) != 0]
            if len(non_zero) > 1:
                logger.error("Multiple open positions in LONG_SHORT mode are not supported by this env")
                status["position_status"] = POSITION_UNKNOWN
                return status
            pos = non_zero[0] if non_zero else None

            if pos is not None:
                raw_pos = float(pos.get("pos", 0))
                pos_side = pos.get("posSide")

                # An unrecognised posSide used to fall through to the net-mode branch and
                # keep raw_pos POSITIVE -- so a hedge-mode short read as a long, with no
                # error, straight into the trade path. Worse than bitget/bybit, which at
                # least degrade to POSITION_UNKNOWN. A direction is never guessed (#341).
                if pos_side not in ("short", "long", "net"):
                    logger.error(
                        "okx reported an unusable posSide (%r); refusing to infer a "
                        "direction", pos_side
                    )
                    status["position_status"] = POSITION_UNKNOWN
                    return status

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
                    # `in (None, "")` and not `or`: a venue-reported leverage of numeric 0 is
                    # falsy, and `or` would swap it for the config value -- inventing a
                    # liquidation distance from a leverage the venue never confirmed (#277).
                    leverage=float(
                        self.leverage if pos.get("lever") in (None, "") else pos.get("lever")
                    ),
                    margin_mode=pos.get("mgnMode", self.margin_mode.value),
                    liquidation_price=liq_price,
                )
            else:
                status["position_status"] = None

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            status["position_status"] = POSITION_UNKNOWN

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
            # See bitget: a fabricated 0 equity disables every downstream guard (#277).
            raw_equity = account.get("totalEq")
            if raw_equity is None or raw_equity == "":
                raise ValueError(
                    f"okx returned no total equity (got {raw_equity!r}); refusing to "
                    f"report an equity of 0")
            total_equity = float(raw_equity)
            # Available equity for trading
            details = account.get("details", [])
            available = 0.0
            for detail in details:
                if detail.get("ccy") == "USDT":
                    available = float(detail.get("availBal") or "0")
                    break

            total_pnl = float(account.get("upl") or "0")
            maintenance = account.get("mmr")

            result = {
                "total_wallet_balance": total_equity,
                "available_balance": available,
                "total_unrealized_profit": total_pnl,
                "total_margin_balance": total_equity,
                "total_maintenance_margin": (
                    float(maintenance) if maintenance not in (None, "") else None
                ),
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
                pos_side = pos.get("posSide")
                if pos_side not in ("long", "short", "net"):
                    logger.error(f"Refusing to close a position with posSide {pos_side!r}")
                    all_closed = False
                    continue
                if pos_side in ("long", "net") and raw_pos > 0:
                    close_side = "sell"
                else:
                    close_side = "buy"

                # Use raw string from exchange (strip sign) to avoid float artifacts
                size_str = pos.get("pos", "0").lstrip("-")

                params = {
                    "instId": self.symbol,
                    "tdMode": self.margin_mode.value,
                    "side": close_side,
                    "ordType": "market",
                    "sz": size_str,
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
                mgnMode=mode.value,
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
