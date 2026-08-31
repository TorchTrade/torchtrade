import logging

from torchtrade.envs.live.shared.executor_helpers import (
    ExecutorHelpersMixin,
    TickSizeMixin,
)

from torchtrade.envs.utils.precision import decimals_for_step
import math
from decimal import Decimal
from typing import Dict, List, Optional, Union
import warnings

from torchtrade.envs.core.common import TradeMode
from torchtrade.envs.core.common_types import PositionStatus, MarginMode, OrderStatus
from torchtrade.envs.core.state import POSITION_UNKNOWN
from torchtrade.envs.utils.leverage import leverage_already_set, require_leverage_applied


logger = logging.getLogger(__name__)

# Venue taker rate, read by env.py and env_sltp.py alike (#278).
TAKER_FEE = 0.0004


_FALLBACK_QTY_STEP = 0.001
_FALLBACK_QTY_STEP_PAIR = (_FALLBACK_QTY_STEP, 3)


def _step_and_decimals(step_str: str):
    """(step, decimals) from a LOT_SIZE stepSize string such as "0.001" or "1".

    Through Decimal rather than string surgery: `"1E-8".split('.')` yields no fractional
    part, so the naive version reported 0 decimals and the final rounding then annihilated
    the quantity. Binance sends fixed-point today, but okx's _format_size carries a comment
    about being bitten by exactly this, so it is not a hypothetical.

    This was the ONLY venue that got it right, and the rule stayed local while four other
    sites kept doing the string surgery -- which is how bybit came to refuse every open
    over a 1e-06 step (#278). Shared now.
    """
    step = float(Decimal(step_str))
    # isfinite, because decimals_for_step is total by design and answers 0 for NaN. This
    # guard is deliberately fail-CLOSED -- a junk stepSize must abort construction, not
    # sail through and surface as "cannot convert float NaN to integer" at order time.
    if not math.isfinite(step):
        raise ValueError(f"non-finite LOT_SIZE step {step_str!r}")
    return step, decimals_for_step(step_str)


class BinanceFuturesOrderClass(ExecutorHelpersMixin, TickSizeMixin):
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
        trade_mode: TradeMode = "quantity",
        api_key: str = "",
        api_secret: str = "",
        demo: bool = True,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.ISOLATED,
        client: Optional[object] = None,
    ):
        """
        Initialize the BinanceFuturesOrderClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            trade_mode: "quantity" for unit-based orders
            api_key: Binance API key
            api_secret: Binance API secret
            demo: Whether to use demo trading (default: True for safety)
            leverage: Leverage to use (1-125, default: 1)
            margin_mode: ISOLATED (margin per position, limits loss) or
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
        self.margin_mode = margin_mode
        self.last_order_id = None

        self._tick_size: Optional[float] = None
        # symbol -> (step, decimals). Every symbol, not just this one: see
        # _fetch_symbol_filters -- close_all_positions closes whatever the account holds.
        self._qty_steps: Dict[str, tuple] = {}
        self._min_qtys: Dict[str, float] = {}
        # Always present, so `get_lot_size` needs no getattr fallback: a fail-open
        # default is how a missing floor becomes an unchecked order (#414).
        self._min_notional: float = 0.0
        self._tick_decimals: int = 0

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

        # Setup futures account and fetch price precision
        self._setup_futures_account()
        self._fetch_symbol_filters()

    def _setup_futures_account(self):
        """Configure futures account settings."""
        # Not tolerated like the margin type below: leverage sizes every position (#277).
        # Binance caps leverage per notional bracket, so the echo is checked too --
        # accepting the call is not applying the request.
        try:
            response = self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )
        except Exception as e:
            if not leverage_already_set(e):
                raise
        else:
            require_leverage_applied(self.symbol, self.leverage, response, "leverage")

        # Margin type stays a preference for SIZING; note it does affect liquidation
        # risk, so a construction that succeeds guarantees leverage only.
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType=self.margin_mode.value
            )
        except Exception as e:
            # May fail if already set to this margin type
            if "No need to change margin type" not in str(e):
                logger.warning(f"Could not set margin type: {e}")

    def _fetch_symbol_filters(self):
        """Cache the tick size and every symbol's LOT_SIZE step and minQty (#271).

        Every symbol, not just this one: close_all_positions closes whatever the account
        holds, and another symbol's quantity must not be rounded with this symbol's step.
        """
        try:
            symbols = self.client.futures_exchange_info()['symbols']
        except Exception as e:
            # No payload is a transient failure; a payload that omits this symbol is a
            # config error and raises below.
            logger.warning(f"Could not fetch symbol filters for {self.symbol}: {e}")
            return

        for s in symbols:
            symbol = s.get('symbol', '?')
            # Scoped per FILTER: a malformed LOT_SIZE must not cost this symbol its
            # PRICE_FILTER, which would send SLTP stop prices out unrounded.
            for f in s.get('filters', []):
                try:
                    if f['filterType'] == 'LOT_SIZE':
                        self._qty_steps[symbol] = _step_and_decimals(f['stepSize'])
                        self._min_qtys[symbol] = float(f.get('minQty') or 0.0)
                    elif f['filterType'] == 'MIN_NOTIONAL' and symbol == self.symbol:
                        # Read here rather than re-fetched: the env's own
                        # `_get_min_notional` walks these same filters on the plain path,
                        # and the SLTP path had no notional check at all (#414).
                        self._min_notional = float(f.get('notional') or 0.0)
                    elif f['filterType'] == 'PRICE_FILTER' and symbol == self.symbol:
                        tick_str = f['tickSize']
                        tick_size = float(tick_str)
                        # bybit and okx both gate on this; binance did not. _round_price
                        # divides by it, and its only callers are the bracket legs, placed
                        # in a try/except AFTER the entry filled -- so a zero tick does not
                        # crash, it logs "position opened without SL" and leaves a live
                        # position the policy has no way to exit under lock_position_until_sltp.
                        if tick_size > 0:
                            tick_decimals = decimals_for_step(tick_size)
                            self._tick_size, self._tick_decimals = tick_size, tick_decimals
                except Exception as e:
                    logger.warning(
                        f"Skipping malformed {f.get('filterType', '?')} for {symbol}: {e}"
                    )

        if self._tick_size is None:
            logger.warning(f"No PRICE_FILTER found for {self.symbol}, prices will not be rounded")

        # Fail closed: the fallback precision is the bug this fixes, and binance has no
        # minQty check downstream to catch the result.
        if self.symbol not in self._qty_steps:
            raise ValueError(
                f"binance returned no LOT_SIZE for {self.symbol}: not a futures symbol on "
                f"this venue, or the payload changed shape."
            )

    def get_lot_size(self) -> Dict[str, float]:
        """The same `{min_qty, qty_step, min_notional}` the other three venues return.

        binance was the only venue without this accessor, so the shared sizing could not
        ask it for a notional floor and the SLTP path submitted whatever fraction it had
        computed (#414). The values come from filters already cached by
        `_fetch_symbol_filters`; this adds no API call.
        """
        # The SAME fallback `round_quantity` and `_format_quantity` use. Answering 0.0
        # here made this file disagree with itself after a transient exchange-info
        # failure -- `_fetch_symbol_filters` returns early on that path, BEFORE the
        # missing-LOT_SIZE raise, so the caches stay empty and nothing says so.
        # Lazily re-fetch on an empty cache, like bitget/bybit/okx already do.
        # `_fetch_symbol_filters` RETURNS EARLY on a transient exchange-info failure --
        # before its own fail-closed "no LOT_SIZE" raise -- so a single blip at
        # construction left every cache empty and nothing ever repaired them. binance was
        # the only venue that never retried, so one bad moment at startup silently
        # disabled both floors for the life of the process (#414).
        if self.symbol not in self._qty_steps:
            self._fetch_symbol_filters()

        step = self._qty_steps.get(self.symbol, _FALLBACK_QTY_STEP_PAIR)
        return {
            "min_qty": float(self._min_qtys.get(self.symbol, 0.0)),
            "qty_step": float(step[0]),
            "min_notional": float(self._min_notional),
        }

    def round_quantity(self, quantity: float, symbol: Optional[str] = None) -> float:
        """Floor a quantity toward zero to the symbol's LOT_SIZE step.

        Floor, not round-to-nearest: rounding up asks for more than the caller sized and
        can exceed the available margin. Toward zero rather than down, so a negative is
        not floored to MORE than was asked.

        The tolerance is relative because quantity/step lands just under an integer for
        many exact multiples (0.29/0.01 is 28.999999999999996), and a fixed one stops
        covering that above a ratio of ~1e7 -- where repeated rounding would shave a step.
        """
        step, decimals = self._qty_steps.get(symbol or self.symbol, _FALLBACK_QTY_STEP_PAIR)
        if step <= 0:
            return quantity
        sign = -1.0 if quantity < 0 else 1.0
        ratio = abs(quantity) / step
        return sign * round(math.floor(ratio + max(1e-9, ratio * 1e-12)) * step, decimals)

    def _format_quantity(
        self, quantity: float, symbol: Optional[str] = None, reduce_only: bool = False
    ) -> str:
        """The venue-precision string for an order.

        A string because `str(7.0)` carries a decimal a quantityPrecision-0 symbol does
        not define, which binance rejects with -1111.

        Below the venue minimum, an opening order raises rather than submitting the
        floored value -- which is 0 whenever the request is under one step. A reduceOnly
        order clamps up instead: the venue caps it at the actual position, so it fills and
        closes a sub-step residual that would otherwise be stranded.
        """
        key = symbol or self.symbol
        if key not in self._qty_steps:
            logger.warning(f"No LOT_SIZE cached for {key}; using fallback precision")
        step, decimals = self._qty_steps.get(key, _FALLBACK_QTY_STEP_PAIR)
        rounded = self.round_quantity(quantity, key)
        minimum = max(self._min_qtys.get(key, 0.0), step)

        if rounded < minimum:
            if not reduce_only:
                raise ValueError(
                    f"{key}: {quantity} floors to {rounded} at step {step}, below the "
                    f"minimum {minimum}. Refusing to submit an order that cannot fill."
                )
            rounded = minimum
        return f"{rounded:.{decimals}f}"

    def trade(
        self,
        side: str,
        quantity: float,
        order_type: str = "market",
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
            side: "buy" or "sell" (case-insensitive; upper-cased for the API)
            quantity: Amount to trade in base asset units
            order_type: "market", "limit", "stop_market", "take_profit_market"
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
                "quantity": self._format_quantity(quantity),
            }

            # Add reduce only flag
            if reduce_only:
                order_params["reduceOnly"] = "true"

            # Add price parameters based on order type
            if binance_order_type == "LIMIT":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                order_params["price"] = self._round_price(limit_price)
                order_params["timeInForce"] = time_in_force

            elif binance_order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                if stop_price is None:
                    raise ValueError("stop_price is required for stop orders")
                order_params["stopPrice"] = self._round_price(stop_price)

            # Submit main order
            response = self.client.futures_create_order(**order_params)
            self.last_order_id = response.get("orderId")
            logger.info(f"Order executed: {response}")

        except Exception as e:
            logger.error(f"Error executing main order: {str(e)}")
            return False

        # Main order succeeded — attempt bracket orders separately.
        # Failures here are non-fatal: the position is already open on the
        # exchange, so we return True regardless. bracket_status tracks which
        # legs actually placed so the env can avoid phantom SL/TP state.
        self.bracket_status = {"tp_placed": False, "sl_placed": False}

        if take_profit is not None and not reduce_only:
            try:
                tp_params = {
                    "symbol": self.symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": self._round_price(take_profit),
                    "quantity": self._format_quantity(quantity),
                    "reduceOnly": "true",
                }
                self.client.futures_create_order(**tp_params)
                self.bracket_status["tp_placed"] = True
            except Exception as e:
                logger.warning(f"TP order failed (position opened without TP): {e}")

        if stop_loss is not None and not reduce_only:
            try:
                sl_params = {
                    "symbol": self.symbol,
                    "side": "SELL" if side == "BUY" else "BUY",
                    "type": "STOP_MARKET",
                    "stopPrice": self._round_price(stop_loss),
                    "quantity": self._format_quantity(quantity),
                    "reduceOnly": "true",
                }
                self.client.futures_create_order(**sl_params)
                self.bracket_status["sl_placed"] = True
            except Exception as e:
                logger.warning(f"SL order failed (position opened without SL): {e}")

        return True

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
                    filled_qty=float(order.get("executedQty", 0)),
                    filled_avg_price=float(order.get("avgPrice", 0)),
                    status=order["status"],
                    side=order["side"],
                )

            # Get position status
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                qty = float(pos["positionAmt"])
                if qty != 0:
                    entry_price = float(pos["entryPrice"])
                    mark_price = float(pos["markPrice"])
                    unrealized_pnl = float(pos["unRealizedProfit"])

                    unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(
                        qty, entry_price, mark_price
                    )

                    status["position_status"] = PositionStatus(
                        qty=qty,
                        # `float(...) or` and not `get(...) or`: binance sends notional as a
                        # STRING, and "0" is truthy, so the outer test has to be on the
                        # parsed number. qty != 0 is guaranteed above, so the fallback is a
                        # real notional rather than another 0 (#277).
                        notional_value=float(pos.get("notional") or 0.0) or abs(qty * mark_price),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        mark_price=mark_price,
                        # `in (None, "")` and not `or`: a venue-reported leverage of numeric 0 is
                        # falsy, and `or` would swap it for the config value -- inventing a
                        # liquidation distance from a leverage the venue never confirmed (#277).
                        leverage=float(
                            self.leverage if pos.get("leverage") in (None, "") else pos.get("leverage")
                        ),
                        margin_mode=pos.get("marginType", self.margin_mode.value),
                        liquidation_price=float(pos.get("liquidationPrice", 0)),
                    )
                    break
            else:
                status["position_status"] = None

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            status["position_status"] = POSITION_UNKNOWN

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
            maintenance = account.get("totalMaintMargin")
            return {
                "total_wallet_balance": float(account["totalWalletBalance"]),
                "available_balance": float(account["availableBalance"]),
                "total_unrealized_profit": float(account["totalUnrealizedProfit"]),
                "total_margin_balance": float(account["totalMarginBalance"]),
                "total_maintenance_margin": (
                    float(maintenance) if maintenance not in (None, "") else None
                ),
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

    def close_position(self) -> bool:
        """
        Close the current position.

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
                "quantity": self._format_quantity(qty, reduce_only=True),
                "reduceOnly": "true",
            }

            self.client.futures_create_order(**order_params)
            # The submitted size, not the requested one: this PR exists because the
            # two differ, so a log naming the request is a log naming a number that
            # was never sent.
            logger.info(f"Position closed: {order_params['quantity']} {side}")
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
                            quantity=self._format_quantity(abs(qty), symbol, reduce_only=True),
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
