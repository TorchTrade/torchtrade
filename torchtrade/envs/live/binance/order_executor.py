import logging
from dataclasses import dataclass
from enum import Enum
import math
from decimal import Decimal
from typing import Dict, List, Optional, Union
import warnings
import os
from dotenv import load_dotenv

from torchtrade.envs.core.common import TradeMode
from torchtrade.envs.core.common_types import MarginType, OrderStatus
from torchtrade.envs.core.state import POSITION_UNKNOWN

load_dotenv()

logger = logging.getLogger(__name__)


_FALLBACK_QTY_STEP = 0.001
_FALLBACK_QTY_STEP_PAIR = (_FALLBACK_QTY_STEP, 3)


def _step_and_decimals(step_str: str):
    """(step, decimals) from a LOT_SIZE stepSize string such as "0.001" or "1".

    Through Decimal rather than string surgery: `"1E-8".split('.')` yields no fractional
    part, so the naive version reported 0 decimals and the final rounding then annihilated
    the quantity. Binance sends fixed-point today, but okx's _format_size carries a comment
    about being bitten by exactly this, so it is not a hypothetical.
    """
    d = Decimal(step_str).normalize()
    return float(d), max(0, -d.as_tuple().exponent)


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
        trade_mode: TradeMode = "quantity",
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
            trade_mode: "quantity" for unit-based orders
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

        self._tick_size: Optional[float] = None
        # symbol -> (step, decimals). Every symbol, not just this one: see
        # _fetch_symbol_filters -- close_all_positions closes whatever the account holds.
        self._qty_steps: Dict[str, tuple] = {}
        self._min_qtys: Dict[str, float] = {}
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

    def _fetch_symbol_filters(self):
        """Cache tick size and LOT_SIZE step from Binance exchange info.

        Only PRICE_FILTER was read before, so every order quantity went out as
        `round(quantity, 3)` and any symbol whose step is not exactly three decimals got a
        silently wrong size -- a rejected order, or a mis-sized position (#271). Bitget,
        bybit and okx all fetch a real step from the venue; binance was the one that did
        not. This is not full parity with them: they also cache minQty, and until this
        change binance had neither.

        Steps are kept for EVERY symbol, not just this one, because close_all_positions
        closes whatever the account holds -- rounding another symbol's quantity with this
        symbol's step would be its own bug.
        """
        try:
            info = self.client.futures_exchange_info()
        except Exception as e:
            # Transient: fall back and warn. A symbol the venue does not list is a config
            # error and raises below; a fetch that did not happen is not.
            logger.warning(f"Could not fetch symbol filters for {self.symbol}: {e}")
            return

        for s in info['symbols']:
            # Per symbol, so one malformed entry costs one step rather than the whole
            # cache. Sweeping every symbol inside a single try meant an unrelated delisted
            # entry could discard the tick size too -- and whether it did depended on
            # whether it appeared before or after this symbol in the payload.
            try:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        self._qty_steps[s['symbol']] = _step_and_decimals(f['stepSize'])
                        self._min_qtys[s['symbol']] = float(f['minQty'])
                    elif f['filterType'] == 'PRICE_FILTER' and s['symbol'] == self.symbol:
                        tick_str = f['tickSize']
                        self._tick_size = float(tick_str)
                        # Derive decimal places from tick string for clean formatting
                        if '.' in tick_str:
                            decimal_part = tick_str.rstrip('0').split('.')[1]
                            self._tick_decimals = len(decimal_part) if decimal_part else 0
            except Exception as e:
                # Names the offending symbol: the first version logged only self.symbol,
                # which is never the one at fault and sends the reader the wrong way.
                logger.warning(f"Skipping malformed filters for {s.get('symbol', '?')}: {e}")

        if self._tick_size is None:
            logger.warning(f"No PRICE_FILTER found for {self.symbol}, prices will not be rounded")
        else:
            logger.info(f"Tick size for {self.symbol}: {self._tick_size} ({self._tick_decimals} decimals)")

        # Fail closed: the venue answered and does not list this symbol, so every order
        # would go out on the fallback precision -- which is bit-identical to the bug this
        # fixes, and binance is the one exchange with no min_qty net downstream to catch
        # the result. A futures executor pointed at a spot-only symbol used to log two
        # warnings and then trade happily against a symbol that does not exist.
        if self.symbol not in self._qty_steps:
            raise ValueError(
                f"binance returned no LOT_SIZE for {self.symbol}: it is not a futures "
                f"symbol on this venue, or the payload changed shape. Refusing to size "
                f"orders on fallback precision."
            )

    def _round_price(self, price: float) -> float:
        """Round a price to the nearest tick size."""
        if self._tick_size is not None:
            rounded = round(price / self._tick_size) * self._tick_size
            return round(rounded, self._tick_decimals)
        return price

    def _round_quantity(self, quantity: float, symbol: Optional[str] = None) -> float:
        """Floor a quantity to the symbol's LOT_SIZE step.

        Floor rather than round to nearest: rounding up asks for more than the caller
        sized, which can exceed the margin available and get the whole order rejected.
        (bybit floors too, but in `bybit/env.py`'s sizing rather than in its executor --
        so this is the same reasoning, not the same code.)
        """
        step, decimals = self._qty_steps.get(symbol or self.symbol, _FALLBACK_QTY_STEP_PAIR)
        if step <= 0:
            return quantity
        # The epsilon is not decoration: quantity/step lands just under an integer for
        # plenty of exact multiples in binary (0.29/0.01 is 28.999999999999996), and a bare
        # floor would then shave a whole step off a perfectly valid size.
        return round(math.floor(quantity / step + 1e-9) * step, decimals)

    def _format_quantity(self, quantity: float, symbol: Optional[str] = None) -> str:
        """The venue-precision string for an order, or raise if the size rounds away.

        A string, not a float: `str(7.0)` carries a decimal a symbol with
        quantityPrecision 0 does not define, which is the documented shape of binance's
        `-1111 Precision is over the maximum defined for this asset`. okx formats the same
        way for the same reason.

        Raises below the venue minimum rather than submitting the floored value. That
        value is 0 whenever the request is under one step -- reachable on any symbol whose
        step is coarser than the configured size, and routinely in the fractional and
        notional modes when a small balance meets a six-figure price. The old
        `round(q, 3)` did it too; the difference is that this says so instead of letting
        the venue reject an order whose own error message points at the pre-rounding size.
        """
        key = symbol or self.symbol
        step, decimals = self._qty_steps.get(key, _FALLBACK_QTY_STEP_PAIR)
        rounded = self._round_quantity(quantity, key)
        minimum = max(self._min_qtys.get(key, 0.0), step)
        if rounded < minimum:
            raise ValueError(
                f"{key}: a quantity of {quantity} floors to {rounded} at the venue step "
                f"{step}, below the minimum {minimum}. Refusing to submit an order that "
                f"cannot fill."
            )
        return f"{rounded:.{decimals}f}"

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
                "quantity": self._format_quantity(quantity),
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
                if position_side != "BOTH":
                    tp_params["positionSide"] = position_side
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
                if position_side != "BOTH":
                    sl_params["positionSide"] = position_side
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
                "quantity": self._format_quantity(qty),
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
                            quantity=self._format_quantity(abs(qty), symbol),
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
        trade_mode="quantity",
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
