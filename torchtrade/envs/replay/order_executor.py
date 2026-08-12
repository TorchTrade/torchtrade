"""Replay order executor for simulated trading with historical data."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

from torchtrade.envs.utils.liquidation import (
    DEFAULT_MAINTENANCE_MARGIN_RATE,
    isolated_liquidation_price,
    stop_precedes_liquidation,
)
from torchtrade.envs.utils.sltp_helpers import stop_fill_price

logger = logging.getLogger(__name__)


@dataclass
class PositionStatus:
    qty: float
    notional_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    mark_price: float
    leverage: int
    margin_type: str    # Binance-compatible
    margin_mode: str    # Bybit-compatible
    liquidation_price: float


# The grid replay has effectively used all along -- see round_quantity (#271).
REPLAY_QTY_STEP = 0.001


class ReplayOrderExecutor:
    """Simulated order executor for replaying historical data through live envs.

    Implements the same interface as BinanceFuturesOrderClass/BybitFuturesOrderClass
    so it can be injected into any live SLTP environment.

    Features:
    - Position tracking (long/short with entry price)
    - Balance management (margin, fees, P&L)
    - Bracket order simulation (SL/TP prices)
    - Intrabar SL/TP trigger detection via advance_bar()
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: int = 1,
        transaction_fee: float = 0.0,
        maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee
        # Configurable, because the offline env takes it from config: hardcoding the
        # default here means a backtest tuned off it liquidates at a different price
        # in replay than offline -- the divergence this module exists to prevent.
        if not math.isfinite(maintenance_margin_rate) or not 0 <= maintenance_margin_rate < 1:
            raise ValueError(
                f"maintenance_margin_rate must be in [0, 1), got {maintenance_margin_rate}"
            )
        self.maintenance_margin_rate = maintenance_margin_rate

        # Position state
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.balance = initial_balance

        # Bracket orders
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.bracket_status = {"tp_placed": False, "sl_placed": False}

        # Current market price (updated by advance_bar)
        self.current_price = 0.0

        # Order tracking
        self.last_order_id = None
        self._order_counter = 0

    @property
    def liquidation_price(self) -> float:
        """Where an isolated-margin position loses its margin; 0.0 when flat or unlevered.

        Hardcoded 0.0 before (#269). futures_live_base already falls back to this same
        formula when a venue omits the price, so account_state[5] was not wrong -- the
        real bug is that advance_bar never checked liquidation at all.
        """
        if self.position_qty == 0 or self.leverage <= 1 or self.entry_price <= 0:
            return 0.0
        return isolated_liquidation_price(
            self.entry_price, is_long=self.position_qty > 0, leverage=self.leverage,
            maintenance_margin_rate=self.maintenance_margin_rate,
        )

    def advance_bar(self, ohlc: Dict[str, float]):
        """Advance to new bar and check SL/TP triggers.

        Called by ReplayObserver after each bar advance.

        Args:
            ohlc: Dict with keys "open", "high", "low", "close"
        """
        self.current_price = float(ohlc["close"])

        # Liquidation is checked even with no bracket set: the old guard returned early
        # when sl and tp were both 0, so a leveraged replay position could run to negative
        # equity and then fully recover when price came back (#269).
        if self.position_qty == 0:
            return

        high = float(ohlc["high"])
        low = float(ohlc["low"])
        open_price = float(ohlc["open"])

        # Liquidation outranks the bracket -- unless the stop sits nearer and the bar did
        # not open past liquidation, in which case price crossed the stop on the way (#300,
        # which superseded the #299 ordering). Shared with the offline env rather than
        # restated, because replay answering this differently is the divergence #278 is
        # about; an earlier version of this liquidated unconditionally and left 400 where
        # offline leaves 5000 on the same bar.
        is_long = self.position_qty > 0
        liq = self.liquidation_price
        touched = liq > 0 and ((is_long and low <= liq) or (not is_long and high >= liq))
        stop_first = self.sl_price > 0 and (
            (is_long and low <= self.sl_price) or (not is_long and high >= self.sl_price)
        ) and stop_precedes_liquidation(self.sl_price, liq, open_price, is_long=is_long)

        if touched and not stop_first:
            # Booked AT the liquidation price, as the offline env does: that price is the
            # isolated-margin cap, so filling worse would breach the cap the venue enforces.
            self._close_at_price(liq)
            return

        if self.sl_price == 0 and self.tp_price == 0:
            return

        # Check SL first (pessimistic -- matching offline env)
        sl_triggered = False
        tp_triggered = False

        if self.sl_price > 0:
            if self.position_qty > 0 and low <= self.sl_price:
                sl_triggered = True
            elif self.position_qty < 0 and high >= self.sl_price:
                sl_triggered = True

        if self.tp_price > 0:
            if self.position_qty > 0 and high >= self.tp_price:
                tp_triggered = True
            elif self.position_qty < 0 and low <= self.tp_price:
                tp_triggered = True

        # SL wins over TP (pessimistic)
        if sl_triggered:
            self._close_at_price(
                stop_fill_price(self.sl_price, open_price, is_long=self.position_qty > 0)
            )
        elif tp_triggered:
            self._close_at_price(self.tp_price)

    def _close_at_price(self, price: float):
        """Close position at specified price, updating balance."""
        pnl = self.position_qty * (price - self.entry_price)
        notional = abs(self.position_qty * price)
        fee = notional * self.transaction_fee
        margin_return = abs(self.position_qty * self.entry_price) / self.leverage

        self.balance += pnl - fee + margin_return
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.bracket_status = {"tp_placed": False, "sl_placed": False}

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
        """Execute a simulated trade.

        Args:
            side: "BUY" or "SELL" (case-insensitive)
            quantity: Amount in base asset units
            order_type: Only "market" supported
            take_profit: TP price for bracket order
            stop_loss: SL price for bracket order
            Other args: Accepted for interface compat, ignored

        Returns:
            True (always succeeds in simulation)
        """
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError(f"Unsupported side: {side}")
        if order_type.lower() != "market":
            raise ValueError(f"Unsupported order_type: {order_type}")
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        price = self.current_price
        if price <= 0:
            raise RuntimeError("ReplayOrderExecutor.trade() called before advance_bar() set a valid price")

        # Handle reduce_only: close existing position, don't open new one
        if reduce_only:
            return self.close_position()

        # Close existing position first (if any) to avoid margin accounting errors
        if self.position_qty != 0:
            self._close_at_price(price)

        # Calculate margin and fee
        notional = quantity * price
        fee = notional * self.transaction_fee
        margin_required = notional / self.leverage

        # Check sufficient balance
        if self.balance < margin_required + fee:
            logger.warning(f"Insufficient balance: need={margin_required + fee:.2f} have={self.balance:.2f}")
            return False

        # Deduct margin and fee from balance
        self.balance -= margin_required + fee

        # Set position
        if side_upper == "BUY":
            self.position_qty = quantity
        else:
            self.position_qty = -quantity
        self.entry_price = price

        # Set bracket orders
        self.sl_price = float(stop_loss) if stop_loss is not None else 0.0
        self.tp_price = float(take_profit) if take_profit is not None else 0.0

        self.bracket_status = {
            "tp_placed": take_profit is not None,
            "sl_placed": stop_loss is not None,
        }

        # Track order
        self._order_counter += 1
        self.last_order_id = str(self._order_counter)

        return True

    def get_status(self) -> Dict[str, Optional[PositionStatus]]:
        """Get current position status."""
        if self.position_qty == 0:
            return {"position_status": None}

        unrealized_pnl = self.position_qty * (self.current_price - self.entry_price)
        if self.entry_price > 0:
            if self.position_qty > 0:
                unrealized_pnl_pct = (self.current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = (self.entry_price - self.current_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0

        return {
            "position_status": PositionStatus(
                qty=self.position_qty,
                notional_value=abs(self.position_qty * self.current_price),
                entry_price=self.entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                mark_price=self.current_price,
                leverage=self.leverage,
                margin_type="ISOLATED",
                margin_mode="isolated",
                liquidation_price=self.liquidation_price,
            )
        }

    def get_account_balance(self) -> Dict[str, float]:
        """Get simulated account balance."""
        unrealized_pnl = (
            self.position_qty * (self.current_price - self.entry_price)
            if self.position_qty != 0
            else 0.0
        )
        total = self.balance + unrealized_pnl
        if self.position_qty != 0:
            margin_used = abs(self.position_qty * self.entry_price) / self.leverage
            total += margin_used

        return {
            "total_wallet_balance": total,
            "available_balance": self.balance,
            "total_unrealized_profit": unrealized_pnl,
            "total_margin_balance": total,
        }

    def round_quantity(self, quantity: float, symbol: Optional[str] = None) -> float:
        """Part of the trader interface the live envs size through (#271).

        Replay has no venue to query, so it floors to a fixed default grid. Returning the
        quantity untouched looked cleaner and was wrong twice over: it silently moved
        every existing backtest's numbers, and it put replay on NO grid while the same
        change put live on the venue's real one -- widening the live/replay divergence
        that #278 tracks, in a PR whose point is to narrow it.

        The grid is the one replay already used. Before this, binance's env reached
        `futures_exchange_info()` through a trader that has no `client`, and the resulting
        AttributeError fell through to a default filter set whose stepSize is 0.001. That
        was an accident of the error path, but it is what every recorded backtest was run
        against, so it is preserved deliberately rather than changed silently.
        """
        step = REPLAY_QTY_STEP
        # Same tolerance reasoning as the live executor: a bare floor shaves a whole step
        # off exact multiples that land just under an integer in binary.
        ratio = abs(quantity) / step
        sign = -1.0 if quantity < 0 else 1.0
        return sign * round(math.floor(ratio + max(1e-9, ratio * 1e-12)) * step, 3)

    def get_mark_price(self) -> float:
        """Get current mark price (latest close)."""
        return self.current_price

    def close_position(self, position_side: str = "BOTH") -> bool:
        """Close the current position at current price."""
        if self.position_qty == 0:
            return True
        self._close_at_price(self.current_price)
        return True

    def cancel_open_orders(self) -> bool:
        """Cancel active bracket orders."""
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.bracket_status = {"tp_placed": False, "sl_placed": False}
        return True

    def get_open_orders(self):
        """Get open orders (empty in replay)."""
        return []

    def get_lot_size(self) -> Dict[str, float]:
        """Get lot size constraints (permissive in replay)."""
        return {"min_qty": 0.000001, "qty_step": 0.000001}

    def _round_amount(self, amount: float) -> float:
        """Floor a quantity to the replay lot step (never rounds up — margin-safe,
        matching the live truncation path)."""
        step = self.get_lot_size()["qty_step"]
        return (amount // step) * step

    def reset(self):
        """Reset executor to initial state."""
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.current_price = 0.0
        self.bracket_status = {"tp_placed": False, "sl_placed": False}
        self.last_order_id = None
        self._order_counter = 0
