"""Replay order executor for simulated trading with historical data."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

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
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee

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

    def advance_bar(self, ohlc: Dict[str, float]):
        """Advance to new bar and check SL/TP triggers.

        Called by ReplayObserver after each bar advance.

        Args:
            ohlc: Dict with keys "open", "high", "low", "close"
        """
        self.current_price = float(ohlc["close"])

        if self.position_qty == 0 or (self.sl_price == 0 and self.tp_price == 0):
            return

        high = float(ohlc["high"])
        low = float(ohlc["low"])

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
            self._close_at_price(self.sl_price)
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
        price = self.current_price

        # Close existing position first (if any) to avoid margin accounting errors
        if self.position_qty != 0:
            self._close_at_price(price)

        # Calculate margin and fee
        notional = quantity * price
        fee = notional * self.transaction_fee
        margin_required = notional / self.leverage

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
                liquidation_price=0.0,
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
        return True

    def get_open_orders(self):
        """Get open orders (empty in replay)."""
        return []

    def get_lot_size(self) -> Dict[str, float]:
        """Get lot size constraints (permissive in replay)."""
        return {"min_qty": 0.000001, "qty_step": 0.000001}

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
