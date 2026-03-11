"""Shared SLTP (Stop-Loss/Take-Profit) functionality for live trading environments."""

import logging

logger = logging.getLogger(__name__)


class SLTPMixin:
    """Mixin providing common SLTP functionality for environments with bracket orders.

    This mixin provides shared methods for environments that support stop-loss
    and take-profit bracket orders across all exchange environments.

    Required attributes (must be set by the inheriting class):
        - self.position.current_position: int (0=no position, 1=long, -1=short)
        - self.trader: Object with get_status() method
        - self.active_stop_loss: float (current SL price)
        - self.active_take_profit: float (current TP price)
    """

    def _sync_position_from_exchange(self, position_status) -> bool:
        """Sync internal position state from exchange and detect SL/TP closures.

        Must be called at the start of each _step() BEFORE the duplicate-action
        guard runs. This ensures self.position.current_position always reflects
        the exchange's actual state, preventing position stacking when bracket
        orders fail but the main order succeeds.

        Args:
            position_status: Position status from trader.get_status(), or None
                if no position exists on the exchange.

        Returns:
            True if a position was closed since the last step (SL/TP trigger
            or external closure), False otherwise.
        """
        prev_position = self.position.current_position

        if position_status is not None and position_status.qty != 0:
            if position_status.qty > 0:
                self.position.current_position = 1
            else:
                self.position.current_position = -1
        else:
            self.position.current_position = 0

        # Detect position closure (had position, now don't)
        position_closed = (prev_position != 0 and self.position.current_position == 0)
        if position_closed:
            logger.info("Position closed by SL/TP or external action")
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        # Detect direction flip (e.g., long→short via external action).
        # Reset SL/TP since the old bracket levels are stale for the new direction.
        if (prev_position != 0 and self.position.current_position != 0
                and prev_position != self.position.current_position):
            logger.warning(
                f"Position direction changed unexpectedly: {prev_position} -> "
                f"{self.position.current_position}"
            )
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        return position_closed

    def _reset_sltp_state(self) -> None:
        """Reset SLTP-specific state variables.

        Call this in the environment's _reset() method.
        """
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0
