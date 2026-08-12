"""Shared SLTP (Stop-Loss/Take-Profit) functionality for live trading environments."""

import logging

from torchtrade.envs.core.state import position_direction_from_status

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

    # The direction each SLTP side targets. Also used by the duplicate-action check
    # further down each env, which is why it lives in one place.
    SIDE_DIRECTION = {"long": 1, "short": -1}

    def _record_sltp_position(self, side) -> None:
        """The position the ACTION targets, never the order side (#276).

        binance and bitget had their close branch behind an elif that always matched
        first, so an SLTP close could never be recorded at all.
        """
        self.position.current_position = self.SIDE_DIRECTION.get(side, 0)

    def _sync_position_from_exchange(self, position_status) -> bool:
        """Sync internal position state from exchange and detect SL/TP closures.

        Must be called at the start of each _step() BEFORE the duplicate-action
        guard runs. This ensures self.position.current_position always reflects
        the exchange's actual state, preventing position stacking when bracket
        orders fail but the main order succeeds.

        An unknown status raises rather than syncing: it is indistinguishable here from an
        SL/TP fill, and would clear brackets the exchange still holds. In practice _step
        raises earlier, on its own status read.

        Args:
            position_status: Position status from trader.get_status(), None if the
                exchange confirmed no position, or POSITION_UNKNOWN if it did not
                answer -- which raises rather than syncing.

        Returns:
            True if a position was closed since the last step (SL/TP trigger
            or external closure), False otherwise.
        """
        prev_position = self.position.current_position
        self.position.current_position = position_direction_from_status(position_status)

        # The position we were counting is gone, or was never ours. Its age must not be
        # inherited by whatever is there now -- including a re-entry made in THIS same step.
        if self.position.current_position != prev_position:
            self.position.hold_counter = 0

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
