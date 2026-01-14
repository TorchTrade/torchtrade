"""Shared SLTP (Stop-Loss/Take-Profit) functionality for live trading environments."""


class SLTPMixin:
    """Mixin providing common SLTP functionality for environments with bracket orders.

    This mixin provides shared methods for environments that support stop-loss
    and take-profit bracket orders (AlpacaSLTPTorchTradingEnv, BinanceFuturesSLTPTorchTradingEnv).

    Required attributes (must be set by the inheriting class):
        - self.current_position: int (0=no position, 1=long, -1=short)
        - self.trader: Object with get_status() method
        - self.active_stop_loss: float (current SL price)
        - self.active_take_profit: float (current TP price)
    """

    def _check_position_closed(self) -> bool:
        """Check if position was closed by stop-loss or take-profit trigger.

        Returns:
            True if position was closed (env has position but exchange doesn't),
            False otherwise
        """
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        # If we had a position but now we don't, it was closed
        if self.current_position != 0 and position_status is None:
            return True
        return False

    def _reset_sltp_state(self) -> None:
        """Reset SLTP-specific state variables.

        Call this in the environment's _reset() method.
        """
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0
