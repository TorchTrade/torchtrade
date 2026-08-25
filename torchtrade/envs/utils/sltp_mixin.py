"""Shared SLTP (Stop-Loss/Take-Profit) functionality for live trading environments."""

import logging

import torch
from tensordict import TensorDictBase

from torchtrade.envs.core.state import position_direction_from_status

logger = logging.getLogger(__name__)


class SLTPMixin:
    """Shared behaviour for every env that places stop-loss / take-profit brackets.

    As of #288 this owns the STEP ITSELF, not just helpers: `_step` and `_reset` were
    four copies, 100% identical within each venue pair. All five SLTP envs inherit them
    -- alpaca included, which is why deleting its own `_reset` mattered: with both in the
    MRO, `_reset_sltp_state` ran twice per reset.

    The one venue-specific piece is `_dispatch_sltp_trade`. bybit and okx forward the
    mark `_step` already acquired under the halt policy; binance and bitget price their
    brackets off a candle close and take the default. That split is deliberate and is a
    #409 decision, not an accident to unify here.

    Required attributes (set by the inheriting class):
        - self.position.current_position: int (0 flat, 1 long, -1 short)
        - self.trader: object with get_status()
        - self.active_stop_loss / self.active_take_profit: float
        - self.action_map: dense dict of index -> (side, sl, tp)
    """

    # The direction each SLTP side targets. Also used by the duplicate-action check
    # further down each env, which is why it lives in one place.
    SIDE_DIRECTION = {"long": 1, "short": -1}

    def _resolve_action_tuple(self, tensordict):
        """The validated action index, resolved against this env's `action_map`.

        The dict-side twin of `_resolve_action_level`, so `len(self.action_map)` is not
        restated at five call sites where a copy-paste could pass the wrong container.
        """
        return self.action_map[
            self._resolve_action_index(tensordict, len(self.action_map))
        ]

    def _record_sltp_position(self, side) -> None:
        """The position the ACTION targets, never the order side (#276)."""
        self.position.current_position = self.SIDE_DIRECTION.get(side, 0)

    def _dispatch_sltp_trade(self, action_tuple, current_price: float):
        """Hand the action to the venue's executor. The ONE thing `_step` varies by venue.

        Default: the venue prices its own bracket off a candle close and does not want the
        mark (binance, bitget). bybit and okx take the threaded price -- see #295, where
        re-reading it inside the trade path bypassed the halt policy.
        """
        return self._execute_trade_if_needed(action_tuple)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Four byte-identical copies (#288)."""
        result = super()._reset(tensordict, **kwargs)
        self._reset_sltp_state()
        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """One step. Four copies, 100% identical within each venue pair and 88% across --
        the 12% was solely the dispatch call, which `_dispatch_sltp_trade` now owns (#288).

        This is the file where #295 kept finding a fix applied to some copies and not
        others: the unguarded balance read, the mark re-read, the close that left the
        cache stale. One copy is the point.
        """
        # `status` and `position_size` were unpacked and unused in all four copies.
        # One canonical copy is the place to stop carrying that.
        _, position_status, current_price, _ = self._acquire_pre_trade_state()

        # Source of truth: detects SL/TP closures AND state drift from failed brackets.
        position_closed = self._sync_position_from_exchange(position_status)

        action_tuple = self._resolve_action_tuple(tensordict)

        # `trade_info["position_closed"]` was set here in all four copies and read by
        # nothing -- `_sync_position_from_exchange` already acts on its own return value.
        # Asserting a dead field is asserting metadata, so the field and the test that
        # pinned it both go.
        trade_info = self._dispatch_sltp_trade(action_tuple, current_price)

        # Eagerly update position from the trade result so the rest of this step sees the
        # new state without waiting for the next sync cycle.
        if trade_info["executed"] and trade_info.get("success") is not False:
            self._record_sltp_position(action_tuple[0])

        self._wait_for_next_timestamp()

        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: there is no position mark to read, and fetching
        # one would add a round-trip that can halt the episode. The pre-trade price is the
        # honest fallback -- flat rows carry no PnL anyway.
        new_price = new_price if new_price is not None else current_price

        side, _, _ = action_tuple
        action_value = 1.0 if side == "long" else (-1.0 if side == "short" else 0.0)

        # History FIRST: the reward function reads it.
        self.history.record_step(
            price=new_price,
            action=action_value,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=new_qty,
        )
        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        done = self._check_termination(new_portfolio_value)
        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        self._finalize_step_flags(next_tensordict, terminated=done)

        return next_tensordict

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
            # The exchange closed it, so the realised P&L has already moved equity and the
            # cached sizing balance is stale by exactly that amount. This closure path is
            # the one the ENV never asked for -- a bracket firing, or a manual close -- so
            # it has no `close_position()` call for the close-site guard to find (#295).
            getattr(self, "_last_confirmed_read", {}).pop("balance", None)

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
