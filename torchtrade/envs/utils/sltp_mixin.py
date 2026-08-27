"""Shared SLTP (Stop-Loss/Take-Profit) functionality for live trading environments."""

import logging
import math
from typing import Dict, Optional, Tuple

from tensordict import TensorDictBase

from torchtrade.envs.core.state import position_direction_from_status
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

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

    Required of the inheriting class -- the full list, because owning `_step` means this
    mixin now depends on the whole live-env surface, not just SLTP state:
        state   - position.current_position, active_stop_loss, active_take_profit,
                  action_map (dense index -> (side, sl, tp)), history, reward_function
        venue   - trader.get_status()
        step    - _acquire_pre_trade_state(), _acquire_post_bar_state(),
                  _wait_for_next_timestamp(), _check_termination(),
                  _finalize_step_flags()
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
        self._sync_position_from_exchange(position_status)

        action_tuple = self._resolve_action_tuple(tensordict)

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

        return self._record_and_score(
            next_tensordict, price=new_price, action=action_value,
            portfolio_value=new_portfolio_value, position=new_qty,
        )

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

    def _close_action(self, trade_info: dict) -> dict:
        """Flatten the position. One copy for all four futures SLTP venues (#288).

        Was three byte-identical copies -- the mixin's, bybit's and okx's -- which is one
        more than the two this PR set out to fold. It had already cost a round: the
        `success=False` contract below first landed on the mixin alone, so binance and
        bitget disagreed with bybit and okx about what a refused close reports.

        Sits ABOVE the caller's price read on purpose. A policy must still be able to
        flatten under a degraded feed, which a stale-bar ValueError would otherwise block.
        """
        if self.position.current_position == 0:
            return trade_info
        try:
            success = self.trader.close_position()
        except Exception as e:
            logger.error(f"Close position failed for {self.config.symbol}: {e}")
            trade_info["success"] = False
            return trade_info
        if success:
            # A realised close moves equity; the cached balance is now wrong by the
            # trade's P&L. SUCCESS only -- a failed close leaves the position (#295).
            self._last_confirmed_read.pop("balance", None)
            close_side = "sell" if self.position.current_position > 0 else "buy"
            self.position.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0
            trade_info.update({
                "executed": True, "side": close_side,
                "success": True, "closed_position": True,
            })
        else:
            # Otherwise a refused close returns success=None -- what HOLD returns -- and
            # the refusal is invisible to the caller.
            trade_info["success"] = False
        return trade_info

    def _execute_trade_if_needed(
        self, action_tuple: Tuple[Optional[str], Optional[float], Optional[float]]
    ) -> Dict:
        """Place the bracket for the venues that price it off their own candle.

        The four SLTP venues split by SIGNATURE, not by exchange: binance and bitget price
        the bracket off the observer's own candle, while bybit and okx require the mark
        threaded in by the caller (`*, current_price`) -- see #295, where re-reading it
        inside the trade path bypassed the halt policy. Those two override this.

        Args:
            action_tuple: (side, stop_loss_pct, take_profit_pct); (None, None, None) is HOLD.

        Returns:
            Dict with trade execution info
        """
        trade_info = {
            "executed": False,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }

        side, stop_loss_pct, take_profit_pct = action_tuple

        # HOLD action - do nothing
        if side is None:
            return trade_info

        # Position locking: ignore all actions while in position
        if self.config.lock_position_until_sltp and self.position.current_position != 0:
            return trade_info

        if side == "close":
            return self._close_action(trade_info)

        if side in self.SIDE_DIRECTION and self.position.current_position == self.SIDE_DIRECTION[side]:
            return trade_info

        # Read AND verdict under `_halting` (#295): `get_observations` raises on a short
        # window or a stale bar, and outside the policy that escapes as a bare ValueError.
        # bybit/okx take a threaded mark; these two price off the candle close on purpose.
        def read_close():
            obs = self.observer.get_observations(return_base_ohlc=True)
            current_price = float(obs["base_features"][-1, 3])
            # This price divides the notional sizing AND prices both brackets in every
            # mode, including the "quantity" default which checked nothing. dropna() does
            # not clear a candle close of inf (#347). The name is load-bearing:
            # test_sltp_sizing_rejects_a_non_finite_price_or_balance greps for it.
            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError(
                    f"unusable close price ({current_price}) for {self.config.symbol}"
                )
            return current_price

        # cache_key is load-bearing, not decoration: without it `cached` is None, grace
        # cannot apply, and this still raises -- it just raises a nicer type. The claimed
        # behaviour is "serve the last CONFIRMED close and flag the bar", which needs a
        # slot to serve from. Its own slot, because it is a candle close, not the mark.
        current_price = self._halting(read_close, cache_key="candle_close")

        quantity = self._resolve_bracket_quantity(current_price)
        if quantity is None:
            trade_info["success"] = False
            return trade_info

        # Priced BEFORE anything reaches the venue -- including the flatten below, not
        # just `trade()`. `calculate_bracket_prices` rejects any side that is neither
        # long nor short, so a bad side fails closed here rather than closing a position
        # and then raising on the way to reopening it.
        trade_side = "buy" if side == "long" else "sell"
        stop_loss_price, take_profit_price = calculate_bracket_prices(
            side, current_price, stop_loss_pct, take_profit_pct
        )

        # Switching directions: flatten first. No per-side test, because a non-zero
        # position here IS the opposite one -- `side` is long or short by now, and the
        # duplicate guard above already returned on a same-direction position.
        if self.position.current_position != 0:
            try:
                close_success = self.trader.close_position()
            except Exception as e:
                logger.error(f"Close position failed for {self.config.symbol}: {e}")
                return trade_info
            if not close_success:
                return trade_info
            self._last_confirmed_read.pop("balance", None)
            self.position.current_position = 0

        try:
            success = self.trader.trade(
                side=trade_side,
                quantity=quantity,
                order_type="market",
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
            )

            if success:
                # Only record SL/TP levels that actually placed on-exchange
                bs = getattr(self.trader, "bracket_status",
                             {"tp_placed": True, "sl_placed": True})
                self.active_stop_loss = stop_loss_price if bs["sl_placed"] else 0.0
                self.active_take_profit = take_profit_price if bs["tp_placed"] else 0.0

            trade_info.update({
                "executed": True,
                "quantity": quantity,
                "side": trade_side,
                "success": success,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
            })
        except Exception as e:
            logger.error(
                f"{side.capitalize()} trade failed for {self.config.symbol}: "
                f"quantity={quantity}, SL={stop_loss_price:.2f}, "
                f"TP={take_profit_price:.2f}, error={e}"
            )
            trade_info["success"] = False

        return trade_info
