"""Shared base class for live futures trading environments.

Post-#253, the `_get_observation` bodies of all four futures exchanges (Binance, Bitget,
Bybit, OKX) are functionally identical -- the only differences were two dead locals
(`cash`, `entry_price`) in binance/bitget and cosmetic comments. This class holds the one
shared implementation so a future account_state fix only needs to land once.

Alpaca (spot) is NOT a futures env: it hardcodes leverage=1 and distance_to_liquidation=1.0
and reads cash rather than total_wallet_balance. It keeps its own `_get_observation` and
inherits `TorchTradeLiveEnv` directly.
"""
from typing import Dict
import logging
import math

import torch
from tensordict import TensorDict, TensorDictBase

from torchtrade.envs.core.state import HistoryTracker
from torchtrade.envs.core.live import (
    LiveObservationHalt,
    ObservationFailurePolicy,
    TorchTradeLiveEnv,
    STATUS_UNKNOWN_KEY,
)
from torchtrade.envs.core.state import (
    PositionUnknownError,
    advance_hold_counter,
    position_direction_from_status,
    position_qty_from_status,
)
from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
)
from torchtrade.envs.utils.liquidation import (
    isolated_liquidation_price,
    nearest_liquidation_price,
)


def _normalized_margin_mode(position_status):
    """Normalize concrete adapter labels to ``cross``/``isolated`` or unknown.

    Every adapter exposes ``margin_mode`` since #289; the two-name fallback this used to
    need died with the rename. Unknown labels deliberately remain unknown: when the venue
    also omits a native liquidation price, guessing either route can report a dangerously
    safe distance.
    """
    margin_mode = getattr(position_status, "margin_mode", None)
    value = getattr(margin_mode, "value", margin_mode)
    value = str(value).lower()
    if value in {"0", "cross", "crossed"}:
        return "cross"
    if value in {"1", "isolated"}:
        return "isolated"
    return None

logger = logging.getLogger(__name__)


class TorchTradeFuturesLiveEnv(TorchTradeLiveEnv):
    """Base class for live futures trading environments (Binance, Bitget, Bybit, OKX).

    Holds the single _get_observation (account_state assembly) and _get_portfolio_value
    (total_margin_balance) shared by all four futures exchanges, so an account_state fix
    lands once here instead of in four drifting copies.

    Standard account state (6 elements):
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Subclasses (the plain per-exchange envs) must implement:
    - _execute_fractional_action(): venue sizing/rounding, called by the shared trade gate
    """

    #: The venue's observation and order classes. Set by each exchange's base (#288).
    OBSERVER_CLS = None
    TRADER_CLS = None

    #: Build the trader before the observer. bybit shares the trader's pybit session with
    #: its observer; okx keeps its pre-fold order for the reason below. Preserved per venue
    #: rather than normalised: every trader __init__ WRITES leverage and margin mode to the
    #: venue, so this decides whether that write lands before an observer failure.
    TRADER_FIRST = False

    def _observer_kwargs(self) -> dict:
        """Arguments for `OBSERVER_CLS`. Override to ADD venue-specific ones."""
        return dict(
            symbol=self.config.symbol,
            time_frames=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
            demo=self.config.demo,
        )

    def _trader_kwargs(self, api_key: str, api_secret: str) -> dict:
        """Arguments for `TRADER_CLS`. Override to ADD venue-specific ones.

        """
        return dict(
            symbol=self.config.symbol,
            api_key=api_key,
            api_secret=api_secret,
            demo=self.config.demo,
            leverage=self.config.leverage,
            margin_mode=self.config.margin_mode,
        )

    def _init_trading_clients(self, api_key, api_secret, observer, trader):
        """Four near-identical copies, folded (#288).

        Dependency injection: a supplied instance wins, otherwise build from the class
        attributes.
        """
        def make_trader():
            self.trader = trader if trader is not None else self.TRADER_CLS(
                **self._trader_kwargs(api_key, api_secret)
            )

        def make_observer():
            self.observer = observer if observer is not None else self.OBSERVER_CLS(
                **self._observer_kwargs()
            )

        if self.TRADER_FIRST:
            make_trader()
            make_observer()
        else:
            make_observer()
            make_trader()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """One step for the four PLAIN futures envs (#288).

        The SLTP envs override this via SLTPMixin, which precedes this class in their MRO;
        alpaca does not inherit this class.
        """
        # One PRE-TRADE status read: the trade below reuses this qty and price rather
        # than fetching its own, so an outage cannot open a window between the read and
        # the trade (#295). The post-bar read is a second, separate one.
        _, position_status, current_price, position_size = self._acquire_pre_trade_state()

        self._sync_position_from_exchange(position_status)

        desired_action = self._resolve_action_level(tensordict)
        trade_info = self._execute_trade_if_needed(
            desired_action, current_qty=position_size, current_price=current_price,
        )
        self._record_position_after_trade(desired_action, trade_info)

        self._wait_for_next_timestamp()

        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: no position mark to read. `_acquire_post_bar_state`
        # argues the rest.
        new_price = new_price if new_price is not None else current_price

        return self._record_and_score(
            next_tensordict, price=new_price, action=desired_action,
            portfolio_value=new_portfolio_value, position=new_qty,
        )

    # Set on each venue's <Venue>BaseTorchTradingEnv, NOT on the plain leaf: the SLTP
    # sibling's MRO runs SLTPMixin -> <Venue>Base -> here and never touches the leaf.
    TAKER_FEE: float

    def _read_sizing_balance(self) -> dict:
        """The account read that sizing decisions are made from, under the halt policy.

        The verdict lives INSIDE the closure. `_halting` caches on the way out of a read
        that SUCCEEDED and flags the bar only on one that raised, so a check made one
        frame above it gets neither: the rejected value is served as last-confirmed state
        and the outage counter never advances (#416). `equity == 0.0` is what a venue
        reports while liquidating you.

        Returns the dict, not the float: `_reset` threads the whole thing into
        `_get_observation(snapshot=...)` and reads `available_balance` off it.
        """
        def read_balance():
            info = self.trader.get_account_balance()
            # KeyError deliberately not caught here or in `_halting`: a venue that omits
            # the field is a config error, not an account state.
            raw = info["total_margin_balance"]
            # ONE verdict for every way the balance can be unusable. Coercion failure
            # belongs inside it: `float()` raising ahead of the eviction left the stale
            # balance cached, `_halting` caught the ValueError, and grace sized a live
            # order from it -- this same bug, one line up, for an unparseable value
            # rather than an invalid number.
            #
            # isfinite, not `not (x > 0)`: that catches NaN but passes +inf, and an inf
            # balance sizes an inf target (#277, #347).
            try:
                usable = math.isfinite(float(raw)) and float(raw) > 0
            except (TypeError, ValueError):
                usable = False
            if not usable:
                # Drop the cached balance too -- the same invalidation
                # `_handle_close_action` and `_mark_flat` already do. A venue answering 0,
                # negative, NaN or gibberish is not one whose EARLIER equity we may still
                # size against, on this bar or on a later grace bar (#416).
                self._last_confirmed_read.pop("balance", None)
                raise ValueError(f"cannot size against a portfolio value of {raw}")
            return info

        return self._halting(read_balance, cache_key="balance")

    def _calculate_fractional_position(
        self, action_value: float, current_price: float
    ) -> tuple[float, float, str]:
        """Target position size from a fractional action, for all four futures venues."""
        # Above the balance read on purpose: a flat action must not need the exchange.
        # No caller reaches this today -- all four pre-filter zero.
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        # The 2% buffer is the venue's maintenance-margin headroom.
        effective_balance = float(
            self._read_sizing_balance()["total_margin_balance"]
        ) * 0.98
        return calculate_fractional_position(PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=self.TAKER_FEE,
        ))

    def _resolve_bracket_quantity(self, current_price: float) -> float | None:
        """The quantity an SLTP bracket opens with, for all four futures venues (#288).

        None means the venue would reject the ORDER -- sizing produced nothing, or the
        quantity is below a venue minimum -- and the caller reports a failed trade.

        An unusable ACCOUNT is not that case: it raises inside the `_halting` closure, as
        the plain path does, so the halt policy decides (#416).
        """
        # Every mode goes through the venue-minimum refusal below. The two fixed-size
        # modes returned early and skipped it, which is the same "validated one thing,
        # sent another" shape as the direction-switch bug -- here it was "validated
        # nothing at all" for two of the three modes.
        if self.config.trade_mode == "notional":
            return self._refuse_below_venue_minimums(
                float(self.config.quantity_per_trade) / current_price, current_price
            )
        if self.config.trade_mode == "quantity":
            return self._refuse_below_venue_minimums(
                float(self.config.quantity_per_trade), current_price
            )
        if self.config.trade_mode != "fractional":
            raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")

        balance = float(self._read_sizing_balance()["total_margin_balance"])

        # Reserve what the trader will CHARGE: ReplayOrderExecutor carries its own rate,
        # so reserving the venue constant refused every open for a higher-fee caller (#278).
        raw = getattr(self.trader, "transaction_fee", None)
        try:
            fee = self.TAKER_FEE if raw is None else float(raw)
        except (TypeError, ValueError):
            fee = None
        if fee is None or not 0 <= fee < 1:
            logger.warning(
                f"{self.config.symbol}: trader.transaction_fee={raw!r} is not a usable "
                f"rate; reserving the venue constant {self.TAKER_FEE}. If the trader "
                f"charges more, opens will be refused."
            )
            fee = self.TAKER_FEE
        # Same 0.98 maintenance buffer as `_calculate_fractional_position`.
        # No abs(): `position_fraction` is validated to (0, 1.0], so direction is always
        # +1 and the wrap would be dead -- but a dead `abs()` is not harmless, it launders
        # a corrupted fraction into a plausible positive order.
        #
        # Refusing here rather than letting a bad value reach the venue, because the four
        # formatters disagree about what a negative quantity means: okx FLOORS it and then
        # clamps up to min_qty, turning -244.5 into a 0.001 long; bitget and bybit pass it
        # through and rely on the exchange to reject. One deterministic refusal beats three
        # different downstream behaviours.
        quantity = calculate_fractional_position(PositionCalculationParams(
            balance=balance * 0.98,
            action_value=self.config.position_fraction,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=fee,
        ))[0]
        if not quantity > 0:
            logger.error(
                f"{self.config.symbol}: sizing produced {quantity!r}, refusing rather than "
                f"handing it to the venue formatter"
            )
            return None
        return self._refuse_below_venue_minimums(quantity, current_price)

    def _refuse_below_venue_minimums(
        self, quantity: float, current_price: float
    ) -> float | None:
        """Refuse a quantity the venue will reject, rather than submitting it (#414).

        Refuse, never round UP: rounding up allocates more than the action asked for.
        NOTIONAL only -- the min-quantity floor is okx's `_resolve_bracket_quantity`.
        """
        lot = self.trader.get_lot_size()

        # Ask the executor what it will submit, rather than reproducing its rule here.
        # A shared floor is wrong per venue: binance epsilon-floors, bitget truncates via
        # CCXT, okx clamps UP to min_qty, bybit sends the raw float. Copying one venue's
        # arithmetic accepted 0.0499999999999995 as 5.00 that bitget submits as 4.90.
        sendable = float(self.trader.quantize_quantity(quantity))
        if not sendable > 0:
            logger.warning(
                f"{self.config.symbol}: quantity {quantity} quantizes to {sendable}; "
                f"refusing rather than submitting nothing"
            )
            return None

        notional = sendable * current_price

        # UNKNOWN is not "no floor". A failed metadata fetch used to report 0.0, which
        # skipped the check entirely -- so the guard was off precisely during an outage,
        # which is when the venue is least predictable. None means "not read yet";
        # refuse until it is. okx reports a real 0.0 (its derivatives bind on minSz).
        raw_floor = lot["min_notional"]
        if raw_floor is None:
            logger.warning(
                f"{self.config.symbol}: the venue minimum is not known (metadata fetch "
                f"failed); refusing rather than assuming there is no floor"
            )
            return None

        min_notional = float(raw_floor)
        if min_notional > 0 and notional < min_notional:
            logger.warning(
                f"{self.config.symbol}: notional {notional:.2f} is below the venue "
                f"minimum {min_notional:.2f}; refusing rather than submitting an order "
                f"the exchange will reject"
            )
            return None
        return quantity

    def _finish_futures_init(self) -> None:
        """The tail every futures env ran verbatim after `super().__init__` (#288).

        Ordering is load-bearing and was identical in all four: flatten before the
        baseline, so the baseline is the balance the episode actually starts from.
        """
        self.execute_on = self.config.execute_on

        self.trader.cancel_open_orders()
        if self.config.close_position_on_init:
            self.trader.close_position()

        self._capture_bankruptcy_baseline()
        self._build_observation_specs()

        # `self.position` is already built in TorchTradeLiveEnv.__init__, which runs
        # before this tail. binance and bitget rebuilt it; bybit and okx never did.
        self.history = HistoryTracker()

    def _halting(self, read, cache_key=None):
        """Run a venue read under the halt policy, degrading through a grace period.

        #343's rule is that ANY failure to read account state while a position is open
        must halt -- not only the post-bar read. The pre-trade read at the top of _step
        raised a bare PositionUnknownError that `except LiveObservationHalt` did not
        catch, and no emergency flatten ran even under FLATTEN (#355). The reset read is
        routed through here too as of #295 -- the docstring claimed it already was, and
        it was not.

        #295 adds the grace period. With `max_unknown_status_steps > 0` a transient blip
        no longer kills the process: the env keeps stepping on the last CONFIRMED read,
        publishes `status_unknown=1.0` so the policy can see it is flying blind, and
        truncates once the outage outlasts the budget. At 0 -- the default, and the
        pre-#295 posture -- the first failure still raises.

        The grace period trades on unconfirmed state, which is a real risk and the reason
        it is opt-in. What it must NEVER do is fabricate: with no cached read to fall back
        on (a failure in `_reset`, or the first bar of an episode) it raises regardless of
        the budget, because there is no last-known truth to stand on.
        """
        try:
            value = read()
        # RuntimeError is caught for GRACE ONLY, and re-raised untouched otherwise -- see
        # the `raise` at the bottom. All four adapters wrap a failed balance read in
        # `RuntimeError("Failed to get account balance: ...")`, so without this the grace
        # period never engaged for the failure mode production actually produces; the
        # tests only ever injected PositionUnknownError.
        #
        # It must NOT become halt-and-flatten material (#394, docs/environments/online.md):
        # a read timeout arrives as RuntimeError too, and flattening a live position on a
        # timeout is exactly what that decision refused. Grace does not flatten -- it
        # serves last-known state and flags the bar -- so riding one out is safe where
        # halting on it is not. KeyError stays out: that is a config error.
        except (PositionUnknownError, ValueError, RuntimeError) as error:
            # The BAR is unconfirmed. The counter advances once per step in
            # `_finalize_step_flags`, not here: counting per read site double-counts a bar
            # where both reads fail, and counting only the pre-trade read let a persistent
            # POST-BAR-only outage run forever -- every bar flagged unknown while the
            # healthy pre-trade read reset the counter to zero each time.
            self._status_unknown_this_step = True

            # FLATTEN and a grace period are contradictory postures: FLATTEN means "get
            # me out while I cannot see the account", so riding out the outage would
            # defeat the only thing it is for. Grace applies to HALT only.
            # COPIED on the way out, not on the way in. Copying at the store side was
            # the round-1 mistake: every grace bar returned the SAME physical object,
            # which `_step` then stamps with reward and the done family, so a collector
            # holding it in two rollout slots reads the later bar's values in both.
            # `TensorDict.set()` stores references -- this repo's own recorded rule.
            cached = self._last_confirmed_read.get(cache_key) if cache_key else None
            if isinstance(cached, tuple):
                cached = tuple(
                    v.clone() if isinstance(v, TensorDictBase) else v for v in cached
                )
            grace = (
                self._max_unknown_status_steps > 0
                and cached is not None
                and self.config.observation_failure_policy
                is not ObservationFailurePolicy.FLATTEN
            )
            if grace:
                # No budget check here. Whether the outage has outlasted its budget is a
                # question about the BAR, answered once in `_finalize_step_flags`; asking
                # it per read site is what made the budget mean reads rather than bars.
                # Raising on the spent bar would also reintroduce the process crash #295
                # exists to remove -- the step has to finish so it can truncate.
                if cache_key != "post_bar":   # log once per bar, not per read
                    logger.warning(
                        "%s: venue read failed (%s); running on unconfirmed state "
                        "(bar %d of a %d-bar budget)",
                        self.config.symbol, type(error).__name__,
                        self.consecutive_unknown_status + 1,
                        self._max_unknown_status_steps,
                    )
                return cached

            # Grace did not apply. A RuntimeError leaves untouched -- it was caught only
            # to give the grace path a chance at it, and turning one into a halt (or a
            # FLATTEN close) is the #394 decision inverted.
            if isinstance(error, RuntimeError) and not isinstance(
                error, (PositionUnknownError, LiveObservationHalt)
            ):
                # Clear the flag before aborting. `_finalize_step_flags` is what normally
                # consumes it, and this path never reaches it -- so a caller that catches
                # the timeout and retries would have the next HEALTHY bar report
                # status_unknown=1, and count toward truncation.
                self._status_unknown_this_step = False
                raise

            accepted = flatten_error = None
            if (self.config.observation_failure_policy
                    is ObservationFailurePolicy.FLATTEN):
                try:
                    accepted = bool(self.trader.close_position())
                except Exception as exc:
                    flatten_error = exc
                    logger.exception(
                        "Emergency close_position failed for %s", self.config.symbol
                    )
            raise LiveObservationHalt(
                error, self.config.observation_failure_policy, accepted, flatten_error
            ) from error

        if cache_key:
            self._last_confirmed_read[cache_key] = value
        return value

    def _acquire_pre_trade_state(self):
        """The venue read at the TOP of _step, under the halt policy (#355).

        Wrapping `get_status` alone catches nothing: it RETURNS the POSITION_UNKNOWN
        sentinel rather than raising, and PositionUnknownError comes later, when something
        touches the sentinel's attributes. So the wrapper has to span the reads that touch
        it -- the direction, the size and the mark -- not just the call that fetched it.

        Returns (status, position_status, current_price, position_size).
        """
        def read():
            status = self.trader.get_status()
            position_status = status.get("position_status", None)
            return (
                status,
                position_status,
                self._current_mark_price(position_status),
                position_qty_from_status(position_status),
            )

        return self._halting(read, cache_key="pre_trade")

    def _create_trade_info(self, executed: bool = False, **kwargs) -> Dict:
        """The trade-info dict every futures `_step` returns. Four identical copies (#288).

        The DEFAULTS are the point: `_step` builds this on paths that did not trade, and
        a missing key there is read downstream as an absent fact rather than a default
        one. Four copies meant four chances for one venue's default set to drift from
        the others while every test stayed green.
        """
        return {
            "executed": executed,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
            **kwargs,
        }

    def _handle_close_action(self, current_qty: float) -> Dict:
        """Close the position and report the ORDER SIDE that closed it.

        Four identical copies once binance stopped reporting the literal "CLOSE" (#288).
        `side` is what went to the venue -- `closed_position=True` in the same dict
        already says it was a close, so a literal there lost the only fact this field
        carries.
        """
        if current_qty == 0:
            return self._create_trade_info(executed=False)

        try:
            success = self.trader.close_position()
        except Exception as e:
            logger.error(f"Close position failed for {self.config.symbol}: {e}")
            return self._create_trade_info(executed=False, success=False)

        side = "sell" if current_qty > 0 else "buy"

        if success:
            self.position.current_position = 0
            # A realised close moves equity, so the cached balance is now wrong by the
            # trade's P&L. Only on SUCCESS: a failed close leaves the position, and with
            # it the equity the cache already describes. Without this, a later failed
            # sizing read serves pre-close equity from an arbitrary number of bars ago --
            # the sizing path early-returns before the balance read, so nothing refreshes
            # it (#295).
            self._last_confirmed_read.pop("balance", None)

        return self._create_trade_info(
            executed=True,
            quantity=abs(current_qty),
            side=side,
            success=success,
            closed_position=True,
        )

    def _execute_market_order(self, side: str, quantity: float) -> Dict:
        """Place a market order and report what happened. Four identical copies (#288).

        `side` is lowercase on every venue: binance passed uppercase until this hoist,
        and its executor upper-cases whatever it receives, so the difference only ever
        surfaced as a mixed-case `trade_info["side"]` across exchanges.
        """
        try:
            success = self.trader.trade(
                side=side,
                quantity=quantity,
                order_type="market",
            )
            return self._create_trade_info(
                executed=True,
                quantity=quantity,
                side=side,
                success=success,
            )
        except Exception as e:
            logger.error(f"{side.capitalize()} trade failed for {self.config.symbol}: quantity={quantity}, error={e}")
            return self._create_trade_info(executed=False, success=False)


    def _acquire_post_bar_state(self) -> tuple[float, float, float, TensorDictBase]:
        """Post-bar portfolio value and observation, or halt.

        Raises rather than returning a cached observation: see docs/environments/online.md.

        Observation FIRST, and the order is load-bearing (#278). A tuple evaluates left
        to right, so reading the portfolio value first sampled it before the observer had
        advanced the clock. Live that is merely early; in replay the clock advances only
        inside `ReplayObserver.get_observations()`, so every recorded PV was the PREVIOUS
        bar's equity: the reward at step t belonged to the action at t-1, and an SL/TP
        close during the bar was invisible to the step that caused it. Measured before
        the swap: recorded PV matched the decision bar 8/8 and the next bar 0/8, while
        `account_state` in the SAME step was already at the new bar -- observation and
        reward disagreeing about which bar it was.

        Returns (portfolio_value, mark_price, position_qty, observation), all read AFTER the
        bar. The price is in here because the first version of this fix moved only the
        PV and left `price=` on the pre-trade mark, so a history row carried two
        different bars: `price[t] == close[t-1]` while `portfolio_value[t] == equity[t]`.
        Offline records both at the new bar (`offline/sequential.py`), and replay
        agreeing with offline is the whole point of #278.

        With a position open the mark comes from the snapshot `_get_observation` ALREADY
        read -- no extra round-trip, and the same moment as the `unrealized_pnl_pct` and
        `exposure_pct` in the row. A FLAT account has no position mark, and taking the
        pre-trade price there was wrong for the row that matters most: the EXIT row is
        flat and carries the realized PnL, so 12 of 14 rows ended up with
        `price[t] == close[t-1]` against `portfolio_value[t] == equity[t]`.

        So flat rows do fetch, but NEVER fatally. The first version of this fix called
        `_current_mark_price()` inside the halt wrapper, where a non-positive or missing
        mark raises -- and under FLATTEN that emergency-closes a real position. A price
        that only labels a history row is not worth a market order, so an unavailable
        mark returns None and the caller records the pre-trade price instead.
        """
        def read():
            self._last_observed_mark = None
            self._last_observed_qty = None
            observation = self._get_observation()
            portfolio_value = self._get_portfolio_value()
            mark = self._last_observed_mark
            if mark is None:
                try:
                    mark = self.trader.get_mark_price()
                    if not math.isfinite(mark) or mark <= 0:
                        mark = None
                except Exception:
                    logger.warning(
                        "post-bar mark unavailable for %s; the history row will carry the "
                        "pre-trade price", self.config.symbol,
                    )
                    mark = None
            return portfolio_value, mark, self._last_observed_qty, observation

        return self._halting(read, cache_key="post_bar")

    def _current_mark_price(self, position_status=None) -> float:
        """The bar's mark price, validated before it can size an order (#347).

        An open position's mark and a flat account's fetched mark are the same money-path
        number, so they get the same rule: the sizing paths DIVIDE by it, and it also
        reaches `history.record_step` and the reward. `<= 0` passes NaN and `isfinite`
        passes a negative -- and a negative flips the sign, so a long action opens a short.
        Every venue reads its mark as `float(pos.get("markPrice") or entry_price)`, which
        is 0.0 when both fields are blank.
        """
        if position_status:
            price = position_status.mark_price
        else:
            try:
                price = self.trader.get_mark_price()
            except Exception as error:
                # `_halting` catches (PositionUnknownError, ValueError) and deliberately
                # not RuntimeError, which adapters use for timeouts -- so the wrapped and
                # the raw venue errors both escaped it and the policy was bypassed (#394).
                # Broad on purpose: any tuple fails open the first time an SDK adds a type.
                raise ValueError(
                    f"could not read the mark price for {self.config.symbol}: {error}"
                ) from error
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"venue reported an unusable mark price ({price})")
        return price

    def _get_observation(self, advance_hold: bool = True, *, snapshot=None) -> TensorDictBase:
        """Get the current observation state.

        Args:
            advance_hold: If True (the default, used by `_step()`), ages `hold_counter`
                by one bar using the direction observed in THIS method's single
                `get_status()` call -- holding_time and position_direction in the
                emitted account_state are always derived from the same snapshot.
                `_reset()` passes False so a reset can never itself count a bar.
        """
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        market_data = [
            obs_dict[features_name] for features_name in self.observer.get_keys()
        ]

        # Get account state from trader (single fetch: holding_time and
        # position_direction below MUST come from this same snapshot).
        #
        # `snapshot` is `_reset`'s already-CONFIRMED pair. Re-reading there meant the
        # guarded reads succeeded and were then thrown away, and a failure on the second
        # pair escaped the policy entirely -- FLATTEN could not act on it. Wrapping this
        # whole method instead would be wrong: it also reads the OBSERVER, so a window
        # that can never fill is a config error, and halting it would market-close a
        # position on every episode start (`reset-is-not-halt-wrapped`).
        if snapshot is not None:
            status, balance = snapshot
        else:
            status = self.trader.get_status()
            balance = self.trader.get_account_balance()

        # exposure_pct denominator: use total_margin_balance (equity incl. unrealized PnL),
        # NOT total_wallet_balance. The latter's meaning diverges across exchanges -- Binance's
        # excludes uPnL while Bitget/Bybit/OKX map equity to the same key -- which made Binance's
        # exposure_pct read differently for the same position. total_margin_balance is uniformly
        # equity across all four, so exposure_pct is comparable cross-exchange (and matches the
        # portfolio value _get_portfolio_value returns).
        # Indexed, not .get(..., 0): all four adapters build this key unconditionally, so a
        # missing one means the adapter broke. The default turned that into a silent
        # exposure_pct of 0.0 -- every position reading as flat, forever (#277).
        total_balance = balance["total_margin_balance"]
        # A NaN equity passes the held-position guard below only because that one is
        # keyed on direction; on a FLAT account nothing catches it, and the same value
        # reaches is_bankrupt(), where `nan < threshold * initial` is False -- bankruptcy
        # silently disabled for the rest of the episode (#277).
        if not math.isfinite(total_balance):
            raise ValueError(
                f"venue reported a non-finite equity ({total_balance}); refusing to "
                f"derive an account state or a bankruptcy check from it"
            )
        position_status = status.get("position_status", None)

        # Dust is not a position: gating on `is None` let a 1e-12 residual left behind a
        # close take the position branch and read stale fields off it. An unknown status
        # raises here rather than taking the flat branch below, which would report a held
        # position as flat (invariant #3) -- fail-closed like get_account_balance() above,
        # which already raises rather than inventing a balance. #295 adds a stale-but-marked
        # observation so a blip need not end the episode.
        position_direction = float(position_direction_from_status(position_status))
        if advance_hold:
            advance_hold_counter(self.position, position_direction)
        holding_time = float(self.position.hold_counter)

        if position_direction == 0:
            position_size = 0.0
            self._last_observed_qty = 0.0
            position_value = 0.0
            # No venue call: distance_to_liquidation short-circuits to 1.0 when flat,
            # so this price is never read. Fetching it cost a round-trip per bar and
            # -- once validated -- would raise on a flat account over an unused value.
            current_price = 0.0
            unrealized_pnl_pct = 0.0
            leverage = float(self.config.leverage)
            liquidation_price = 0.0
        else:
            # Every guard below is a comparison, and NaN compares False to all of them --
            # a NaN liquidation price would skip the fallback, reach the arithmetic, and
            # clamp to a distance of 0.0, telling the policy a healthy position is AT
            # liquidation on one garbage tick. Checked once, here, rather than at each
            # comparison (#277).
            # Every venue number this branch reads, not just the ones feeding
            # distance_to_liquidation: a NaN notional_value is worse than a NaN
            # liquidation price, because exposure_pct = nan/equity puts NaN straight into
            # the observation tensor and from there into the policy network.
            for _name, _value in (
                ("qty", position_status.qty),
                ("mark_price", position_status.mark_price),
                ("leverage", position_status.leverage),
                ("liquidation_price", position_status.liquidation_price),
                ("notional_value", position_status.notional_value),
                ("unrealized_pnl_pct", position_status.unrealized_pnl_pct),
                ("entry_price", position_status.entry_price),
            ):
                if not math.isfinite(_value):
                    raise ValueError(
                        f"venue reported a non-finite {_name} ({_value}) for an open "
                        f"position; refusing to derive an account state from it"
                    )
            # Finite is not enough for the mark: 0 and negative both pass the loop above and
            # then short-circuit distance_to_liquidation to "safe". Every venue's mark read
            # is `float(pos.get("markPrice") or entry_price)`, so two blank fields give 0.0.
            if position_status.mark_price <= 0:
                raise ValueError(
                    f"venue reported a non-positive mark price "
                    f"({position_status.mark_price}) for an open position"
                )

            position_size = position_status.qty
            self._last_observed_qty = position_size
            position_value = abs(position_status.notional_value)
            current_price = position_status.mark_price
            self._last_observed_mark = current_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            liquidation_price = position_status.liquidation_price

        # Build 6-element account state
        # Equity gone with the position still on: there is no exposure_pct to report, the
        # ratio being unbounded, and the old `else 0.0` reported the one value that is
        # certainly wrong -- a flat-looking account that is actually holding an underwater
        # position (invariant #3). Fail closed, like the unknown-status path above. An
        # account that is merely empty still reports 0.0 exposure, which is true.
        # `not (x > 0)` rather than `x <= 0`: NaN compares False to everything, so `<= 0`
        # skips the raise and the ternary below then hands back 0.0 -- a held position
        # reading flat, the exact bug this raise exists to prevent.
        # Keyed on position_direction (which goes through the dust rule), NOT on
        # position_value: a venue that omits the notional would zero the second
        # conjunct and skip the raise on a position that is very much held.
        if not (total_balance > 0) and position_direction != 0:
            raise ValueError(
                f"Position worth {position_value} held against non-positive equity "
                f"({total_balance}). The venue is mid-liquidation or reporting "
                f"inconsistently; refusing to report this account as flat."
            )
        exposure_pct = position_value / total_balance if total_balance > 0 else 0.0

        if position_direction == 0:
            distance_to_liquidation = 1.0
        elif liquidation_price <= 0 and leverage == 1:
            # No leverage, nothing to be liquidated by, and the offline env says the same
            # via its has_liquidation gate -- so this must NOT fall through to the
            # arithmetic below, where a short would compute (0 - price)/price = -1 and
            # clamp to 0.0, reporting an unlevered position as AT liquidation.
            #
            # `== 1`, not `<= 1`: leverage of 0, negative or fractional is venue nonsense,
            # and letting it take this branch hands a held position the maximally-safe
            # answer -- #277 one field over. Those fall through to the helper, which
            # raises. Offline refuses the same inputs in __post_init__.
            distance_to_liquidation = 1.0
        else:
            if liquidation_price <= 0:
                # Venues can omit liquidation prices even for real positions: OKX sends
                # liqPx="" for some cross positions, while Bybit can blank liqPrice when
                # the estimate falls outside instrument bounds (and always does so in
                # portfolio margin). Routing therefore depends on the adapter's normalized
                # actual margin mode, never on the mere absence of the price.
                # Defaulting to 1.0 reported a 20x long one move away from liquidation as
                # exactly as safe as a flat spot account (#277).
                #
                margin_mode = _normalized_margin_mode(position_status)
                if margin_mode == "cross":
                    # The aggregate is measured at the current mark. The helper subtracts
                    # the focal position's current maintenance, keeps that focal term
                    # price-sensitive, and treats the remainder as locally constant.
                    # Missing aggregate maintenance must fail closed rather than silently
                    # recreate the single-position assumption from #344.
                    total_account_maintenance = balance["total_maintenance_margin"]
                    if total_account_maintenance is None:
                        raise ValueError(
                            "Cross-margin liquidation price is unavailable because the "
                            "venue did not report total account maintenance"
                        )
                    liquidation_price = nearest_liquidation_price(
                        position_size=position_size,
                        entry_price=position_status.entry_price,
                        mark_price=current_price,
                        equity=total_balance,
                        leverage=leverage,
                        total_account_maintenance=total_account_maintenance,
                    )
                elif margin_mode == "isolated":
                    # Isolated positions retain their position-only fallback; account-wide
                    # maintenance does not determine their liquidation threshold.
                    liquidation_price = isolated_liquidation_price(
                        position_status.entry_price,
                        is_long=position_size > 0,
                        leverage=leverage,
                    )
                else:
                    raise ValueError(
                        "Liquidation price is unavailable because the venue's actual "
                        "margin mode is missing or unsupported"
                    )
            if position_size > 0:
                distance_to_liquidation = (
                    current_price - liquidation_price
                ) / current_price
            else:
                distance_to_liquidation = (
                    liquidation_price - current_price
                ) / current_price
            distance_to_liquidation = max(0.0, distance_to_liquidation)

        account_state = torch.tensor(
            [
                exposure_pct,
                position_direction,
                unrealized_pnl_pct,
                holding_time,
                leverage,
                distance_to_liquidation,
            ],
            dtype=torch.float,
        )

        out_td = TensorDict(
            {
                self.account_state_key: account_state,
                # Always 0.0: `_finalize_step_flags` sets it per step (#295).
                STATUS_UNKNOWN_KEY: torch.zeros(1, dtype=torch.float),
            },
            batch_size=(),
        )
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (includes unrealized PnL)."""
        balance = self.trader.get_account_balance()
        # Indexed, for the same reason as _get_observation above: this is the `current`
        # side of is_bankrupt(), where a default of 0 is instant false bankruptcy.
        equity = balance["total_margin_balance"]
        # Guarded here too, not only in _get_observation: this is a SECOND, independent
        # fetch, and it is the literal argument to _check_termination. The two reads can
        # disagree -- a NaN in this one alone would reach is_bankrupt(), where
        # `nan < threshold * initial` is False, while the observation's read passed.
        if not math.isfinite(equity):
            raise ValueError(
                f"venue reported a non-finite equity ({equity}); refusing to run a "
                f"bankruptcy check against it"
            )
        return equity

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment.

        `observer.reset()` runs before ANY read below: ReplayObserver rewinds the sampler
        AND the simulated executor here, so the balance/position reads that follow must
        see the rewound state (#278).

        A failed `cancel_open_orders` leaves live brackets attached to a position the new
        episode believes is clean; a failed `close_position` leaves real exposure the
        account state will not show. Neither is recoverable here -- the episode has to
        start -- so they warn. All four venues return True when flat, so a clean reset is
        silent.
        """
        self._reset_outage_state()
        self.observer.reset()
        if not self.trader.cancel_open_orders():
            logger.warning(
                "cancel_open_orders failed during reset; proceeding with potentially stale orders"
            )
        self.history.reset()

        if self.config.close_position_on_reset:
            if not self.trader.close_position():
                logger.warning(
                    "close_position failed during reset; proceeding with residual exposure"
                )

        # Under `_halting`, like every other account read (#295). NO cache_key: an
        # episode must not START on stale state -- the grace period exists to ride out an
        # outage mid-episode, not to begin one blind. So this raises LiveObservationHalt
        # rather than a bare PositionUnknownError, which is what `except
        # LiveObservationHalt` and the FLATTEN policy were always documented to cover and
        # what the reset path actually bypassed until now.
        # Through the sizing reader, so the "balance" slot has one writer AND one
        # standard (#416). `_reset_outage_state` has just cleared it, so a failure here
        # still raises -- an episode cannot start on cache.
        balance = self._read_sizing_balance()

        # The CONVERSION is inside the closure, not just the call. No adapter raises from
        # `get_status`: all four RETURN the POSITION_UNKNOWN sentinel, and the error comes
        # from the first attribute touch -- which is `position_direction_from_status`.
        # Wrapping the call alone caught nothing, exactly as `_acquire_pre_trade_state`'s
        # docstring warns twelve lines up.
        def read_status():
            status = self.trader.get_status()
            return status, position_direction_from_status(status.get("position_status"))

        status, direction = self._halting(read_status)
        self.balance = balance.get("available_balance", 0)
        self.position.hold_counter = 0
        self.position.current_position = direction

        # Load-bearing: `_execute_trade_if_needed` returns executed=False when the action
        # matches `current_action_level`, so a position predating the episode would leave a
        # stale level behind and the guard would refuse the trade that closes it (#243).
        self._sync_action_level_after_reset()

        # advance_hold=False: hold_counter was just zeroed above; a reset must never
        # itself count a bar (see advance_hold docstring).
        # Deliberately NOT under `_halting`, unlike the two account reads above. A review
        # flagged the asymmetry -- a NaN equity here raises bare while the same NaN in the
        # post-bar read halts and flattens -- and it is intentional:
        # `test_what_a_short_observation_costs_under_flatten[reset-is-not-halt-wrapped]`
        # pins it. `_get_observation` also reads the OBSERVER, so a config whose window can
        # never fill is a metadata gap, not an outage; halting it would market-close a real
        # position on every episode start under FLATTEN. Cheap-and-loud is the right cost
        # for a config error. The account reads that CAN be an outage are wrapped above.
        # Built from the reads already confirmed above, not a second raw pair.
        return self._get_observation(advance_hold=False, snapshot=(status, balance))

    def _execute_trade_if_needed(
        self, desired_action: float, *, current_qty: float, current_price: float,
    ) -> dict:
        """Skip the venue round-trip when the agent asks for the level it already holds.

        Without this the target is recomputed at each new mark, so one repeated action
        resizes on every price move -- fees the offline env, which holds, never charged the
        policy for. Safe against state drift (rejected orders, partial fills, manual
        intervention): `_sync_position_from_exchange` NaNs `current_action_level` whenever
        the venue's qty disagrees with the target, and NaN never compares equal, so the
        guard releases and the agent can correct (#243).
        """
        if desired_action == self.position.current_action_level:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(
            desired_action, current_qty=current_qty, current_price=current_price,
        )

    def close(self, *, raise_if_closed: bool = True):
        """Cancel open orders. Deliberately does NOT close positions.

        Automated closure on cleanup could liquidate an intended position or interrupt a
        longer-term strategy, so it warns instead; call `env.trader.close_position()`
        first if you want flat. Must never raise: close() runs during teardown, where an
        exception replaces whatever error you were trying to see.
        """
        try:
            status = self.trader.get_status()
            if position_direction_from_status(status.get("position_status")) != 0:
                logger.warning(
                    "Closing environment with open position! "
                    "Call env.trader.close_position() before env.close() if needed."
                )
        except Exception:
            pass

        try:
            self.trader.cancel_open_orders()
        except Exception as e:
            logger.error(f"Failed to cancel open orders on close(): {e}")

        # keyword-only, matching EnvBase.close: TransformedEnv and the collector's
        # shutdown both forward it, and all four venues' bare `def close(self)` raised
        # TypeError on the way out of a rollout. polymarket already carried this fix.
        super().close(raise_if_closed=raise_if_closed)
