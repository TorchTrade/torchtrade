"""Shared base class for live futures trading environments.

Post-#253, the `_get_observation` bodies of all four futures exchanges (Binance, Bitget,
Bybit, OKX) are functionally identical -- the only differences were two dead locals
(`cash`, `entry_price`) in binance/bitget and cosmetic comments. This class holds the one
shared implementation so a future account_state fix only needs to land once.

Alpaca (spot) is NOT a futures env: it hardcodes leverage=1 and distance_to_liquidation=1.0
and reads cash rather than total_wallet_balance. It keeps its own `_get_observation` and
inherits `TorchTradeLiveEnv` directly.
"""
from abc import abstractmethod
from typing import Dict
import logging
import math

import torch
from tensordict import TensorDict, TensorDictBase

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

    Subclasses (per-exchange base envs) must still implement:
    - _init_trading_clients(): provider-specific client construction
    - _execute_trade_if_needed(): trade execution, whose action space differs by env
    """

    def _halting(self, read, cache_key=None):
        """Run a venue read under the halt policy, degrading through a grace period.

        #343's rule is that ANY failure to read account state while a position is open
        must halt -- not only the post-bar read. The pre-trade read at the top of _step
        and the initial read in _reset bypassed this, so a POSITION_UNKNOWN between bars
        raised a bare PositionUnknownError that `except LiveObservationHalt` did not
        catch, and no emergency flatten ran even under FLATTEN (#355).

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
        # NOT RuntimeError: adapters wrap timeouts in it, and KeyError is a config
        # error. See docs/environments/online.md.
        except (PositionUnknownError, ValueError) as error:
            budget = self._max_unknown_status_steps
            cached = self._last_confirmed_read.get(cache_key) if cache_key else None
            # Only the pre-trade read advances the counter. It runs exactly once per
            # `_step`, so the budget reads as BARS. Counting every read site would make
            # `max_unknown_status_steps=3` truncate after two bars, since the post-bar
            # read fails in the same outage -- a budget that silently means something
            # other than what it says.
            if cache_key != "post_bar":
                self.consecutive_unknown_status += 1
            self._status_unknown_this_step = True

            # FLATTEN and a grace period are contradictory postures: FLATTEN means "get
            # me out while I cannot see the account", so riding out the outage would
            # defeat the only thing it is for. Grace applies to HALT only.
            grace = (
                budget > 0
                and cached is not None
                and self.config.observation_failure_policy
                is not ObservationFailurePolicy.FLATTEN
            )
            if grace:
                if self.consecutive_unknown_status < budget:
                    if cache_key != "post_bar":   # the counting site logs, once per bar
                        logger.warning(
                                "%s: venue read failed (%s); running on unconfirmed "
                            "state, %d of %d grace steps used",
                            self.config.symbol, type(error).__name__,
                            self.consecutive_unknown_status, budget,
                        )
                    return cached
                # Budget spent. Still attempt the protective close under FLATTEN, then
                # let the step finish on cached state so `_set_done_family` can truncate
                # -- an exception here would be the process crash #295 exists to remove.
                self._flatten_if_policy_says_so()
                if cache_key != "post_bar":
                    logger.error(
                        "%s: venue unreadable for %d consecutive bars; truncating",
                        self.config.symbol, self.consecutive_unknown_status,
                    )
                return cached

            accepted, flatten_error = self._flatten_if_policy_says_so()
            raise LiveObservationHalt(
                error, self.config.observation_failure_policy, accepted, flatten_error
            ) from error

        if cache_key != "post_bar":
            self.consecutive_unknown_status = 0
        if cache_key:
            self._last_confirmed_read[cache_key] = value
        return value

    def _flatten_if_policy_says_so(self):
        """Attempt the emergency close when the policy is FLATTEN. Returns (accepted, error).

        `accepted` records that the venue accepted a close REQUEST, not that the position
        is gone -- the distinction LiveObservationHalt already documents.
        """
        if self.config.observation_failure_policy is not ObservationFailurePolicy.FLATTEN:
            return None, None
        try:
            return bool(self.trader.close_position()), None
        except Exception as exc:
            logger.exception("Emergency close_position failed for %s", self.config.symbol)
            return None, exc

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
                        "pre-trade price", self.symbol,
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

    def _get_observation(self, advance_hold: bool = True) -> TensorDictBase:
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
        # position_direction below MUST come from this same snapshot)
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
                # Declared for all ten envs so the observation contract does not fork by
                # venue. Alpaca has no observation-failure policy, so its counter never
                # advances and this is always 0.0 there -- see _max_unknown_status_steps.
                STATUS_UNKNOWN_KEY: torch.tensor(
                    [float(self.consecutive_unknown_status > 0)], dtype=torch.float
                ),
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

    def _get_current_position_quantity(self) -> float:
        """The signed size the exchange holds, dust read as flat.

        One copy: binance, bitget and bybit each carried a byte-identical
        `position.qty if position is not None else 0.0`, which returned the residual
        rather than 0 and let `abs(current_qty) > 0` fire on a flat account (#283).
        """
        return position_qty_from_status(self.trader.get_status().get("position_status"))

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

        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)

        status = self.trader.get_status()
        self.position.hold_counter = 0
        self.position.current_position = position_direction_from_status(
            status.get("position_status")
        )

        # Load-bearing for binance and bitget, whose _execute_trade_if_needed compares
        # `desired_action == self.position.current_action_level` and returns executed=False
        # on a match: without this, a position that predates the episode leaves a stale
        # level behind which the guard refuses the very trade that would close it (#243).
        # Inert for bybit and okx, which recompute qty live and never read the field.
        self._sync_action_level_after_reset()

        # advance_hold=False: hold_counter was just zeroed above; a reset must never
        # itself count a bar (see advance_hold docstring).
        return self._get_observation(advance_hold=False)

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """Execute trade if position change is needed. Action format varies by subclass."""
        raise NotImplementedError

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
