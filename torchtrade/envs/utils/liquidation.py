"""The isolated-margin liquidation price, in one place.

The offline envs have always computed this; the live envs never did -- they read the
venue's own `liqPrice` and, when the venue omitted it, reported the safest possible
answer (#277). This module holds the scalar rule so the live fallback and the offline
env cannot answer the same question differently.

`vectorized_sequential.py` keeps a tensorised copy of the arithmetic, but the equivalence
tests pin it to the scalar env, so it cannot drift unnoticed. The live path had no such
pin, which is why it shares this.
"""

import math

# Both offline configs default their maintenance_margin_rate to this, so a policy meets
# the same liquidation geometry offline and live. The live side has no config knob, so an
# offline env configured off the default has no matching live fallback.
DEFAULT_MAINTENANCE_MARGIN_RATE = 0.004


def isolated_liquidation_price(
    entry_price: float,
    is_long: bool,
    leverage: float,
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
) -> float:
    """Price at which an isolated-margin position loses its margin.

    A long is liquidated on the way down and a short on the way up, each after the
    position has burned through `1/leverage` of its notional less the maintenance
    buffer the venue keeps back.

    Raises on inputs that cannot produce a price rather than returning one. A venue that
    reports an open position with no leverage or no entry price is the case #277 is
    about: the caller must not silently receive a number that reads as "safe". Both
    callers gate on leverage > 1 before getting here, and both validate leverage >= 1, so
    the sub-1x case that would price a liquidation below zero cannot arrive.

    Written `not (x >= 1)` rather than `x < 1`: NaN compares False to every operator, so
    the natural spelling lets it through and returns a NaN price. The caller then divides
    it into a NaN distance and clamps that to 0.0 -- reporting a healthy position as AT
    liquidation. Fail-closed, but silently and wrongly, on one garbage venue tick.
    """
    if not (leverage >= 1):
        raise ValueError(f"leverage must be at least 1 to price a liquidation, got {leverage}")
    if not (entry_price > 0):
        raise ValueError(f"entry_price must be positive to price a liquidation, got {entry_price}")

    margin_fraction = 1.0 / leverage
    if is_long:
        return entry_price * (1 - margin_fraction + maintenance_margin_rate)
    return entry_price * (1 + margin_fraction - maintenance_margin_rate)


def cross_liquidation_price(
    position_size: float,
    mark_price: float,
    equity: float,
    total_account_maintenance: float,
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
) -> float:
    """Price at which the whole account's equity stops covering maintenance margin.

    The venue's aggregate maintenance is measured at the current mark. Split it into the
    focal position's current maintenance and everything else, then keep the focal term
    price-sensitive while treating the remainder as locally constant:

    `other = max(total_maintenance - rate * abs(size) * mark, 0)`
    `equity + size*(P - mark) = other + rate * abs(size) * P`

    This is what makes the estimate honest in the case the isolated formula gets wrong:
    when losses elsewhere have eaten the collateral, equity is already lower and this
    prices liquidation NEARER than isolated would. When the account is amply funded it
    prices it further away -- correctly, because it genuinely is.

    This remains a local estimate: other positions also move, risk tiers can change, and
    portfolio margin is not represented by this linear model.
    """
    if not math.isfinite(mark_price) or not (mark_price > 0):
        raise ValueError(f"mark_price must be positive to price a liquidation, got {mark_price}")
    if not math.isfinite(position_size) or not position_size:
        raise ValueError("a flat position has no liquidation price")
    if not math.isfinite(equity) or not (equity > 0):
        raise ValueError(f"equity must be positive to price a liquidation, got {equity}")
    if not math.isfinite(total_account_maintenance) or total_account_maintenance < 0:
        raise ValueError(
            "total_account_maintenance must be a finite non-negative number, "
            f"got {total_account_maintenance}"
        )
    if not math.isfinite(maintenance_margin_rate) or maintenance_margin_rate < 0:
        raise ValueError(
            "maintenance_margin_rate must be a finite non-negative number, "
            f"got {maintenance_margin_rate}"
        )

    focal_at_mark = maintenance_margin_rate * abs(position_size) * mark_price
    other_maintenance = max(total_account_maintenance - focal_at_mark, 0.0)
    denominator = position_size - maintenance_margin_rate * abs(position_size)
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError("cross-margin liquidation equation has a degenerate denominator")

    price = (
        position_size * mark_price + other_maintenance - equity
    ) / denominator
    if not math.isfinite(price):
        raise ValueError(f"cross-margin liquidation price is non-finite: {price}")
    return price


def nearest_liquidation_price(
    position_size: float,
    entry_price: float,
    mark_price: float,
    equity: float,
    leverage: float,
    total_account_maintenance: float,
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
) -> float:
    """The more urgent of the isolated and cross estimates, for a venue that publishes none.

    Neither estimate dominates. The isolated price preserves the venue-independent
    position geometry, while the cross price incorporates the account's current aggregate
    maintenance. Taking whichever sits nearer the mark avoids trusting an implausibly far
    local cross estimate when the account is amply funded.
    """
    isolated = isolated_liquidation_price(
        entry_price, is_long=position_size > 0, leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )
    cross = cross_liquidation_price(
        position_size, mark_price, equity, total_account_maintenance,
        maintenance_margin_rate=maintenance_margin_rate,
    )
    # A long is liquidated from below, so nearer means higher; a short from above.
    return max(isolated, cross) if position_size > 0 else min(isolated, cross)


def stop_precedes_liquidation(
    stop_price: float, liquidation_price: float, open_price: float, is_long: bool
) -> bool:
    """Was a triggered stop crossed before liquidation on this bar? (#300)

    A stop-loss sits on the SAME side of entry as liquidation, so price cannot reach the
    further level without crossing the nearer one, and the bar's own extreme says which.
    Booking a liquidation when the stop was crossed first is not pessimism, it is an
    outcome the data contradicts -- at 10x on 10000 cash it leaves 400 where the stop
    leaves 5000.

    The exception is a bar that OPENED past liquidation: nothing was crossed on the way
    there, the margin was already gone when the bar began, and no resting order could have
    worked first.

    Shared, because replay answering this differently from the offline env is precisely
    the divergence this module was created to prevent -- and replay did answer it
    differently, citing the rule #300 replaced.
    """
    # Spelled as `not (open <= liq)` rather than `open > liq` because the two differ on a
    # NaN open: the first says "did not gap", the second says "gapped". This is a pure
    # extraction of the offline rule, so it keeps the offline answer -- a grid over both
    # forms found 90 divergences, every one NaN-driven. Unreachable in practice (the
    # sampler guarantees a finite open), and which answer a NaN should get is a real
    # question -- but not one a refactor gets to decide by accident.
    # No stop set is not a stop that was crossed first. Without this, a short with
    # stop_loss == 0 satisfies `0 < liq and open < liq` and is exempted from liquidation
    # entirely -- so the rule carries it, rather than each caller's own guard.
    if not stop_price > 0:
        return False

    if is_long:
        gapped_past_liquidation = open_price <= liquidation_price
        stop_is_nearer = stop_price > liquidation_price
    else:
        gapped_past_liquidation = open_price >= liquidation_price
        stop_is_nearer = stop_price < liquidation_price
    return stop_is_nearer and not gapped_past_liquidation


def bankruptcy_price(
    entry_price: float, *, is_long: bool, leverage: float, transaction_fee: float
) -> float:
    """The price at which a position has consumed exactly the margin it posted (#314).

    Past it the venue closes and the insurance fund absorbs the rest, so it is the worst
    price a liquidation can be booked at however far the bar gapped.

    Net of the exit fee: the venue takes the liquidation fee out of the margin -- that is
    what the maintenance buffer is for -- so this is the fill where loss + fee equals the
    margin. Charging the fee on top instead left replay with negative equity.

    The vectorized engine keeps a tensorised copy, as it does for stop_fill_price; the
    equivalence tests pin it, so it cannot drift unnoticed.
    """
    margin_fraction = 1.0 / leverage
    return entry_price * (
        (1 - margin_fraction) / (1 - transaction_fee) if is_long
        else (1 + margin_fraction) / (1 + transaction_fee)
    )


def require_fee_fits_maintenance(
    transaction_fee: float, *, leverage: float, maintenance_margin_rate: float
) -> None:
    """Refuse a fee the maintenance buffer cannot absorb (#314).

    `bankruptcy_price` divides by `1 -+ fee`, so a fee at or above the buffer pushes it
    PAST the liquidation price and the fill clamp stops being a floor: at L=125,
    mmr=0.004, fee=0.01, liquidation is 99.6 and the clamp returns 100.194 -- above the
    bar and above entry, so a liquidated long books a profit on price. A fee of exactly
    1 divides by zero outright.

    Lives here rather than in any one config because there are THREE public paths to
    that formula -- the scalar env, the vectorized config and ReplayOrderExecutor -- and
    the first fix validated only the scalar one, leaving the defect reachable through
    the other two.

    Real venues sit far from this: Binance futures taker ~0.04% against ~0.4%
    maintenance, so the buffer is an order of magnitude above the fee.
    """
    if not (0 <= transaction_fee < 1):
        raise ValueError(
            f"Transaction fee must be in [0, 1), got {transaction_fee}"
        )
    # Strict: at fee=0 with mmr=0 the bankruptcy price EQUALS the liquidation price, so
    # the clamp is a no-op rather than a violation.
    if leverage > 1 and transaction_fee * (1 + 1 / leverage) > maintenance_margin_rate:
        raise ValueError(
            f"transaction_fee {transaction_fee} does not fit inside "
            f"maintenance_margin_rate {maintenance_margin_rate} at leverage {leverage}: "
            f"the fee would consume the maintenance buffer, putting the bankruptcy price "
            f"past the liquidation price"
        )
