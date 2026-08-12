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
