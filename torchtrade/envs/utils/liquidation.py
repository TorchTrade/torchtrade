"""The isolated-margin liquidation price, in one place.

The offline envs have always computed this; the live envs never did -- they read the
venue's own `liqPrice` and, when the venue omitted it, reported the safest possible
answer (#277). This module holds the scalar rule so the live fallback and the offline
env cannot answer the same question differently.

`vectorized_sequential.py` keeps a tensorised copy of the arithmetic, but the equivalence
tests pin it to the scalar env, so it cannot drift unnoticed. The live path had no such
pin, which is why it shares this.
"""

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
    the natural spelling lets it through, and `max(0.0, nan)` is 0.0 -- which is exactly
    the "reads as safe" answer these guards exist to refuse.
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
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
) -> float:
    """Price at which the whole account's equity stops covering maintenance margin.

    Under cross margin the position is not backed by its own isolated margin but by the
    entire account, so the threshold moves with equity. Solving
    `equity + size*(P - mark) = mmr * |size| * P` for P gives the expression below.

    This is what makes the estimate honest in the case the isolated formula gets wrong:
    when losses elsewhere have eaten the collateral, equity is already lower and this
    prices liquidation NEARER than isolated would. When the account is amply funded it
    prices it further away -- correctly, because it genuinely is.

    Still an approximation: it cannot see other positions' own maintenance requirements or
    the venue's tiered margin schedule, which is why the caller pairs it with the isolated
    price and takes whichever is nearer rather than trusting this alone.
    """
    if not (mark_price > 0):
        raise ValueError(f"mark_price must be positive to price a liquidation, got {mark_price}")
    if not position_size:
        raise ValueError("a flat position has no liquidation price")

    sign = 1.0 if position_size > 0 else -1.0
    return (position_size * mark_price - equity) / (position_size * (1 - sign * maintenance_margin_rate))


def nearest_liquidation_price(
    position_size: float,
    entry_price: float,
    mark_price: float,
    equity: float,
    leverage: float,
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
) -> float:
    """The more urgent of the isolated and cross estimates, for a venue that publishes none.

    Neither estimate dominates. Isolated ignores the rest of the account, so it is wrong
    once collateral elsewhere has been consumed; cross reflects equity but assumes this
    position is the account's only maintenance obligation. Taking whichever sits nearer
    the mark is the best available answer from the fields the adapters currently expose.

    It is NOT a guaranteed bound, and must not be described as one. If the account holds
    another cross position, its maintenance requirement is missing from both estimates and
    both overstate the room left: a 1-unit long at mark 100 on equity 5 returns ~95.4
    (4.6% room) where a second position needing 4 of maintenance puts real liquidation at
    ~99.4 (0.6%). Closing that needs an account-level maintenance figure from
    `get_account_balance()`, which no adapter parses today -- see #344.

    The same single-position assumption is already baked into `exposure_pct`, the
    bankruptcy baseline and `_get_portfolio_value`, all of which divide this position by
    account-wide equity. What this function must not do is claim more certainty than they
    do. Against the `1.0` it replaces -- maximally wrong for every cross position,
    always -- it is a strict improvement in every case, which is why it ships.
    """
    isolated = isolated_liquidation_price(
        entry_price, is_long=position_size > 0, leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )
    cross = cross_liquidation_price(
        position_size, mark_price, equity, maintenance_margin_rate=maintenance_margin_rate,
    )
    # A long is liquidated from below, so nearer means higher; a short from above.
    return max(isolated, cross) if position_size > 0 else min(isolated, cross)
