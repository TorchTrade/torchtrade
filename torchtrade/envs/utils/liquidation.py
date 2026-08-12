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
