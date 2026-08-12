"""The isolated-margin liquidation price, in one place.

The offline envs have always computed this; the live envs never did -- they read the
venue's own `liqPrice` and, when the venue omitted it, reported the safest possible
answer (#277). This module holds the scalar rule so the live fallback and the offline
env cannot answer the same question differently.

`vectorized_sequential.py` keeps its own tensorised copy of the same arithmetic; that
one is pinned to the scalar env by the scalar/vectorized equivalence tests, so it
cannot drift unnoticed. The live path had no such pin, which is why it shares this.
"""

# Binance/Bybit/OKX/Bitget all publish tiered maintenance margins that start near 0.4%
# for the smallest notional brackets, which is also the offline envs' config default -- so
# at the default a policy meets the same liquidation geometry offline and live. The live
# side has no config knob for this, so an offline env configured with a different rate
# does NOT have a matching live fallback. Worth a knob only once someone needs one; the
# 1/leverage term dominates, so plausible rate choices move the estimate by ~0.1%.
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

    Clamped at zero: above ~1x leverage the long formula goes negative, which is the
    correct statement that an unlevered long cannot be liquidated, not a price.

    Raises on inputs that cannot produce a price rather than returning one. A venue that
    reports an open position with leverage 0 or no entry price is the case #277 is about:
    the caller must not silently receive a number that reads as "safe".
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be positive to price a liquidation, got {leverage}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive to price a liquidation, got {entry_price}")

    margin_fraction = 1.0 / leverage
    if is_long:
        price = entry_price * (1 - margin_fraction + maintenance_margin_rate)
    else:
        price = entry_price * (1 + margin_fraction - maintenance_margin_rate)
    return max(0.0, price)
