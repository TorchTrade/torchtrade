"""The one bankruptcy rule, shared by the exchange envs and Polymarket."""

import pytest

from torchtrade.envs.utils.termination import is_bankrupt


@pytest.mark.parametrize("current,initial,threshold,enabled,expected", [
    (50.0, 1000.0, 0.1, True, True),    # below 10% of the 1000 start -> bankrupt
    (100.0, 1000.0, 0.1, True, False),  # exactly at the threshold -> NOT yet (strict <)
    (500.0, 1000.0, 0.1, True, False),  # above -> keep going
    (0.0, 1000.0, 0.1, False, False),   # wiped out, but the check is off
    (0.0, 0.0, 0.1, True, False),       # nothing to be bankrupt against
    # A negative initial, with current BELOW threshold*initial (-0.5). Picking current=0
    # here would be useless: the arithmetic returns False anyway, so `<= 0` and `== 0`
    # would agree and the guard would go unpinned.
    (-1.0, -5.0, 0.1, True, False),
], ids=["below", "at-threshold", "above", "disabled", "zero-initial", "negative-initial"])
def test_is_bankrupt(current, initial, threshold, enabled, expected):
    """Bankrupt iff enabled, initial > 0, and current is strictly below the threshold."""
    assert is_bankrupt(current=current, initial=initial,
                       threshold=threshold, enabled=enabled) is expected
