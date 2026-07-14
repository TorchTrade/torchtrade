"""The one bankruptcy rule, shared by the exchange envs and Polymarket."""

import pytest

from torchtrade.envs.utils.termination import is_bankrupt


@pytest.mark.parametrize("current,initial,threshold,enabled,expected", [
    (50.0, 1000.0, 0.1, True, True),    # below 10% of the 1000 start -> bankrupt
    (100.0, 1000.0, 0.1, True, False),  # exactly at the threshold -> NOT yet (strict <)
    (500.0, 1000.0, 0.1, True, False),  # above -> keep going
    (0.0, 1000.0, 0.1, False, False),   # wiped out, but the check is off
], ids=["below", "at-threshold", "above", "disabled"])
def test_is_bankrupt(current, initial, threshold, enabled, expected):
    """Bankrupt iff enabled and current is strictly below threshold * initial."""
    assert is_bankrupt(current=current, initial=initial,
                       threshold=threshold, enabled=enabled) is expected
