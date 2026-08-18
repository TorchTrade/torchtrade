"""The short/long geometry asymmetry is stated out loud, not fixed (#279)."""

import pytest

from torchtrade.envs.utils import create_sltp_action_map


def test_the_asymmetry_is_real_and_no_short_action_has_the_tight_stop():
    """Pins the behaviour the warning describes, so the eventual fix has a baseline.

    Shorts reuse the OPPOSITE list's magnitudes: long action k risks 2.5% to make 5%,
    the mirrored short risks 5% to make 2.5%, and no short action anywhere in the space
    has a 2.5% stop. A policy that learned "tight stop, wide target" gets the opposite
    geometry the moment it goes short.
    """
    action_map = create_sltp_action_map(
        [-0.025, -0.05, -0.1], [0.05, 0.1, 0.2], include_short_positions=True
    )
    shorts = [v for v in action_map.values() if v[0] == "short"]
    longs = [v for v in action_map.values() if v[0] == "long"]

    assert longs[0][1:] == (-0.025, 0.05), "long k risks 2.5% to make 5%"
    assert shorts[0][1:] == (0.05, -0.025), "short k risks 5% to make 2.5% -- inverted"
    assert not any(abs(sl) == pytest.approx(0.025) for _, sl, _ in shorts), (
        "no short action has the tightest stop the long side offers"
    )


@pytest.mark.parametrize("stoploss,takeprofit,shorts,expect_warning", [
    ([-0.025, -0.05], [0.05, 0.1], True, True),    # magnitudes differ -> warn
    ([-0.05, -0.1], [0.05, 0.1], True, False),     # mirrored -> silent
    ([-0.05], [0.05], True, False),                # single mirrored pair -> silent
    ([-0.025, -0.05], [0.05, 0.1], False, False),  # long-only: no short to mismatch
])
def test_the_warning_fires_exactly_when_the_geometry_is_not_mirrored(
    stoploss, takeprofit, shorts, expect_warning, caplog
):
    """A warning on every construction would be noise, and one that never fires is
    decoration. It keys on the magnitude SETS, so `(-0.05, -0.1)` against `(0.05, 0.1)`
    is silent -- there the swap really does mirror.
    """
    with caplog.at_level("WARNING"):
        create_sltp_action_map(stoploss, takeprofit, include_short_positions=shorts)

    fired = any("#279" in record.message for record in caplog.records)
    assert fired is expect_warning


def test_the_action_space_size_is_unchanged_by_the_warning():
    """The warning must not alter the map -- a checkpoint's action indices still mean
    what they meant. That is also why this is a warning and not a fix: mirroring the
    magnitudes keeps the SIZE identical while changing what every short index means, so
    a trained model would load without complaint and trade a different strategy.
    """
    action_map = create_sltp_action_map(
        [-0.025, -0.05, -0.1], [0.05, 0.1, 0.2], include_short_positions=True
    )
    assert len(action_map) == 1 + 2 * 3 * 3
