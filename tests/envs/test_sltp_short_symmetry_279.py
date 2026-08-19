"""Short action k is the mirror of long action k -- same risk, same reward (#279)."""

import pytest

from torchtrade.envs.utils import create_sltp_action_map
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

ENTRY = 100.0


def _geometry(side, sl_pct, tp_pct):
    """Effective (risk %, reward %) at a real entry, through the real bracket helper.

    SIGNED against the position, not `abs()`: a bare magnitude cannot tell a stop below
    entry from one above it, so a mutation putting a short's stop BELOW entry -- a
    real-money inversion -- left every assertion here passing. Risk must come out
    positive for a correctly placed bracket and negative for an inverted one.
    """
    sl_price, tp_price = calculate_bracket_prices(side, ENTRY, sl_pct, tp_pct)
    direction = 1.0 if side == "long" else -1.0
    return ((ENTRY - sl_price) * direction / ENTRY, (tp_price - ENTRY) * direction / ENTRY)


@pytest.mark.parametrize("stoploss,takeprofit", [
    ([-0.025, -0.05, -0.1], [0.05, 0.1, 0.2]),
    # Symmetric-LOOKING, and it was still inverted before the fix: comparing magnitude
    # SETS passes here, because they are equal. Only pairing index against index sees it.
    ([-0.05, -0.1], [0.05, 0.1]),
    # Asymmetric lengths: the relationship is an identity, not an artefact of equal-sized
    # lists -- both halves consume the same product(), so the nth pair always matches.
    ([-0.02, -0.05], [0.03, 0.06, 0.12]),
])
def test_short_k_mirrors_long_k(stoploss, takeprofit):
    """Measured through calculate_bracket_prices, not by reading the tuple.

    The map was built by SWAPPING the two lists, which puts the sides right and the
    magnitudes wrong: short k carried long k's risk and reward exchanged. It negates now.
    Nothing downstream re-negates -- the bracket helper applies `entry * (1 + pct)`
    directly -- so what this asserts is what the venue receives.
    """
    action_map = create_sltp_action_map(stoploss, takeprofit, include_short_positions=True)
    longs = [v for v in action_map.values() if v[0] == "long"]
    shorts = [v for v in action_map.values() if v[0] == "short"]

    for k, (long_action, short_action) in enumerate(zip(longs, shorts)):
        long_risk, long_reward = _geometry(*long_action)
        short_risk, short_reward = _geometry(*short_action)
        assert long_risk > 0 and short_risk > 0, (
            f"action {k}: a stop on the wrong side of entry (long {long_risk:.4f}, "
            f"short {short_risk:.4f}) -- the bracket is inverted, not merely exchanged"
        )
        assert (short_risk, short_reward) == pytest.approx((long_risk, long_reward)), (
            f"action {k}: long is {long_risk:.4f}/{long_reward:.4f}, short is "
            f"{short_risk:.4f}/{short_reward:.4f} -- expected the same geometry"
        )


def test_every_long_geometry_is_reachable_short():
    """The consequence that mattered: no short action had a 2.5% stop anywhere."""
    action_map = create_sltp_action_map(
        [-0.025, -0.05, -0.1], [0.05, 0.1, 0.2], include_short_positions=True
    )
    long_stops = {round(_geometry(*v)[0], 6) for v in action_map.values() if v[0] == "long"}
    short_stops = {round(_geometry(*v)[0], 6) for v in action_map.values() if v[0] == "short"}

    assert long_stops == short_stops == {0.025, 0.05, 0.1}


def test_an_empty_level_list_still_builds_a_hold_only_map():
    """A warning helper briefly turned this working config into a ValueError."""
    assert create_sltp_action_map([], [0.05], include_short_positions=True) == {
        0: (None, None, None)
    }
