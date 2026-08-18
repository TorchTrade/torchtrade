"""Short action k is not the mirror of long action k, and it never is (#279)."""

import pytest

from torchtrade.envs.utils import create_sltp_action_map
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

ENTRY = 100.0


def _geometry(side, sl_pct, tp_pct):
    """Effective (risk %, reward %) at a real entry, through the real bracket helper."""
    sl_price, tp_price = calculate_bracket_prices(side, ENTRY, sl_pct, tp_pct)
    return (abs(sl_price - ENTRY) / ENTRY, abs(tp_price - ENTRY) / ENTRY)


@pytest.mark.parametrize("stoploss,takeprofit", [
    ([-0.025, -0.05, -0.1], [0.05, 0.1, 0.2]),
    # Symmetric-LOOKING, and still inverted. An earlier version of this test asserted
    # this case was fine, because it compared magnitude SETS -- which are equal here --
    # rather than pairing index against index. Long k=1 risks 5% to make 10%; short k=1
    # risks 10% to make 5%.
    ([-0.05, -0.1], [0.05, 0.1]),
    ([-0.01, -0.02, -0.03], [0.01, 0.02, 0.03]),
])
def test_short_k_has_the_risk_and_reward_of_long_k_exchanged(stoploss, takeprofit):
    """Measured through calculate_bracket_prices, not by reading the tuple.

    The tuple is `("short", tp, sl)`, and nothing downstream negates it -- the short
    bracket helper applies `entry * (1 + pct)` directly -- so the swap survives all the
    way to the prices the venue receives.
    """
    action_map = create_sltp_action_map(stoploss, takeprofit, include_short_positions=True)
    longs = [v for v in action_map.values() if v[0] == "long"]
    shorts = [v for v in action_map.values() if v[0] == "short"]

    for k, (long_action, short_action) in enumerate(zip(longs, shorts)):
        long_risk, long_reward = _geometry(*long_action)
        short_risk, short_reward = _geometry(*short_action)
        assert (short_risk, short_reward) == pytest.approx((long_reward, long_risk)), (
            f"action {k}: long is {long_risk:.4f}/{long_reward:.4f}, short is "
            f"{short_risk:.4f}/{short_reward:.4f} -- expected the exchange of the long's"
        )


def test_no_short_action_offers_the_tightest_long_stop():
    """The consequence that matters: a geometry available to longs is unreachable short."""
    action_map = create_sltp_action_map(
        [-0.025, -0.05, -0.1], [0.05, 0.1, 0.2], include_short_positions=True
    )
    long_stops = {round(_geometry(*v)[0], 6) for v in action_map.values() if v[0] == "long"}
    short_stops = {round(_geometry(*v)[0], 6) for v in action_map.values() if v[0] == "short"}

    assert 0.025 in long_stops and 0.025 not in short_stops
    assert long_stops == {0.025, 0.05, 0.1}
    assert short_stops == {0.05, 0.1, 0.2}


def test_mirroring_the_magnitudes_would_not_change_the_action_space_size():
    """Why #279 is a decision and not a patch: the size contract is identical, so a
    trained checkpoint loads without complaint and trades a different strategy."""
    stoploss, takeprofit = [-0.025, -0.05, -0.1], [0.05, 0.1, 0.2]
    current = create_sltp_action_map(stoploss, takeprofit, include_short_positions=True)
    mirrored_size = 1 + 2 * len(stoploss) * len(takeprofit)

    assert len(current) == mirrored_size == 19


def test_an_empty_level_list_still_builds_a_hold_only_map():
    """A warning helper briefly turned this working config into a ValueError."""
    assert create_sltp_action_map([], [0.05], include_short_positions=True) == {
        0: (None, None, None)
    }
