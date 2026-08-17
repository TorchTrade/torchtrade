"""The step->decimals rule (#278)."""

import pytest

from torchtrade.envs.utils.precision import decimals_for_step


@pytest.mark.parametrize("step,expected", [
    (0.001, 3),
    (1e-06, 6),        # str() is '1e-06': no '.', which is what the old rule read as 0
    (1e-08, 8),
    (0.01, 2),
    (1, 0),
    (1.0, 0),
    (1000, 0),         # normalize() gives 1E+3; a NEGATIVE ndigits would raise
    ("0.0001", 4),     # venues send strings (okx lotSz) as well as floats
    ("1e-05", 5),
    (None, 0),         # a missing field must not take the executor down
    ("nonsense", 0),
])
def test_decimals_for_step(step, expected):
    assert decimals_for_step(step) == expected


def test_a_micro_step_does_not_round_a_quantity_up_to_a_whole_unit():
    """The #278 bug: 0.977 BTC became 1.0, and the open was refused for insufficient
    balance -- silently, via logger.warning. Replay then reported a flat strategy for a
    policy that was trading."""
    qty, step = 0.977065134495602, 1e-06
    assert round(int(qty / step) * step, decimals_for_step(step)) == pytest.approx(0.977065)
