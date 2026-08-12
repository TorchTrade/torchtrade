"""#343: a live env that cannot read its own account state must halt, not improvise."""

import pytest
from types import SimpleNamespace

from torchtrade.envs.core.live import (
    LiveObservationHalt,
    ObservationFailurePolicy,
)
from torchtrade.envs.core.state import PositionUnknownError
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv


def _env(error, policy=ObservationFailurePolicy.HALT, close_result=True):
    closed = []

    def close_position():
        closed.append(True)
        return close_result

    return SimpleNamespace(
        _get_portfolio_value=lambda: (_ for _ in ()).throw(error),
        _get_observation=lambda **k: None,
        trader=SimpleNamespace(close_position=close_position),
        config=SimpleNamespace(symbol="BTCUSDT", observation_failure_policy=policy),
    ), closed


@pytest.mark.parametrize("error", [
    PositionUnknownError("exchange status is unknown"),
    ValueError("venue reported a non-finite mark_price"),
], ids=["unknown-position", "impossible-account-state"])
def test_a_terminal_failure_halts_and_raises(error):
    """Terminal means the venue told us something impossible about our own money."""
    env, _ = _env(error)

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert exc.value.original_exception is error
    assert exc.value.policy is ObservationFailurePolicy.HALT


@pytest.mark.parametrize("error", [
    RuntimeError("Failed to get account balance: read timeout"),
    KeyError("features_1m"),
], ids=["adapter-wrapped-transient", "config-error"])
def test_a_non_terminal_failure_propagates_without_halting(error):
    """The money-critical distinction.

    Every adapter wraps any exception in `RuntimeError("Failed to get account balance:
    ...")`, so treating RuntimeError as terminal would classify a read timeout as fatal
    and -- under FLATTEN -- market-close a live position on a blip. KeyError is a
    config/programmer error and should crash rather than end the episode.
    """
    env, closed = _env(error, policy=ObservationFailurePolicy.FLATTEN)

    with pytest.raises(type(error)):
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert closed == [], "a transient failure closed a live position"


@pytest.mark.parametrize("policy,expect_closed", [
    (ObservationFailurePolicy.HALT, False),
    (ObservationFailurePolicy.FLATTEN, True),
])
def test_flatten_policy_controls_the_emergency_close(policy, expect_closed):
    env, closed = _env(PositionUnknownError("outage"), policy=policy)

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert bool(closed) is expect_closed
    assert (exc.value.flatten_accepted is True) is expect_closed


def test_a_failed_emergency_close_is_recorded_not_swallowed():
    """The operator has to know the position may still be open."""
    env, _ = _env(PositionUnknownError("outage"), policy=ObservationFailurePolicy.FLATTEN)
    env.trader.close_position = lambda: (_ for _ in ()).throw(RuntimeError("venue down"))

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert isinstance(exc.value.flatten_error, RuntimeError)
    assert exc.value.flatten_accepted is not True
