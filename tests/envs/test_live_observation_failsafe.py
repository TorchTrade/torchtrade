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


@pytest.mark.parametrize("policy,close_result,expect_closed,expect_accepted", [
    (ObservationFailurePolicy.HALT, True, False, None),
    (ObservationFailurePolicy.FLATTEN, True, True, True),
    # The venue can REJECT the close without raising. Reporting that as accepted is the
    # "operator believes the position is flat when it is not" bug the exception warns of.
    (ObservationFailurePolicy.FLATTEN, False, True, False),
], ids=["halt", "flatten-accepted", "flatten-rejected"])
def test_flatten_policy_controls_the_emergency_close(
    policy, close_result, expect_closed, expect_accepted
):
    env, closed = _env(PositionUnknownError("outage"), policy=policy,
                       close_result=close_result)

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert bool(closed) is expect_closed
    assert exc.value.flatten_accepted is expect_accepted


def test_a_failed_emergency_close_is_recorded_not_swallowed():
    """The operator has to know the position may still be open."""
    env, _ = _env(PositionUnknownError("outage"), policy=ObservationFailurePolicy.FLATTEN)
    env.trader.close_position = lambda: (_ for _ in ()).throw(RuntimeError("venue down"))

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert isinstance(exc.value.flatten_error, RuntimeError)
    assert exc.value.flatten_accepted is not True


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_every_futures_step_reads_state_through_the_halting_helper(exchange, module):
    """The wiring, not the helper.

    A `_step` reading the venue directly gets the raw exception -- no halt, no emergency
    flatten -- and reverting one env to do so fails nothing else in the suite. Structural
    because the alternative is eight live-env harnesses to assert one call site each.
    """
    import importlib
    import inspect

    ns = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}").__dict__
    env_cls = next(v for k, v in ns.items()
                   if inspect.isclass(v) and k.endswith("TorchTradingEnv")
                   and v.__module__.endswith(module))
    source = inspect.getsource(env_cls._step)

    assert "_acquire_post_bar_state()" in source, f"{exchange}/{module} bypasses the halt"
    assert "self._get_observation()" not in source, f"{exchange}/{module} reads directly"


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_every_futures_config_coerces_its_failure_policy(exchange, module):
    """Eight dataclasses each declare the field and coerce it in __post_init__.

    A copy-pasted config missing the coercion would accept a bad string and only fail in
    production -- the opposite of validating at the boundary.
    """
    import importlib
    import inspect

    ns = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}").__dict__
    cfg_cls = next(v for k, v in ns.items()
                   if inspect.isclass(v) and k.endswith("Config")
                   and hasattr(v, "observation_failure_policy"))

    assert cfg_cls().observation_failure_policy is ObservationFailurePolicy.HALT
    assert cfg_cls(observation_failure_policy="flatten").observation_failure_policy is (
        ObservationFailurePolicy.FLATTEN
    )
    with pytest.raises(ValueError):
        cfg_cls(observation_failure_policy="nonsense")
