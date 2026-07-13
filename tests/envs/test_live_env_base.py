"""Contract tests for behaviour shared by every live env via TorchTradeLiveEnv.

These replace the per-exchange copies of the same assertions. Testing a shared method
once is only sound if every env actually inherits it -- so the unit test here is paired
with a guard that asserts exactly that. If an exchange ever re-adds its own override, the
guard fails and tells you to test that exchange separately.
"""

from types import SimpleNamespace

import pytest

import torchtrade.envs  # noqa: F401  -- registers every live env as a subclass
from torchtrade.envs.core.live import TorchTradeLiveEnv


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Discovered, not hand-listed: a hand-listed exchange #6 would silently escape the guard.
# __subclasses__() is a live registry, so do NOT define a TorchTradeLiveEnv subclass in any
# test module -- it would land in here, import-order dependent.
LIVE_ENVS = sorted(_subclasses(TorchTradeLiveEnv), key=lambda c: c.__name__)


@pytest.mark.parametrize("done_on_bankruptcy,portfolio_value,expected", [
    (True, 50.0, True),    # below 10% of the 1000 initial -> bankrupt
    (True, 100.0, False),  # exactly at the threshold -> NOT bankrupt (the check is a strict <)
    (True, 500.0, False),  # above the threshold -> keep trading
    (False, 0.0, False),   # wiped out, but the check is off -> never terminates
], ids=["below-threshold", "at-threshold", "above-threshold", "disabled"])
def test_check_termination(done_on_bankruptcy, portfolio_value, expected):
    """Terminates iff done_on_bankruptcy and portfolio < bankrupt_threshold * initial.

    Called unbound on a stand-in rather than a real env: instantiating an EnvBase subclass
    via __new__ to skip its __init__ flakes (nn.Module.__init__ never runs, so no _modules).
    The stand-in carries only the three attributes the method reads, so a renamed config
    field raises AttributeError here instead of passing silently.
    """
    env = SimpleNamespace(
        config=SimpleNamespace(
            done_on_bankruptcy=done_on_bankruptcy,
            bankrupt_threshold=0.1,
        ),
        initial_portfolio_value=1000.0,
    )
    assert TorchTradeLiveEnv._check_termination(env, portfolio_value) is expected


def test_discovery_covers_every_live_exchange():
    """The override guard below is only as good as this discovery.

    If an exchange ever stops being imported by torchtrade.envs it drops out of LIVE_ENVS
    silently, and the guard would still pass green while covering less. Fail here instead.
    Adding exchange #6 is meant to fail this -- it forces you to confirm the newcomer
    inherits the shared bankruptcy check rather than re-forking it.
    """
    exchanges = {cls.__module__.split(".")[-2] for cls in LIVE_ENVS}
    assert exchanges == {"alpaca", "binance", "bitget", "bybit", "okx"}


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_overrides_check_termination(env_cls):
    """No live env class overrides the shared bankruptcy check.

    This is what makes testing _check_termination once (above) sufficient rather than a
    coverage loss: a re-forked copy of money-moving termination logic fails here.
    """
    assert env_cls._check_termination is TorchTradeLiveEnv._check_termination, (
        f"{env_cls.__name__} overrides _check_termination. Either drop the override, or "
        f"give that exchange its own termination tests -- the single shared test above no "
        f"longer covers it."
    )
