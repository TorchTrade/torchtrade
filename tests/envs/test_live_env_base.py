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
LIVE_ENVS = sorted(set(_subclasses(TorchTradeLiveEnv)), key=lambda c: c.__name__)


@pytest.mark.parametrize("done_on_bankruptcy,portfolio_value,expected", [
    (True, 50.0, True),    # below 10% of the 1000 initial -> bankrupt
    (True, 100.0, False),  # exactly at the threshold -> NOT bankrupt (the check is a strict <)
    (True, 500.0, False),  # above the threshold -> keep trading
    (False, 0.0, False),   # wiped out, but the check is off -> never terminates
], ids=["below-threshold", "at-threshold", "above-threshold", "disabled"])
def test_check_termination(done_on_bankruptcy, portfolio_value, expected):
    """Terminates iff done_on_bankruptcy and portfolio < bankrupt_threshold * initial.

    Called unbound on a stand-in `self`: the method reads two attributes and does
    arithmetic, so there is nothing here an EnvBase instance would add. (Building one via
    __new__ to skip EnvBase.__init__ was tried and flakes -- nn.Module.__init__ never runs,
    so the object has no _modules.)

    The stand-in carries only the two fields the method may read, which makes a renamed
    config field an AttributeError here rather than a silent pass.
    """
    env = SimpleNamespace(
        config=SimpleNamespace(
            done_on_bankruptcy=done_on_bankruptcy,
            bankrupt_threshold=0.1,
        ),
        initial_portfolio_value=1000.0,
    )
    assert TorchTradeLiveEnv._check_termination(env, portfolio_value) is expected


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_overrides_check_termination(env_cls):
    """Every live env inherits the shared bankruptcy check.

    This is what makes testing _check_termination once (above) sufficient, and for the five
    SLTP envs it is their only bankruptcy coverage outside their own _step tests. Duplicated
    logic in this codebase has drifted into real bugs before (three divergent SLTP action
    maps), so a re-forked copy of money-moving termination logic fails here immediately.
    """
    assert env_cls._check_termination is TorchTradeLiveEnv._check_termination, (
        f"{env_cls.__name__} overrides _check_termination. Either drop the override, or "
        f"give that exchange its own termination tests -- the single shared test above no "
        f"longer covers it."
    )
