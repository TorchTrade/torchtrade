"""Contract tests for behaviour shared by every live env via TorchTradeLiveEnv.

These replace the per-exchange copies of the same assertions. Testing a shared method
once is only sound if every env actually inherits it -- so each unit test here is paired
with a guard that asserts exactly that. If an exchange ever re-adds its own override, the
guard fails and tells you to test that exchange separately.
"""

from types import SimpleNamespace

import pytest

from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv
from torchtrade.envs.live.alpaca.env_sltp import AlpacaSLTPTorchTradingEnv
from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv
from torchtrade.envs.live.binance.env_sltp import BinanceFuturesSLTPTorchTradingEnv
from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv
from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTorchTradingEnv
from torchtrade.envs.live.bybit.env import BybitFuturesTorchTradingEnv
from torchtrade.envs.live.bybit.env_sltp import BybitFuturesSLTPTorchTradingEnv
from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv
from torchtrade.envs.live.okx.env_sltp import OKXFuturesSLTPTorchTradingEnv

LIVE_ENVS = [
    AlpacaTorchTradingEnv, AlpacaSLTPTorchTradingEnv,
    BinanceFuturesTorchTradingEnv, BinanceFuturesSLTPTorchTradingEnv,
    BitgetFuturesTorchTradingEnv, BitgetFuturesSLTPTorchTradingEnv,
    BybitFuturesTorchTradingEnv, BybitFuturesSLTPTorchTradingEnv,
    OKXFuturesTorchTradingEnv, OKXFuturesSLTPTorchTradingEnv,
]


class _ConcreteLiveEnv(TorchTradeLiveEnv):
    """Smallest possible live env: satisfies the abstract methods, overrides nothing else."""

    def _init_trading_clients(self, api_key, api_secret, observer, trader):
        raise NotImplementedError

    def _get_portfolio_value(self, *args, **kwargs):
        raise NotImplementedError

    def _reset(self, tensordict, **kwargs):
        raise NotImplementedError

    def _step(self, tensordict):
        raise NotImplementedError


def _live_env(done_on_bankruptcy=True, bankrupt_threshold=0.1, initial_portfolio_value=1000.0):
    """A live env carrying only what _check_termination reads.

    __new__ skips EnvBase's spec machinery, which the method under test never touches.
    """
    env = _ConcreteLiveEnv.__new__(_ConcreteLiveEnv)
    env.config = SimpleNamespace(
        done_on_bankruptcy=done_on_bankruptcy,
        bankrupt_threshold=bankrupt_threshold,
    )
    env.initial_portfolio_value = initial_portfolio_value
    return env


@pytest.mark.parametrize("done_on_bankruptcy,portfolio_value,expected", [
    (True, 50.0, True),    # below 10% of the 1000 initial -> bankrupt
    (True, 100.0, False),  # exactly at the threshold -> NOT bankrupt (the check is a strict <)
    (True, 500.0, False),  # above the threshold -> keep trading
    (False, 0.0, False),   # wiped out, but the check is off -> never terminates
], ids=["below-threshold", "at-threshold", "above-threshold", "disabled"])
def test_check_termination(done_on_bankruptcy, portfolio_value, expected):
    """Terminates iff done_on_bankruptcy and portfolio < bankrupt_threshold * initial."""
    env = _live_env(done_on_bankruptcy=done_on_bankruptcy)
    assert env._check_termination(portfolio_value) is expected


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_overrides_check_termination(env_cls):
    """Every live env inherits the shared bankruptcy check.

    This is what makes testing _check_termination once (above) sufficient. It also guards
    the hoist: the five exchanges used to carry byte-identical copies, and a copy that
    drifts is a silent divergence in money-moving termination logic.
    """
    assert env_cls._check_termination is TorchTradeLiveEnv._check_termination
