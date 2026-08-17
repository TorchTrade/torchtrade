"""The recorded portfolio value must belong to the bar the action was taken on (#278)."""

import inspect

import pytest

from torchtrade.envs.live.alpaca import env as alpaca_env
from torchtrade.envs.live.alpaca import env_sltp as alpaca_env_sltp
from torchtrade.envs.live.shared import futures_live_base


@pytest.mark.parametrize("module,label", [
    (futures_live_base, "futures_live_base"),   # binance, bitget, bybit, okx
    (alpaca_env, "alpaca.env"),
    (alpaca_env_sltp, "alpaca.env_sltp"),
])
def test_the_observation_is_acquired_before_the_portfolio_value(module, label):
    """Order, not a value, because the lag is invisible to a live-mocked assertion.

    Under a ReplayObserver the clock advances only inside `get_observations()`, so
    reading the portfolio value first records the PREVIOUS bar's equity against this
    bar's action: measured 8/8 against the decision bar and 0/8 against the next one,
    while `account_state` in the same step was already at the new bar. A tuple evaluates
    left to right, so `(self._get_portfolio_value(), self._get_observation())` reads
    them in the wrong order while looking symmetric -- which is why this pins the
    source text rather than a number.
    """
    source = inspect.getsource(module)
    obs_at = source.find("self._get_observation()")
    pv_at = source.find("self._get_portfolio_value()")
    assert obs_at != -1 and pv_at != -1, f"{label} no longer reads both"
    assert obs_at < pv_at, (
        f"{label} reads the portfolio value before the observation, so a replayed "
        f"reward belongs to the previous bar (#278)"
    )
