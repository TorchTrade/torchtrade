"""#339: the flat-enough threshold was denominated in base-asset units."""

import pandas as pd
import pytest

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig


def _env_at(price):
    idx = pd.date_range("2024-01-01", periods=600, freq="1min")
    df = pd.DataFrame({"timestamp": idx, "open": price, "high": price * 1.001,
                       "low": price * 0.999, "close": price, "volume": 10.0})
    config = SequentialTradingEnvConfig(
        symbol="X", time_frames=["1Minute"], window_sizes=[10], execute_on="1Minute",
        initial_cash=10_000.0,
    )
    env = SequentialTradingEnv(df, config)
    env.reset()
    return env


@pytest.mark.parametrize("price", [100_000.0, 0.35], ids=["btc-like", "doge-like"])
@pytest.mark.parametrize("residual_usd,within", [(0.50, True), (50.0, False)],
                         ids=["fifty-cents", "fifty-dollars"])
def test_flat_enough_means_the_same_dollar_amount_on_every_asset(price, residual_usd, within):
    """Driven through the real engine, because the arithmetic alone is true of any
    constant -- a version of this test that did its own conversion passed the mutant.

    0.001 base units was $100 of unclosed position on BTC (1% of a $10k account read as
    flat) and $0.0000 on an asset priced in cents. A residual worth fifty cents should be
    flat on both; one worth fifty dollars should be flat on neither.
    """
    env = _env_at(price)
    env.position.position_size = residual_usd / price     # the residual, in base units

    info = env._execute_fractional_action(action_value=0.0, execution_price=price)

    assert (info["executed"] is False) is within, (
        f"a ${residual_usd} residual at price {price} was "
        f"{'not ' if within else ''}treated as flat"
    )
