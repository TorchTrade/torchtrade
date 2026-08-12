"""#339: the flat-enough threshold was denominated in base-asset units."""

import pandas as pd
import torch
from tensordict import TensorDict
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
@pytest.mark.parametrize("gap_usd,within", [(0.50, True), (5.0, False)],
                         ids=["fifty-cents", "five-dollars"])
def test_flat_enough_means_the_same_dollar_amount_on_every_asset(price, gap_usd, within):
    """Driven through the real engine, because the arithmetic alone is true of any
    constant -- a version of this test that did its own conversion passed the mutant.

    0.001 base units was $100 of unclosed position on BTC (1% of a $10k account read as
    flat) and $0.0000 on an asset priced in cents. A residual worth fifty cents should be
    flat on both; one worth fifty dollars should be flat on neither.
    """
    env = _env_at(price)
    # A RESIZE, not a close. Driving action_value=0.0 tests the close path, which is
    # deliberately exempt from the floor -- a position too small to resize must still be
    # closable, or account_state reports a direction the policy never asked for.
    # A $10 target, so the $1 FLOOR dominates: 2% of $10 is $0.20. On a large target the
    # relative term wins and the floor is never measured -- which is what the first
    # version of this test was accidentally checking.
    target = 10.0 / price
    env.position.position_size = target + gap_usd / price

    info = env._execute_fractional_action(action_value=0.001, execution_price=price)

    assert (info["executed"] is False) is within, (
        f"a ${gap_usd} gap from target at price {price} was "
        f"{'not ' if within else ''}treated as close enough"
    )


@pytest.mark.parametrize("price", [0.0, -50.0, float("nan"), float("inf")],
                         ids=["zero", "negative", "nan", "inf"])
def test_an_unusable_price_is_refused_where_it_enters(price):
    """Invariant 4: at the boundary, not with a clamp inside the rule.

    I first "fixed" this by clamping the price at the tolerance line and wrote a comment
    saying that stopped the ZeroDivisionError. It did not -- _calculate_fractional_position
    divides by the raw price BEFORE that line, so the clamp only ever helped action_value
    == 0. A clamp that turns a nonsense price into a huge tolerance is fail-open anyway:
    every position reads as flat and nothing trades.
    """
    from torchtrade.envs.utils.fractional_sizing import (
        PositionCalculationParams,
        calculate_fractional_position,
    )

    with pytest.raises(ValueError, match="price"):
        calculate_fractional_position(PositionCalculationParams(
            balance=10_000.0, action_value=1.0, current_price=price, leverage=1,
        ))




@pytest.mark.parametrize("price", [100_000.0, 0.35], ids=["btc-like", "doge-like"])
def test_both_engines_close_a_position_worth_less_than_the_floor(price):
    """Drives BOTH engines, because the previous version of this test named both and
    imported neither -- the same self-verifying shape as its first draft.

    That mattered: the CLOSE exemption was added to the scalar engine only, so a sub-$1
    position stayed unclosable in the vectorized one and the two disagreed. The
    equivalence harness anchors every fixture at ~$100 with large positions, so it never
    reaches this regime.
    """
    import torch

    from torchtrade.envs.offline import (
        VectorizedSequentialTradingEnv,
        VectorizedSequentialTradingEnvConfig,
    )

    idx = pd.date_range("2024-01-01", periods=600, freq="1min")
    df = pd.DataFrame({"timestamp": idx, "open": price, "high": price * 1.001,
                       "low": price * 0.999, "close": price, "volume": 10.0})

    scalar = _env_at(price)
    scalar.position.position_size = 0.50 / price
    scalar_closed = scalar._execute_fractional_action(0.0, price)["executed"]
    assert scalar_closed is True, "scalar: a sub-floor position was unclosable"

    vec = VectorizedSequentialTradingEnv(df, VectorizedSequentialTradingEnvConfig(
        symbol="X", time_frames=["1Minute"], window_sizes=[10], execute_on="1Minute",
        initial_cash=10_000.0, num_envs=1, action_levels=[0.0, 1.0],
    ))
    vec.reset()
    # Fifty cents of position, then command FLAT through the REAL step. Computing the
    # tolerance in the test instead lets the production line be deleted with the test
    # green -- which is exactly what happened on the previous draft.
    vec._position_sizes = torch.full_like(vec._position_sizes, 0.50 / price)
    td = TensorDict({"action": torch.zeros(1, dtype=torch.long)}, batch_size=[1])
    vec.step(td)

    assert vec._position_sizes.abs().max().item() == pytest.approx(0.0, abs=1e-12), (
        "vectorized: a sub-floor position was unclosable"
    )


@pytest.mark.parametrize("price", [100_000.0, 0.35], ids=["btc-like", "doge-like"])
def test_a_position_worth_less_than_the_floor_can_still_be_closed(price):
    """The regression the notional floor introduced, and the reason CLOSE is exempt.

    The floor answers "is this resize worth the fee". Going flat is not a resize -- with
    the floor applied to it, a position whose value falls under $1 could never be closed:
    every flat command sat inside the band, and account_state kept reporting a direction
    the policy had explicitly asked to leave. That is invariant 3, introduced by the fix
    for #339 rather than found there.
    """
    env = _env_at(price)
    env.position.position_size = 0.50 / price          # fifty cents of position

    info = env._execute_fractional_action(action_value=0.0, execution_price=price)

    assert info["executed"] is True, "a sub-floor position was unclosable"
