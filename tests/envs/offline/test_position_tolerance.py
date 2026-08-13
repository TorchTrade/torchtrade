"""#339: the flat-enough threshold was denominated in base-asset units."""

import pandas as pd
import torch
from tensordict import TensorDict
import pytest

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)


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


@pytest.mark.parametrize("gap_usd,expect_hold", [(0.50, True), (5.0, False)],
                         ids=["under-the-floor", "over-the-floor"])
def test_the_vectorized_floor_holds_a_sub_dollar_resize(gap_usd, expect_hold):
    """The vectorized floor had ZERO coverage: gutting it left all 2865 tests green.

    The sibling close test does not reach it -- a close sets target 0, which the exemption
    gives a tolerance of 0, so the floor is never consulted. Only a RESIZE consults it, and
    only at a target small enough ($10 here) that the $1 floor beats the 2% term.
    """
    price = 100_000.0
    idx = pd.date_range("2024-01-01", periods=600, freq="1min")
    df = pd.DataFrame({"timestamp": idx, "open": price, "high": price * 1.001,
                       "low": price * 0.999, "close": price, "volume": 10.0})

    vec = VectorizedSequentialTradingEnv(df, VectorizedSequentialTradingEnvConfig(
        symbol="X", time_frames=["1Minute"], window_sizes=[10], execute_on="1Minute",
        initial_cash=10_000.0, num_envs=1, action_levels=[0.0, 0.001],
    ))
    vec.reset()
    target = 10.0 / price                       # action 0.001 of $10k -> a $10 position
    before = target + gap_usd / price
    vec._position_sizes = torch.full_like(vec._position_sizes, before)

    vec.step(TensorDict({"action": torch.ones(1, dtype=torch.long)}, batch_size=[1]))

    held = vec._position_sizes.item() == pytest.approx(before, rel=1e-9)
    assert held is expect_hold, (
        f"a ${gap_usd} gap was {'not ' if expect_hold else ''}held by the floor"
    )


@pytest.mark.parametrize("price", [100_000.0, 100.0, 0.35, 0.01],
                         ids=["btc", "eth-ish", "doge", "cent"])
def test_both_engines_open_a_sub_dollar_position_from_flat(price):
    """An open from flat is not a resize, and the floor must not refuse it.

    The scalar gate was unconditional while the vectorized gates on `has_position`, so
    from flat the floor refused the scalar open and let the vectorized one through -- the
    policy saw a long in one engine and nothing in the other. Worse on the scalar side it
    was absorbing: with the portfolio under $1 the agent could never re-enter at all.

    Every fixture in the equivalence harness holds a position before stepping, so
    `has_position` is always true there and this branch is never reached.
    """
    scalar = _env_at(price)
    assert scalar.position.position_size == 0, "this test is about opening from FLAT"

    info = scalar._execute_fractional_action(action_value=0.00005, execution_price=price)

    assert info["executed"] is True, "the floor refused an open from flat"


@pytest.mark.parametrize("env_cls,cfg_cls", [
    (SequentialTradingEnv, SequentialTradingEnvConfig),
    (VectorizedSequentialTradingEnv, VectorizedSequentialTradingEnvConfig),
])
def test_slippage_of_one_is_refused_by_every_engine(env_cls, cfg_cls):
    """`slippage == 1.0` makes uniform_(0, 2) legal, so a fill price can reach ~0.

    Measured before this was closed: a minimum fill of $0.0005 against a true price of
    $100, and a $4.14e10 position opened from a $10,000 account -- with no exception and
    account_state reading a tidy 1.0. The bound was tightened in core/base.py and the
    vectorized config kept its own copy at `<= 1`, so half the fix landed.
    """
    idx = pd.date_range("2024-01-01", periods=600, freq="1min")
    df = pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 100.1,
                       "low": 99.9, "close": 100.0, "volume": 10.0})
    kwargs = dict(symbol="X", time_frames=["1Minute"], window_sizes=[10],
                  execute_on="1Minute", slippage=1.0)
    if cfg_cls is VectorizedSequentialTradingEnvConfig:
        kwargs["num_envs"] = 1

    with pytest.raises(ValueError, match="[Ss]lippage"):
        env_cls(df, cfg_cls(**kwargs))


def test_a_flat_command_survives_a_zero_price_bar():
    """The boundary raise sat above the neutral return, so a flat command crashed.

    A bar of all zeros is internally consistent and passes the OHLC validator. A policy
    that only ever commands flat and holds nothing does no sizing arithmetic at all --
    raising there killed an episode that previously ran fine.
    """
    from torchtrade.envs.utils.fractional_sizing import (
        PositionCalculationParams, calculate_fractional_position,
    )

    assert calculate_fractional_position(PositionCalculationParams(
        balance=10_000.0, action_value=0.0, current_price=0.0, leverage=1,
    )) == (0.0, 0.0, "flat")


@pytest.mark.parametrize("gap_pct,within", [(0.01, True), (0.05, False)],
                         ids=["inside-2pct", "outside-2pct"])
def test_the_relative_term_binds_on_a_large_position(gap_pct, within):
    """The 2% term never dominated in any other test, so deleting it survived 812 tests.

    Every other case here anchors at a $10 target so the $1 floor wins -- which is right
    for testing the floor, and means the relative term was never the binding one anywhere.
    It exists to stop fee-eating churn from small price moves on LARGE positions, exactly
    the regime none of those tests reach. A $5,000 target makes 2% = $100, well past the
    floor.
    """
    price = 100.0
    env = _env_at(price)
    # Seeded CONSISTENTLY: adding a position without deducting its cash inflates portfolio
    # value, which moves the target the test is measuring the gap against.
    target = 5_000.0 / price                       # action 0.5 of $10k -> a $5,000 position
    env.position.position_size = target * (1 + gap_pct)
    env.balance -= env.position.position_size * price
    env.position.entry_price = price

    info = env._execute_fractional_action(action_value=0.5, execution_price=price)

    assert (info["executed"] is False) is within, (
        f"a {gap_pct:.0%} gap on a $5,000 position was "
        f"{'not ' if within else ''}treated as close enough"
    )


# Charlie's counter-example on #364: at these values the bankruptcy price is 100.194
# while liquidation is 99.6, so max(fill, bankruptcy) stops being a floor and books a
# liquidated long ABOVE the bar and above entry.
_INVERTING = dict(leverage=125, maintenance_margin_rate=0.004, transaction_fee=0.01)
_DIV_ZERO = dict(leverage=10, maintenance_margin_rate=0.004, transaction_fee=1.0)


@pytest.mark.parametrize("kwargs,match", [
    (_INVERTING, "does not fit inside"),
    (_DIV_ZERO, r"\[0, 1\)"),
], ids=["fee-exceeds-buffer", "fee-of-one"])
@pytest.mark.parametrize("boundary", ["scalar", "vectorized", "replay"])
def test_every_public_path_to_the_bankruptcy_price_refuses_an_inconsistent_fee(
    boundary, kwargs, match
):
    """Three constructors reach the same formula, and the first fix validated one (#314).

    The scalar env, the vectorized config and ReplayOrderExecutor are each their own
    public boundary. Validating only the scalar left the inversion and the divide-by-zero
    fully reachable through the other two -- fix-the-instance, not the class.
    """
    from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
    from torchtrade.envs.offline.vectorized_sequential import (
        VectorizedSequentialTradingEnvConfig,
    )

    with pytest.raises(ValueError, match=match):
        common = dict(time_frames=["1Minute"], window_sizes=[10],
                      execute_on="1Minute")
        if boundary == "scalar":
            idx = pd.date_range("2024-01-01", periods=600, freq="1min")
            df = pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 100.1,
                               "low": 99.9, "close": 100.0, "volume": 10.0})
            SequentialTradingEnv(
                df, SequentialTradingEnvConfig(symbol="X", **common, **kwargs)
            )
        elif boundary == "vectorized":
            VectorizedSequentialTradingEnvConfig(num_envs=1, **common, **kwargs)
        else:
            ReplayOrderExecutor(initial_balance=1000.0, **kwargs)
