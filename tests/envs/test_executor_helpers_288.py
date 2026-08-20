"""One copy of the shared executor helpers, and a guard against re-forking (#288)."""

import dataclasses
import inspect

import pytest

from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass
from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
from torchtrade.envs.live.shared.executor_helpers import (
    ExecutorHelpersMixin,
    TickSizeMixin,
)

EXECUTORS = [BinanceFuturesOrderClass, BitgetFuturesOrderClass,
             BybitFuturesOrderClass, OKXFuturesOrderClass]

# The three that carried the dead `position_side` (#289). bybit and okx never had it.
POSITION_SIDE_FREE = [BinanceFuturesOrderClass, BitgetFuturesOrderClass,
                      ReplayOrderExecutor]


@pytest.mark.parametrize("cls", EXECUTORS, ids=lambda c: c.__name__)
def test_no_executor_re_forks_the_shared_pnl(cls):
    """Three copies of this existed. It feeds `account_state[2]`, which the policy reads
    every step and a reward function sees, so a drifted copy is expensive twice over --
    and this repo has already shipped a fix landing on some exchanges but not others
    three times (lot size #271, full_done_spec #272, hedge-mode surface).

    A copy that has not drifted YET is still a copy.
    """
    assert "_calculate_unrealized_pnl_pct" not in cls.__dict__, (
        f"{cls.__name__} defines its own _calculate_unrealized_pnl_pct; use the shared one"
    )
    assert cls._calculate_unrealized_pnl_pct is ExecutorHelpersMixin._calculate_unrealized_pnl_pct


@pytest.mark.parametrize("cls", [BinanceFuturesOrderClass, BybitFuturesOrderClass,
                                 OKXFuturesOrderClass], ids=lambda c: c.__name__)
def test_tick_rounding_goes_through_the_shared_helper(cls):
    """These three had byte-identical tick arithmetic. bitget is deliberately absent --
    it OVERRIDES with CCXT's `price_to_precision`, a different mechanism with a different
    failure mode, and collapsing it in to shorten the file would be the opposite error.
    """
    assert cls._round_price is TickSizeMixin._round_price


@pytest.mark.parametrize("cls", EXECUTORS, ids=lambda c: c.__name__)
def test_no_executor_re_implements_the_pnl_rule_INLINE(cls):
    """By RULE, not by name -- the name check alone certified the one that was wrong.

    Binance never had a method called `_calculate_unrealized_pnl_pct`; it had the same
    arithmetic inlined in `get_status`, still branching on `qty > 0`. So
    `"_calculate_unrealized_pnl_pct" not in cls.__dict__` passed VACUOUSLY on the single
    exchange still carrying the dust bug, and the dust parametrization below calls the
    mixin directly and never touches an executor. The guard certified the miss.

    A re-inlined copy is spelled `(mark_price - entry_price) / entry_price` or its short
    counterpart; anything computing that outside the mixin is a fourth copy.
    """
    source = inspect.getsource(cls)
    for expression in ("(mark_price - entry_price) / entry_price",
                       "(entry_price - mark_price) / entry_price"):
        assert expression not in source, (
            f"{cls.__name__} recomputes the PnL rule inline instead of calling the "
            f"shared helper -- a copy that is not a method is still a copy"
        )


@pytest.mark.parametrize("cls", [BybitFuturesOrderClass, OKXFuturesOrderClass],
                         ids=lambda c: c.__name__)
def test_price_formatting_goes_through_the_shared_helper(cls):
    """Two verbatim copies. It emits a STRING because the venues parse the wire value and
    `repr` of a rounded float can carry more digits than the tick allows -- so a drifted
    copy is a rejected or silently re-rounded order, not a cosmetic difference."""
    assert cls._format_price is TickSizeMixin._format_price


def test_bitget_keeps_its_ccxt_rounding():
    """Pinned so a later dedup pass does not 'finish the job' by folding it in."""
    assert "price_to_precision" in inspect.getsource(BitgetFuturesOrderClass._round_price)


def test_bitget_does_not_inherit_the_tick_helpers_at_all():
    """It has no `_tick_size`, so those methods would raise on call.

    An override only covers the one you remember to write: `_round_price` had a bitget
    override and `_format_price` did not, so bitget silently inherited a method that
    always AttributeErrors. Not mixing them in cannot be half-done.
    """
    assert not issubclass(BitgetFuturesOrderClass, TickSizeMixin)
    assert not hasattr(BitgetFuturesOrderClass, "_format_price")


@pytest.mark.parametrize("cls", [BybitFuturesOrderClass, OKXFuturesOrderClass],
                         ids=lambda c: c.__name__)
def test_price_formatting_is_not_re_inlined_at_a_call_site(cls):
    """By RULE, not identity -- the identity check is blind to an inlined copy.

    That is the same hole the name-based PnL guard had, and this file already carries
    that lesson: replacing `self._format_price(x)` with an inline
    `f"{rounded:.{self._tick_decimals}f}"` leaves every identity assertion passing.
    Tick-decimal formatting belongs in the mixin and nowhere else.
    """
    source = inspect.getsource(cls)
    assert "_tick_decimals}f}" not in source, (
        f"{cls.__name__} formats a price against _tick_decimals inline instead of "
        f"calling the shared _format_price -- an inlined copy is still a copy"
    )


@pytest.mark.parametrize("qty,expected", [
    (1.0, 0.10),      # long
    (-1.0, -0.10),    # short
    (1e-12, 0.0),     # dust left by a full close is NOT a position
    (-1e-12, 0.0),
    (0.0, 0.0),
])
def test_unrealized_pnl_is_signed_by_direction_and_ignores_dust(qty, expected):
    """The three originals branched on `qty > 0`, so a 1e-12 residual took the SHORT
    branch and reported -10% on a position that does not exist -- into account_state[2],
    which is invariant 1 (the dust rule) in the observation.
    """
    assert ExecutorHelpersMixin()._calculate_unrealized_pnl_pct(qty, 100.0, 110.0) == pytest.approx(expected)


def test_a_zero_entry_price_reports_no_pnl_rather_than_dividing_by_it():
    assert ExecutorHelpersMixin()._calculate_unrealized_pnl_pct(1.0, 0.0, 110.0) == 0.0


@pytest.mark.parametrize("executor", POSITION_SIDE_FREE, ids=lambda c: c.__name__)
def test_the_executor_trade_surface_advertises_nothing_it_cannot_do(executor):
    """`position_side` sat on these and nothing ever set it (#289).

    Hedge mode IS implemented, but through a venue-level `position_mode` config (bybit,
    bitget); this was a second, per-call surface for it, and binance and replay have no
    position-mode concept at all.
    """
    assert list(inspect.signature(executor.close_position).parameters) == ["self"]
    assert "position_side" not in inspect.signature(executor.trade).parameters, (
        f"{executor.__name__}.trade advertises position_side again; it is driven by the "
        f"venue-level position_mode config, and these three have no such config"
    )


POSITION_STATUS_MODULES = [
    ("binance", "torchtrade.envs.live.binance.order_executor"),
    ("bitget", "torchtrade.envs.live.bitget.order_executor"),
    ("bybit", "torchtrade.envs.live.bybit.order_executor"),
    ("okx", "torchtrade.envs.live.okx.order_executor"),
    ("replay", "torchtrade.envs.replay.order_executor"),
]


@pytest.mark.parametrize("venue,module", POSITION_STATUS_MODULES, ids=lambda v: v)
def test_every_venue_shares_one_position_status(venue, module):
    """Identity, not a matching field list -- see `common_types.PositionStatus`.

    A name-and-shape check would have passed on all five copies, so it could never have
    caught what this exists to prevent. Sharing the OBJECT is the property that holds.
    """
    import importlib

    from torchtrade.envs.core.common_types import PositionStatus as Shared

    venue_cls = getattr(importlib.import_module(module), "PositionStatus")
    assert venue_cls is Shared, (
        f"{venue} re-declares PositionStatus instead of sharing core's. A matching field "
        f"list is not enough -- three of the five copies were byte-identical."
    )


def test_alpaca_position_status_is_deliberately_its_own_shape():
    """The one that must NOT be folded, stated rather than left to inference.

    `AlpacaOrderClass.get_status()` builds its status inside `except Exception`, so a
    folded-in TypeError is swallowed and every live position read degrades quietly to
    POSITION_UNKNOWN rather than raising. Four lines that name the invariant are cheaper
    than relying on that failure being legible.
    """
    from torchtrade.envs.live.alpaca.order_executor import PositionStatus as Alpaca
    from torchtrade.envs.core.common_types import PositionStatus as Shared

    assert {f.name for f in dataclasses.fields(Alpaca)} != {
        f.name for f in dataclasses.fields(Shared)}


def test_no_unqualified_margin_enum_is_exported_from_the_live_namespace():
    """Four venues, four DIFFERENT margin enums, one class name.

    The values are API wire strings and deliberately differ: core/binance `ISOLATED`,
    bitget and bybit `isolated`, okx `cross` where the others say `crossed`. So an
    unqualified `MarginMode` at a namespace covering all four sends the wrong case to
    whichever venue it did not come from -- measured, `marginType='isolated'` reaching an
    API that wants `'ISOLATED'`.

    okx was aliased for this reason long before #289; renaming binance's `MarginType` to
    `MarginMode` put a THIRD claimant on the name and made `torchtrade.envs.MarginMode`
    resolve to bybit's. Nothing covered the export surface, so the whole suite passed on
    it. Every venue enum is qualified here now; the bare name belongs to core, which is
    what the offline configs validate against.
    """
    import torchtrade.envs as envs
    import torchtrade.envs.live as live

    assert envs.MarginMode.__module__ == "torchtrade.envs.core.common_types", (
        f"torchtrade.envs.MarginMode resolves to {envs.MarginMode.__module__}; the "
        f"offline configs validate against core's, so a venue enum here rejects them"
    )
    assert not hasattr(live, "MarginMode"), (
        "torchtrade.envs.live exports an unqualified MarginMode; four venues claim that "
        "name with incompatible wire values, so it must be aliased per venue"
    )
    assert not hasattr(live, "PositionMode"), "same, for PositionMode"
    assert live.BybitMarginMode.ISOLATED.value == "isolated"
    assert live.OKXMarginMode.CROSS.value == "cross"
    assert len(live.__all__) == len(set(live.__all__)), (
        f"duplicate names in torchtrade.envs.live.__all__: the second import wins "
        f"silently and the wire value changes with it -- "
        f"{sorted(n for n in live.__all__ if live.__all__.count(n) > 1)}"
    )


@pytest.mark.parametrize("venue", ["binance", "bitget", "bybit", "okx", "alpaca"])
def test_sltp_envs_resolve_through_action_map_not_action_levels(venue):
    """`_resolve_action_level` reads `self.action_levels`, which SLTP envs do not have.

    Routing an SLTP `_step` through it would AttributeError rather than clamp -- and that
    is a real temptation, since the same out-of-range hazard exists there through
    `action_map`. This pins the boundary so the attempt fails by name instead of by
    missing attribute. The clamp-vs-raise question for SLTP is open (#288).
    """
    import importlib

    module = importlib.import_module(f"torchtrade.envs.live.{venue}.env_sltp")
    cls = next(v for k, v in vars(module).items()
               if k.endswith("TorchTradingEnv") and v.__module__ == module.__name__)
    src = inspect.getsource(cls)
    assert "action_map[" in src, f"{venue} SLTP no longer resolves through action_map"
    assert "_resolve_action_level" not in src, (
        f"{venue} SLTP calls _resolve_action_level, which reads action_levels -- SLTP "
        f"envs do not have that attribute, so this AttributeErrors at the first step"
    )
