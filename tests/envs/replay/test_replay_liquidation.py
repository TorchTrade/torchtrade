"""#269: the replay executor had no liquidation, and reported liquidation_price=0.0."""

import pytest

from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _long_at(entry, leverage, balance=1000.0):
    ex = ReplayOrderExecutor(initial_balance=balance, leverage=leverage)
    ex.current_price = entry
    ex.trade(side="buy", quantity=(balance * leverage) / entry)
    return ex


def test_a_leveraged_replay_position_liquidates(leverage=5):
    """Equity went to -1000 at 5x and then FULLY RECOVERED when price came back.

    The offline env on identical input liquidates and terminates, so any leveraged replay
    evaluation was unboundedly optimistic -- the one number a backtest exists to produce.
    """
    ex = _long_at(40000.0, leverage)
    ex.advance_bar({"open": 30000.0, "high": 30000.0, "low": 30000.0, "close": 30000.0})

    assert ex.position_qty == 0, "a 25% adverse move did not liquidate"
    assert ex.balance >= 0.0, "isolated margin cannot lose more than the margin posted"


@pytest.mark.parametrize("leverage,positive", [(1, False), (10, True)])
def test_the_reported_liquidation_price_is_real(leverage, positive):
    """Hardcoded 0.0 made futures_live_base fail open, so account_state[5] read 1.0 for
    EVERY position -- a levered position shown to the policy as maximally far from
    liquidation. CLAUDE.md invariant 3."""
    ex = _long_at(40000.0, leverage)
    reported = ex.get_status()["position_status"].liquidation_price

    assert (reported > 0) is positive
    if positive:
        assert 0 < reported < 40000.0, "a long is liquidated below its entry"


def test_liquidation_is_checked_without_any_bracket_set():
    """The old guard returned early when sl and tp were both 0, so a position with no
    bracket -- the default -- could never liquidate at all."""
    ex = _long_at(40000.0, 10)
    assert ex.sl_price == 0 and ex.tp_price == 0

    ex.advance_bar({"open": 20000.0, "high": 20000.0, "low": 20000.0, "close": 20000.0})
    assert ex.position_qty == 0


@pytest.mark.parametrize("open_price,low,expect_stop", [
    (99.0, 88.0, True),
    (89.0, 88.0, False),
], ids=["stop-crossed-first", "opened-past-liquidation"])
def test_a_nearer_stop_beats_liquidation_exactly_as_offline_decides(open_price, low, expect_stop):
    """#300, which superseded #299's ordering -- and which replay originally ignored.

    A stop sits on the SAME side of entry as liquidation, so price cannot reach the
    further level without crossing the nearer one. Liquidating anyway is not pessimism,
    it is an outcome the data contradicts: at 10x this leaves 400 where the stop leaves
    5000. The exception is a bar that OPENED past liquidation -- nothing was crossed on
    the way there.
    """
    ex = ReplayOrderExecutor(initial_balance=10000.0, leverage=10)
    ex.current_price = 100.0
    ex.trade(side="buy", quantity=(10000.0 * 10) / 100.0, stop_loss=95.0)
    assert ex.sl_price == 95.0 and ex.liquidation_price < 95.0

    ex.advance_bar({"open": open_price, "high": open_price, "low": low, "close": low})

    assert ex.position_qty == 0
    if expect_stop:
        assert ex.balance > 4000.0, "the stop was crossed first; liquidating loses 10x more"
    else:
        # Pinned, not bounded: `balance < 1000` cannot tell the correct outcome from a
        # -1000 that breaches the margin cap entirely, and a mutation producing exactly
        # that survived this row. The number is 0 rather than the old 400 because the
        # bar GAPPED past liquidation: 400 was the value of booking at a liquidation
        # price this bar never offered, and offline stopped doing that in #314.
        assert ex.balance == pytest.approx(0.0, abs=1e-6), (
            "the bar opened past liquidation; the loss is capped at the posted margin"
        )


def test_a_liquidation_books_at_the_cap_not_at_the_gap():
    """Isolated margin caps the trader's loss at the posted margin -- the venue's
    insurance fund absorbs the gap. Filling at the gapped open instead lost 4x the margin
    posted and silently ate unrelated free balance, which the offline env never does."""
    ex = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    ex.current_price = 100.0
    ex.trade(side="buy", quantity=50.0)
    posted_margin = 50.0 * 100.0 / 5

    ex.advance_bar({"open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0})

    lost = 10000.0 - ex.balance
    assert lost <= posted_margin * 1.01, (
        f"lost {lost:.2f} against {posted_margin:.2f} posted -- the cap was breached"
    )


def test_a_short_liquidates_too():
    """Disabling the short branch entirely passed the whole replay suite.

    Every test drove a long, so a short at high leverage would never liquidate in replay
    while the offline env liquidates it -- #269's unboundedly-optimistic backtest, alive
    for one side only, which is harder to notice than for both.
    """
    ex = ReplayOrderExecutor(initial_balance=1000.0, leverage=10)
    ex.current_price = 40000.0
    ex.trade(side="sell", quantity=(1000.0 * 10) / 40000.0)
    assert ex.liquidation_price > 40000.0, "a short is liquidated ABOVE its entry"

    # NON-FLAT on purpose: every short bar here used to be open==high==low==close, so
    # reading `low` where the short branch needs `high` was invisible -- that mutation
    # survived all eleven tests in this file.
    ex.advance_bar({"open": 41000.0, "high": 50000.0, "low": 40500.0, "close": 41000.0})
    assert ex.position_qty == 0, "a short did not liquidate on a bar whose HIGH crossed"


def test_the_default_maintenance_rate_is_pinned_numerically():
    """A 20x error in DEFAULT_MAINTENANCE_MARGIN_RATE passed every test in the repo.

    The loose `0 < liq < entry` bound tolerates it, the live/offline comparison tests
    derive both sides from the same constant, and the margin-accounting test zeroes the
    rate out to simplify its arithmetic. So the number itself was unguarded, while it
    shifts every leveraged liquidation price and every account_state[5] in the system.
    """
    from torchtrade.envs.utils.liquidation import DEFAULT_MAINTENANCE_MARGIN_RATE

    assert DEFAULT_MAINTENANCE_MARGIN_RATE == 0.004
    ex = _long_at(40000.0, 10)
    assert ex.liquidation_price == pytest.approx(40000.0 * (1 - 1 / 10 + 0.004))


def test_the_maintenance_rate_follows_the_configured_one():
    """Hardcoded before: a backtest configured off the default liquidated at one price
    offline and another in replay -- the divergence liquidation.py exists to prevent."""
    ex = ReplayOrderExecutor(initial_balance=1000.0, leverage=10, maintenance_margin_rate=0.05)
    ex.current_price = 40000.0
    ex.trade(side="buy", quantity=0.25)

    assert ex.liquidation_price == pytest.approx(40000.0 * (1 - 1 / 10 + 0.05))


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -0.5, 1.5],
                         ids=["nan", "inf", "negative", "above-one"])
def test_an_unusable_maintenance_rate_is_refused_at_the_boundary(rate):
    """Invariant 4, on the parameter this PR added.

    NaN made the liquidation price NaN, so every comparison against it was False and a 5x
    position survived a 50% adverse move -- #269 silently re-armed by its own fix. A
    negative rate moved the price the wrong side of entry. Both offline configs already
    raise in __post_init__; replay accepted all four.
    """
    with pytest.raises(ValueError, match="maintenance_margin_rate"):
        ReplayOrderExecutor(initial_balance=1000.0, leverage=5, maintenance_margin_rate=rate)


@pytest.mark.parametrize("quantity", [float("nan"), float("inf")])
def test_an_unusable_quantity_never_opens_a_position(quantity):
    """`quantity <= 0` is transparent to NaN, and inf defeats the balance check through
    `inf * 0.0 = nan`. A NaN position was handed to the policy as a SHORT with a real
    liquidation price, and advance_bar could never close it -- every NaN comparison is
    False -- so it persisted for the rest of the episode against a NaN balance."""
    ex = ReplayOrderExecutor(initial_balance=1000.0, leverage=5)
    ex.current_price = 100.0

    with pytest.raises(ValueError, match="quantity"):
        ex.trade(side="buy", quantity=quantity)
    assert ex.position_qty == 0


@pytest.mark.parametrize("leverage,fee", [(10, 0.001), (5, 0.002), (20, 0.0005)],
                         ids=["10x", "5x", "20x"])
def test_a_gapped_liquidation_never_leaves_negative_equity(leverage, fee):
    """Every other liquidation test here runs at the default transaction_fee=0.0 (#314).

    At fee=0 charging the exit fee out of the margin and charging it on top are
    indistinguishable, so the whole file was blind to the difference. Charging it on top
    overdrew the account by the fee on every gapped liquidation -- and unlike the offline
    engine, ReplayOrderExecutor has no floor, so it simply ended with negative equity
    (-466.67 at 50x). The bankruptcy price is net of the fee instead: the venue takes it
    out of the margin, which is what the maintenance buffer is for.
    """
    ex = ReplayOrderExecutor(initial_balance=1000.0, leverage=leverage, transaction_fee=fee)
    ex.advance_bar({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    # All-in: margin + entry fee == the whole balance, so the margin is the ONLY
    # cushion. Any leftover cash absorbs the overdraft and hides the bug.
    ex.trade("buy", 1000.0 / (100.0 / leverage + 100.0 * fee))
    ex.advance_bar({"open": 70.0, "high": 70.0, "low": 70.0, "close": 70.0})

    assert ex.position_qty == 0, "a bar opening 30% down must liquidate a leveraged long"
    # Tolerance is float noise around an exact zero, not slack: all-in means the loss
    # consumes precisely the margin. The bug this pins overdrew by the FEE, which is
    # 0.05-0.2% of notional -- orders of magnitude above 1e-9.
    assert ex.balance >= -1e-9, (
        f"the exit fee came out of the margin, so equity cannot go negative -- got "
        f"{ex.balance}"
    )
