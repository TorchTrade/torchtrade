"""
Unit tests for AlpacaSLTPTorchTradingEnv (TorchRL-style environment with SL/TP).

Tests environment initialization, reset, step, and bracket action mapping. Bracket FILL
pricing is not covered here -- that belongs to the order executor and the offline engines.
"""

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import torch
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict

from tests.envs.base_exchange_tests import (
    INVALID_ACTIONS,
    assert_an_invalid_action_cannot_move_an_open_position,
    assert_an_invalid_action_raises_before_trading,
)

from torchtrade.envs.live.alpaca.env_sltp import (
    AlpacaSLTPTorchTradingEnv,
    AlpacaSLTPTradingEnvConfig,
)
from .mocks import MockObserver, MockTrader


def _sltp_env(*, include_close_action=False, sl=(-0.05,), tp=(0.1,)):
    """A constructed env, because the map under test is the one ALPACA asks for.

    A helper that called the shared builder directly instead asserted only that the
    builder honours its own arguments: flipping alpaca's `include_short_positions` to
    True left it green.
    """
    env = AlpacaSLTPTorchTradingEnv(
        config=AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD", window_sizes=[10],
            stoploss_levels=sl, takeprofit_levels=tp,
            include_close_action=include_close_action,
        ),
        observer=MockObserver(window_sizes=[10]),
        trader=MockTrader(initial_cash=10000.0),
    )
    env._wait_for_next_timestamp = lambda: None
    return env


class TestCombinatorActionMap:
    """The action map alpaca builds. Size arithmetic belongs to the shared builder and is
    tested against it in tests/envs/binance/test_torch_env_futures_sltp.py.
    """

    @pytest.mark.parametrize("include_close,expected", [
        (False, {0: (None, None, None), 1: ("long", -0.05, 0.1)}),
        (True, {0: (None, None, None), 1: ("close", None, None), 2: ("long", -0.05, 0.1)}),
    ])
    def test_the_env_builds_a_long_only_map_that_keeps_the_close_marker(self, include_close, expected):
        """No 'short' entry, and CLOSE is a real action rather than a second HOLD.

        The wrapper this replaced returned (sl, tp) 2-tuples, so the "close" marker had
        nowhere to live and was dropped: `include_close_action=True` produced
        `m[0] == m[1] == (None, None)`, widening the action space by a slot that
        provably could not do anything (#418).
        """
        env = _sltp_env(include_close_action=include_close)
        assert env.action_map == expected
        assert env.action_spec.n == len(expected)


class TestAlpacaSLTPTradingEnvInitialization:
    """Tests for environment initialization."""

    def test_action_spec_size(self):
        """Test that action spec has correct size."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.025, -0.05, -0.1),
            takeprofit_levels=(0.05, 0.1, 0.2),
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader()

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # 1 HOLD + 3*3 SL/TP combinations = 10 actions
        assert env.action_spec.n == 10

    def test_action_map_created(self):
        """Test that action map is correctly created."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader()

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # 1 HOLD + 2*2 combinations = 5 actions
        assert len(env.action_map) == 5
        assert env.action_map[0] == (None, None, None)
        assert env.action_map[1] == ("long", -0.02, 0.03)


class TestAlpacaSLTPTradingEnvReset:
    """Tests for environment reset."""

    @pytest.fixture
    def env(self):
        """Create an environment with mocks.

        Deliberately does NOT stub _wait_for_next_timestamp: an instance attribute would
        shadow the class-level patch in test_check_env_specs_passes, leaving that test
        green with its patch deleted and hollowing out the #272 guard.
        """
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader(initial_cash=10000.0)

        return AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

    def test_step_emits_the_whole_done_family(self, env):
        from tests.envs.base_exchange_tests import (
            assert_the_step_emits_the_whole_done_family as assert_done_family,
        )
        assert_done_family(env)

    def test_check_env_specs_passes(self, env):
        """check_env_specs compares the emitted step against every declared spec;
        the done family comes from the spec on both sides here, so a narrowed done spec
        is NOT what this catches -- see assert_the_step_emits_the_whole_done_family."""
        with patch.object(type(env), "_wait_for_next_timestamp"):
            check_env_specs(env)

    @pytest.mark.parametrize("close,harm", [
        (float("nan"), "both bracket legs go to the venue as NaN, unlogged"),
        (0.0, "both legs price at zero"),
        (-50000.0, "the stop lands ABOVE the take-profit -- an inverted bracket"),
    ], ids=["nan", "zero", "negative"])
    def test_a_bracket_is_never_priced_off_an_unusable_close(self, env, close, harm):
        """#347 swept alpaca/env.py and never reached alpaca/env_sltp.py.

        The entry is a full-balance market BUY in every case; if the venue takes it and
        rejects the legs, bracket_status zeroes active_stop_loss/active_take_profit and the
        position sits unprotected in an env whose ONLY exit is SL/TP. And {harm}.
        """
        env._wait_for_next_timestamp = lambda: None
        env.reset()
        env.observer.get_observations = lambda return_base_ohlc=False: {
            "1Minute_10": np.zeros((10, 4), dtype=np.float32),
            **({"base_features": np.full((10, 4), close, dtype=np.float32),
                "base_timestamps": np.arange(10)} if return_base_ohlc else {}),
        }

        with pytest.raises(ValueError, match="unusable close price"):
            env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        assert env.trader.position_qty == 0, "no entry may be opened on an unusable close"

    @pytest.mark.parametrize("cash", [float("nan"), float("inf"), -250.0],
                             ids=["nan", "inf", "negative"])
    def test_a_bracket_is_never_sized_off_an_unusable_cash_balance(self, env, cash):
        """#347: alpaca SLTP sizes off self.balance, written raw in _reset.

        _get_portfolio_value DOES validate -- but it is a SEPARATE fetch, and this env's
        _step calls it AFTER _execute_trade_if_needed. So the venue could report healthy
        cash on that later call while the NaN captured at reset had already sized a
        full-balance bracket buy: the raise landed after the order was live.
        """
        env._wait_for_next_timestamp = lambda: None
        env.trader.cash = cash

        with pytest.raises(ValueError, match="unusable cash balance"):
            env.reset()

        assert env.trader.position_qty == 0, "no entry may be sized off an unusable balance"

    def test_reset_clears_a_live_bracket(self, env):
        """Reset must clear SL/TP levels that an episode actually set.

        Asserting this on a virgin env cannot fail -- the levels are already 0.0 from
        __init__, so a no-op _reset_sltp_state passes. The bracket has to be opened first
        for the assertion to mean anything.
        """
        env._wait_for_next_timestamp = lambda: None
        env.reset()
        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))
        assert env.active_stop_loss > 0, "the bracket action should have set the levels"

        env.reset()

        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0


class TestAlpacaSLTPCloseAction:
    """`include_close_action=True` must produce an action that flattens (#418).

    Before the map switched to the shared 3-tuple builder, enabling the flag widened the
    Categorical by one and put `(None, None)` -- HOLD's own tuple -- in the new slot. The
    policy could emit it, and nothing happened. The two that go through `_step` are what
    fail on a map that carries the marker but an executor that ignores it; the rest drive
    `_execute_trade_if_needed` directly to reach branches a map cannot produce.
    """

    @pytest.fixture
    def env(self):
        return _sltp_env(include_close_action=True, sl=(-0.02,), tp=(0.03,))

    def test_close_flattens_an_open_position(self, env):
        """The action the flag adds must reach the venue and leave the account flat."""
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        assert env.trader.position_qty > 0, "setup: the bracket entry must fill"

        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        assert env.trader.position_qty == 0
        assert env.position.current_position == 0
        # Brackets go with the position -- a stale leg would price against a position
        # that no longer exists.
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0
        # The history row, not just the account: recording the CLOSE as a long (1.0) left
        # the position correctly flat and the action trace quietly wrong, and passed the
        # whole alpaca suite. `actions` feeds the reward and every offline read of what
        # the policy did.
        assert env.history.actions[-1] == 0.0

    @pytest.mark.parametrize("outcome", ["refused", "raised"])
    def test_a_close_the_venue_does_not_take_is_not_reported_as_a_close(self, env, outcome):
        """Refused or raised, the position stays and the caller is told (#295).

        Without `success=False` a failed close returns `success=None` -- HOLD's value --
        and the refusal is invisible to the caller.
        """
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        env.trader.close_position = (
            (lambda qty=None: False) if outcome == "refused"
            else MagicMock(side_effect=RuntimeError("venue down"))
        )

        info = env._execute_trade_if_needed(("close", None, None))

        assert info["success"] is False
        assert env.position.current_position == 1, "the position is still at the venue"
        assert env.active_stop_loss != 0.0, "brackets must not be cleared on a failed close"

    def test_a_close_on_a_flat_account_does_not_reach_the_venue(self, env):
        """No position, no order -- and `executed` stays False so it reads as a no-op."""
        env.trader.close_position = MagicMock()

        info = env._execute_trade_if_needed(("close", None, None))

        env.trader.close_position.assert_not_called()
        assert info["executed"] is False and info["closed_position"] is False

    def test_a_long_action_on_top_of_a_short_does_not_buy(self, env):
        """Alpaca equities permit shorts, so a synced -1 is a reachable state.

        The guard used to read `current_position == 1`, which a -1 falls straight through:
        the entry then submitted a full-balance market BUY on top of the short, with
        brackets priced for a long. Nothing covered it.
        """
        env.position.current_position = -1

        info = env._execute_trade_if_needed(("long", -0.02, 0.03))

        assert info["executed"] is False
        assert env.trader.position_qty == 0

    def test_a_side_alpaca_cannot_fill_raises_rather_than_buying(self, env):
        """A "short" tuple would otherwise submit a BUY -- the opposite of the action."""
        with pytest.raises(ValueError, match="long-only"):
            env._execute_trade_if_needed(("short", 0.02, -0.03))
        assert env.trader.position_qty == 0

    def test_close_is_ignored_while_the_position_is_locked(self, env):
        """`lock_position_until_sltp` means SL/TP is the only exit, CLOSE included."""
        env.config.lock_position_until_sltp = True
        env._step(TensorDict({"action": torch.tensor(2)}, batch_size=()))
        qty = env.trader.position_qty

        env._step(TensorDict({"action": torch.tensor(1)}, batch_size=()))

        assert env.trader.position_qty == qty


class TestAlpacaSLTPTradingEnvStep:
    """Tests for environment step."""

    @pytest.fixture
    def env(self):
        """Create an environment with mocks that skips waiting."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader(initial_cash=10000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        return env

    @pytest.mark.parametrize("action", INVALID_ACTIONS)
    def test_an_invalid_action_raises_before_trading(self, env, action):
        """Venue wiring: this `_step` must route through the shared validator (#288)."""
        env.trader.trade = MagicMock(wraps=env.trader.trade)
        assert_an_invalid_action_raises_before_trading(env, action)

    @pytest.mark.parametrize("action", INVALID_ACTIONS)
    def test_an_invalid_action_cannot_move_an_open_position(self, env, action):
        """The expensive direction -- every other case here starts flat (#288)."""
        env.trader.trade = MagicMock(wraps=env.trader.trade)
        assert_an_invalid_action_cannot_move_an_open_position(env, action)

    def test_step_hold_action(self, env):
        """Test hold action (action=0)."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(0)}, batch_size=())
        env._step(td_in)

        assert env.position.current_position == 0

    def test_holding_time_ages_from_one(self, env):
        """holding_time reads 1 on the opening bar, then increments (#49).

        alpaca is spot (no flip), so this open->hold->hold covers this SLTP env's
        _step -> _get_observation(advance_hold=True) aging path directly."""
        env.reset()
        seq = []
        for a in (1, 0, 0):  # open bracket, hold, hold
            td = env._step(TensorDict({"action": torch.tensor(a)}, batch_size=()))
            seq.append(td["account_state"][3].item())
        assert seq == [1.0, 2.0, 3.0]

    def test_step_buy_with_sltp(self, env):
        """Test buy action with SL/TP (action > 0)."""
        env.reset()

        # Action 1 maps to first SL/TP combination
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_in)

        assert env.position.current_position == 1

    def test_step_contains_reward(self, env):
        """The done family is asserted by assert_the_step_emits_the_whole_done_family."""
        td = env.reset()
        td["action"] = torch.tensor(1)
        nxt = env.step(td)["next"]

        assert "reward" in nxt.keys()

    def test_cannot_buy_when_holding(self, env):
        """Test that buying when already holding doesn't execute."""
        env.reset()

        # First buy
        td_buy1 = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_buy1)

        cash_after_buy = env.trader.cash

        # Second buy attempt - should not execute
        td_buy2 = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy2)

        assert env.trader.cash == cash_after_buy


class TestAlpacaSLTPTradingEnvTermination:
    """Tests for episode termination."""

    @pytest.mark.parametrize("done_on_bankruptcy,expected_done", [
        (True, True),    # portfolio collapses below the threshold -> episode terminates
        (False, False),  # same collapse, check disabled -> keep trading
    ], ids=["enabled-terminates", "disabled-keeps-trading"])
    def test_bankruptcy_termination(self, done_on_bankruptcy, expected_done):
        """A collapsed portfolio ends the episode through _step iff done_on_bankruptcy.

        Threshold arithmetic is covered in tests/envs/test_live_env_base.py; the disabled
        case is this file's only guard against a _step that hardcodes done=True.
        """
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            done_on_bankruptcy=done_on_bankruptcy,
            bankrupt_threshold=0.1,
        )
        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=MockObserver(window_sizes=[10]),
            trader=MockTrader(initial_cash=500.0),
        )
        env.initial_portfolio_value = 10000.0  # the 500 cash is below 10% of this
        env._wait_for_next_timestamp = lambda: None

        env.reset()
        next_td = env.step(TensorDict({"action": torch.tensor(0)}, batch_size=()))
        assert next_td["next"]["done"].item() is expected_done


class TestAlpacaSLTPTradingEnvClose:
    """Tests for environment cleanup."""

    def test_close(self):
        """Test that close cleans up resources."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockTrader(initial_cash=10000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_buy)

        env.close()

        # close() cancels orders and LEAVES the position, as all four futures envs do
        # (#289). Flattening on shutdown made close_position_on_init=False pointless --
        # the position it preserved was market-closed the moment the process exited.
        # Call trader.close_position() explicitly if you want it flat.
        assert env.trader.position_qty != 0.0


@pytest.mark.parametrize("sltp", [True, False], ids=["sltp", "plain"])
def test_alpaca_history_records_price_and_position_on_a_flat_bar(sltp):
    """The SLTP env read the price inline, and neither env recorded the position (#290).

    `position_status.current_price if position_status else 0.0` gives 0 on every flat bar,
    where the non-SLTP sibling has a three-tier fallback -- so a reward function reading
    history.base_prices saw zero whenever the agent was out of the market.

    And neither alpaca env passed position= to record_step, so history.positions was all
    zeros while every other exchange recorded it. BOTH envs are covered because the fix is
    per-env: dropping it from the non-SLTP one passed the entire 2139-test suite.
    """
    from torchtrade.envs.live.alpaca.env import (
        AlpacaTorchTradingEnv, AlpacaTradingEnvConfig,
    )

    trader = MockTrader(initial_cash=10000.0)
    observer = MockObserver(window_sizes=[10])
    if sltp:
        env = AlpacaSLTPTorchTradingEnv(
            config=AlpacaSLTPTradingEnvConfig(symbol="BTC/USD", window_sizes=[10]),
            observer=observer, trader=trader,
        )
        open_action, hold_action = 1, 0
    else:
        env = AlpacaTorchTradingEnv(
            config=AlpacaTradingEnvConfig(symbol="BTC/USD", window_sizes=[10]),
            observer=observer, trader=trader,
        )
        open_action, hold_action = 2, 0  # action_levels [0.0, 0.5, 1.0]
    env._wait_for_next_timestamp = lambda: None

    def step(a):
        env._step(TensorDict({"action": torch.tensor(a)}, batch_size=()))

    env.reset()
    step(hold_action)
    # Asserted on the EXCHANGE, not on env.position.position_size: that field is never
    # assigned on any alpaca env, so it reads 0.0 while the exchange holds a position and
    # the guard would pass either way.
    assert trader.get_status()["position_status"] is None, "this cell must be FLAT"
    assert env.history.base_prices[-1] > 0, (
        f"flat bar recorded price {env.history.base_prices[-1]} -- the fallback chain "
        "should have supplied one"
    )

    step(open_action)
    # POST-trade, so the size appears on the bar that opened the position -- not the one
    # after. Recording the entering size against a post-bar price and portfolio value
    # labelled the return with the exposure that did not produce it, and is what offline
    # has always done (#278).
    assert env.history.positions[-1] != 0, (
        "the opening bar must record the position it opened, not the flat size it "
        "entered with"
    )
    # The next bar records where it ENDED, which differs by variant: on the plain env
    # `hold_action` is index 0 of [0.0, 0.5, 1.0], a 0% allocation that exits, so the row
    # reads flat. On the SLTP env index 0 is a no-op and the position exits only via
    # SL/TP, so it is still open. Under the old pre-trade semantics the plain env's
    # closing row claimed the exposure it had just given up.
    step(hold_action)
    if sltp:
        assert env.history.positions[-1] != 0, "the position is still open on this bar"
    else:
        assert env.history.positions[-1] == 0, (
            "the closing bar must record the flat size it ended at, not the one it "
            "entered with"
        )
    env.close()
