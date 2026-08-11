"""
Unit tests for AlpacaSLTPTorchTradingEnv (TorchRL-style environment with SL/TP).

Tests environment initialization, reset, step, and bracket action mapping. Bracket FILL
pricing is not covered here -- that belongs to the order executor and the offline engines.
"""

from unittest.mock import patch

import pytest
import torch
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict

from torchtrade.envs.live.alpaca.env_sltp import (
    AlpacaSLTPTorchTradingEnv,
    AlpacaSLTPTradingEnvConfig,
)
from torchtrade.envs.utils.action_maps import create_alpaca_sltp_action_map as combinatory_action_map
from .mocks import MockObserver, MockTrader


class TestCombinatorActionMap:
    """Tests for action map generation."""

    def test_action_map_basic(self):
        """Test basic action map generation."""
        stoploss_levels = [-0.05, -0.1]
        takeprofit_levels = [0.05, 0.1]

        action_map = combinatory_action_map(stoploss_levels, takeprofit_levels, include_close_action=False)

        # Action 0 should be HOLD
        assert action_map[0] == (None, None)
        # Should have 1 + (2 * 2) = 5 actions (HOLD + 4 SL/TP combinations)
        assert len(action_map) == 5

    def test_action_map_single_level(self):
        """Test action map with single SL/TP level."""
        stoploss_levels = [-0.05]
        takeprofit_levels = [0.1]

        action_map = combinatory_action_map(stoploss_levels, takeprofit_levels, include_close_action=False)

        assert len(action_map) == 2  # HOLD + 1 SL/TP combination
        assert action_map[0] == (None, None)
        assert action_map[1] == (-0.05, 0.1)

    def test_action_map_multiple_levels(self):
        """Test action map with multiple levels."""
        stoploss_levels = [-0.025, -0.05, -0.1]
        takeprofit_levels = [0.05, 0.1, 0.2]

        action_map = combinatory_action_map(stoploss_levels, takeprofit_levels, include_close_action=False)

        # 1 HOLD + 3*3 combinations = 10 actions (no CLOSE)
        assert len(action_map) == 10


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
        assert env.action_map[0] == (None, None)


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

    def test_step_contains_reward_and_done(self, env):
        """A stepped env emits reward and the whole done family (#272).

        Goes through step(), not _step(): truncated is filled by EnvBase from the done
        spec, so asserting it on _step's raw output would pin a hardcoded write rather
        than the spec that makes the collector work (#313).
        """
        td = env.reset()
        td["action"] = torch.tensor(1)
        nxt = env.step(td)["next"]

        for key in ("reward", "done", "terminated", "truncated"):
            assert key in nxt.keys(), f"{key} missing from the emitted step"
        assert nxt["truncated"].dtype is torch.bool

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

        # After close, position should be closed
        assert env.trader.position_qty == 0.0


class TestAlpacaSLTPTradingEnvMultipleEpisodes:
    """Tests for running multiple episodes."""

