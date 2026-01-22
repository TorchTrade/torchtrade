"""
Unit tests for AlpacaSLTPTorchTradingEnv (TorchRL-style environment with SL/TP).

Tests environment initialization, reset, step, action mapping, and bracket order mechanics.
"""

import pytest
import numpy as np
import torch
from tensordict import TensorDict

from torchtrade.envs.alpaca.torch_env_sltp import (
    AlpacaSLTPTorchTradingEnv,
    AlpacaSLTPTradingEnvConfig,
)
from torchtrade.envs.action_maps import create_alpaca_sltp_action_map as combinatory_action_map
from torchtrade.envs.alpaca.order_executor import TradeMode
from .mocks import MockObserver, MockTrader, PositionStatus


class MockSLTPTrader(MockTrader):
    """Extended MockTrader that handles bracket orders with SL/TP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_stop_loss = None
        self.active_take_profit = None
        self.bracket_order_active = False

    def trade(
        self,
        side: str,
        amount: float,
        order_type: str = "market",
        take_profit: float = None,
        stop_loss: float = None,
        **kwargs
    ) -> bool:
        result = super().trade(side, amount, order_type, **kwargs)

        if result and side.lower() == "buy" and take_profit and stop_loss:
            self.active_take_profit = take_profit
            self.active_stop_loss = stop_loss
            self.bracket_order_active = True

        return result

    def simulate_price_movement(self, new_price: float):
        """Simulate price movement and check SL/TP triggers."""
        old_price = self.current_price
        self.current_price = new_price

        if self.position_qty > 0:
            self.position_value = self.position_qty * new_price

            # Check if SL or TP triggered
            if self.bracket_order_active:
                if self.active_stop_loss and new_price <= self.active_stop_loss:
                    # Stop loss triggered
                    self._close_position_at_price(self.active_stop_loss)
                    return "stop_loss"
                elif self.active_take_profit and new_price >= self.active_take_profit:
                    # Take profit triggered
                    self._close_position_at_price(self.active_take_profit)
                    return "take_profit"

        return None

    def _close_position_at_price(self, price: float):
        """Close position at specified price (for SL/TP)."""
        sell_value = self.position_qty * price
        self.cash += sell_value
        self.position_qty = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.bracket_order_active = False
        self.active_stop_loss = None
        self.active_take_profit = None


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

        action_map = combinatory_action_map(stoploss_levels, takeprofit_levels)

        # 1 HOLD + 3*3 combinations = 10 actions
        assert len(action_map) == 10


class TestAlpacaSLTPTradingEnvInitialization:
    """Tests for environment initialization."""

    def test_init_with_mocks(self):
        """Test initialization with injected mocks."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            paper=True,
            stoploss_levels=(-0.05, -0.1),
            takeprofit_levels=(0.05, 0.1),
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=10000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert env.config == config
        assert env.observer is mock_observer
        assert env.trader is mock_trader

    def test_action_spec_size(self):
        """Test that action spec has correct size."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.025, -0.05, -0.1),
            takeprofit_levels=(0.05, 0.1, 0.2),
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader()

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # 1 HOLD + 3*3 SL/TP combinations = 10 actions
        assert env.action_spec.n == 10

    def test_observation_spec_account_state(self):
        """Test that observation spec has 7-element account state."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        mock_observer = MockObserver(window_sizes=[10], num_features=4)
        mock_trader = MockSLTPTrader()

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert "account_state" in env.observation_spec.keys()
        # Account state should have 7 elements
        assert env.observation_spec["account_state"].shape == (7,)

    def test_action_map_created(self):
        """Test that action map is correctly created."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader()

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
        """Create an environment with mocks."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=10000.0)

        return AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

    def test_reset_returns_tensordict(self, env):
        """Test that reset returns a TensorDict."""
        td = env.reset()
        assert isinstance(td, TensorDict)

    def test_reset_contains_account_state(self, env):
        """Test that reset returns account state with 7 elements."""
        td = env.reset()

        assert env.account_state_key in td.keys()
        assert td[env.account_state_key].shape == (7,)

    def test_reset_resets_sltp_state(self, env):
        """Test that reset clears active SL/TP levels."""
        env.reset()

        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0
        assert env.position.current_position == 0.0


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
        mock_trader = MockSLTPTrader(initial_cash=10000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        return env

    def test_step_returns_tensordict(self, env):
        """Test that step returns a TensorDict."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(0)}, batch_size=())
        td_out = env._step(td_in)

        assert isinstance(td_out, TensorDict)

    def test_step_hold_action(self, env):
        """Test hold action (action=0)."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(0)}, batch_size=())
        env._step(td_in)

        assert env.position.current_position == 0

    def test_step_buy_with_sltp(self, env):
        """Test buy action with SL/TP (action > 0)."""
        env.reset()

        # Action 1 maps to first SL/TP combination
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        assert env.position.current_position == 1

    def test_step_contains_reward_and_done(self, env):
        """Test that step returns reward and done flags."""
        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()
        assert "terminated" in td_out.keys()
        assert "truncated" in td_out.keys()

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


class TestAlpacaSLTPTradingEnvAccountState:
    """Tests for account state observation."""

    @pytest.fixture
    def env(self):
        """Create an environment with mocks."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=10000.0, current_price=100000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        return env

    def test_account_state_has_7_elements(self, env):
        """Test account state has 7 elements."""
        td = env.reset()
        account_state = td[env.account_state_key]

        # [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
        assert account_state.shape == (7,)

    def test_account_state_after_buy(self, env):
        """Test account state after buy action."""
        env.reset()

        td_buy = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_buy)

        account_state = td_out[env.account_state_key]

        # cash should be lower, position_size should be > 0
        assert account_state[0].item() < 10000.0  # cash
        assert account_state[1].item() > 0  # position_size
        assert account_state[2].item() > 0  # position_value
        assert account_state[3].item() > 0  # entry_price


class TestAlpacaSLTPTradingEnvTermination:
    """Tests for episode termination."""

    def test_bankruptcy_termination(self):
        """Test that bankruptcy triggers termination."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
            done_on_bankruptcy=True,
            bankrupt_threshold=0.1,
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=500.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env.initial_portfolio_value = 10000.0
        env._wait_for_next_timestamp = lambda: None

        env.reset()
        td_in = TensorDict({"action": torch.tensor(0)}, batch_size=())
        td_out = env._step(td_in)

        # Portfolio value (500) is below 10% of initial (1000)
        assert td_out["done"].item() is True


class TestAlpacaSLTPTradingEnvClose:
    """Tests for environment cleanup."""

    def test_close(self):
        """Test that close cleans up resources."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=10000.0)

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

    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        config = AlpacaSLTPTradingEnvConfig(
            symbol="BTC/USD",
            window_sizes=[10],
        )
        mock_observer = MockObserver(window_sizes=[10])
        mock_trader = MockSLTPTrader(initial_cash=10000.0)

        env = AlpacaSLTPTorchTradingEnv(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )
        env._wait_for_next_timestamp = lambda: None

        for _ in range(5):
            td = env.reset()
            assert isinstance(td, TensorDict)
            assert env.active_stop_loss == 0.0
            assert env.active_take_profit == 0.0
