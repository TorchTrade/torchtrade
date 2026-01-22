"""
Unit tests for AlpacaTorchTradingEnv (TorchRL-style environment).

Tests environment initialization, reset, step, and trading mechanics using mock clients.
Inherits common tests from BaseEnvTests.
"""

import pytest
import numpy as np
import torch
from tensordict import TensorDict

from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.live.alpaca.order_executor import TradeMode
from tests.mocks.alpaca import MockObserver, MockTrader
from tests.envs.base_exchange_tests import BaseEnvTests


# Fixtures for mocks
@pytest.fixture
def mock_observer():
    """Create a mock observer."""
    return MockObserver(window_sizes=[10], num_features=4)


@pytest.fixture
def mock_trader():
    """Create a mock trader."""
    return MockTrader(initial_cash=10000.0, current_price=100000.0)


class TestAlpacaTorchTradingEnv(BaseEnvTests):
    """Tests for AlpacaTorchTradingEnv - inherits common tests from base."""

    def create_env(self, config, observer, trader):
        """Create an AlpacaTorchTradingEnv instance."""
        env = AlpacaTorchTradingEnv(
            config=config,
            observer=observer,
            trader=trader,
        )
        # Patch wait method to avoid delays
        env._wait_for_next_timestamp = lambda: None
        return env

    def create_config(self, **kwargs):
        """Create an AlpacaTradingEnvConfig instance."""
        return AlpacaTradingEnvConfig(
            symbol=kwargs.get('symbol', 'BTC/USD'),
            window_sizes=kwargs.get('window_sizes', [10]),
            paper=kwargs.get('paper', True),
            done_on_bankruptcy=kwargs.get('done_on_bankruptcy', False),
            bankrupt_threshold=kwargs.get('bankrupt_threshold', 0.1),
            reward_scaling=kwargs.get('reward_scaling', 1.0),
            trade_mode=kwargs.get('trade_mode', TradeMode.NOTIONAL),
            include_base_features=kwargs.get('include_base_features', False),
            seed=kwargs.get('seed'),
            time_frames=kwargs.get('time_frames'),
        )

    # Alpaca-specific tests

    def test_action_spec_three_actions(self, mock_observer, mock_trader):
        """Test that action spec has 3 actions (sell, hold, buy) by default."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # Default action_levels is [-1.0, 0.0, 1.0]
        assert env.action_spec.n == 3  # sell, hold, buy

    def test_account_state_has_7_elements(self, mock_observer, mock_trader):
        """Test that Alpaca account state has exactly 7 elements."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()
        account_state = td[env.account_state_key]

        # [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
        assert account_state.shape == (7,)

    def test_initial_portfolio_value(self, mock_observer):
        """Test that initial portfolio value is set correctly."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])
        trader = MockTrader(initial_cash=25000.0)

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=trader,
        )

        assert env.initial_portfolio_value == 25000.0

    def test_multiple_timeframes(self, mock_trader):
        """Test initialization with multiple timeframes."""
        config = self.create_config(
            symbol="BTC/USD",
            time_frames=["1Min", "5Min"],
            window_sizes=[10, 20],
        )

        observer = MockObserver(
            window_sizes=[10, 20],
            keys=["1Minute_10", "5Minute_20"],
        )

        env = self.create_env(
            config=config,
            observer=observer,
            trader=mock_trader,
        )

        assert len(env.market_data_keys) == 2

    def test_reset_contains_account_state_7_elements(self, mock_observer, mock_trader):
        """Test that reset returns account state with 7 elements."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()
        assert env.account_state_key in td.keys()
        assert td[env.account_state_key].shape == (7,)

    def test_reset_contains_market_data_shape(self, mock_observer, mock_trader):
        """Test that reset returns market data with correct shape."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()

        market_key = env.market_data_keys[0]
        assert market_key in td.keys()
        assert td[market_key].shape == (10, 4)

    def test_step_buy_action(self, mock_observer, mock_trader):
        """Test buy action (action=2)."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        initial_position = env.position.current_position

        td_in = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out = env._step(td_in)

        assert env.position.current_position == 1

    def test_step_sell_action_with_position(self, mock_observer, mock_trader):
        """Test sell action when holding position."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy first
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy)

        # Sell
        td_sell = TensorDict({"action": torch.tensor(0)}, batch_size=())
        env._step(td_sell)

        assert env.position.current_position == 0

    def test_step_hold_action(self, mock_observer, mock_trader):
        """Test hold action (action=1)."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_in)

        assert env.position.current_position == 0

    def test_step_updates_account_state(self, mock_observer, mock_trader):
        """Test that step updates account state."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out = env._step(td_buy)

        # Account state: [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
        account_state = td_out[env.account_state_key]
        assert account_state[1] > 0  # position_size > 0

    def test_reward_on_hold_no_position(self, mock_observer, mock_trader):
        """Test reward on hold action without position."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10], reward_scaling=1.0)

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        # Reward should be 0 for holding without position
        assert td_out["reward"].item() == 0.0

    def test_buy_decreases_cash(self, mock_observer, mock_trader):
        """Test that buy decreases cash."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        initial_cash = env.trader.cash

        td_in = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_in)

        assert env.trader.cash < initial_cash

    def test_sell_increases_cash(self, mock_observer, mock_trader):
        """Test that sell increases cash."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy first
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy)

        cash_after_buy = env.trader.cash

        # Sell
        td_sell = TensorDict({"action": torch.tensor(0)}, batch_size=())
        env._step(td_sell)

        assert env.trader.cash > cash_after_buy

    def test_buy_when_already_holding_no_trade(self, mock_observer, mock_trader):
        """Test that buy when already holding doesn't execute trade."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # First buy
        td_buy1 = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy1)

        cash_after_first_buy = env.trader.cash

        # Second buy - should not execute
        td_buy2 = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy2)

        assert env.trader.cash == cash_after_first_buy

    def test_holding_time_increases(self, mock_observer, mock_trader):
        """Test that holding time increases while holding position."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out1 = env._step(td_buy)

        # Hold (maintain 100% position)
        td_hold = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out2 = env._step(td_hold)

        # Account state: [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time]
        holding_time_1 = td_out1[env.account_state_key][6].item()
        holding_time_2 = td_out2[env.account_state_key][6].item()

        assert holding_time_2 > holding_time_1

    def test_holding_time_resets_on_sell(self, mock_observer, mock_trader):
        """Test that holding time resets after selling."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy)

        # Hold a few steps
        td_hold = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_hold)
        env._step(td_hold)

        # Sell
        td_sell = TensorDict({"action": torch.tensor(0)}, batch_size=())
        td_out = env._step(td_sell)

        # After selling, holding time should be 0
        holding_time = td_out[env.account_state_key][6].item()
        assert holding_time == 0

    def test_include_base_features(self, mock_observer, mock_trader):
        """Test that base features are included when configured."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
            include_base_features=True,
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()

        assert "base_features" in td.keys()

    def test_close(self, mock_observer, mock_trader):
        """Test that close cleans up resources."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy)

        env.close()

        # After close, position should be closed
        assert env.trader.position_qty == 0.0

    def test_multiple_resets(self, mock_observer, mock_trader):
        """Test that multiple resets work correctly."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        for _ in range(5):
            td = env.reset()
            assert isinstance(td, TensorDict)

    def test_multiple_episodes(self, mock_observer, mock_trader):
        """Test running multiple episodes."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        for episode in range(3):
            env.reset()
            for step in range(5):
                action = step % 3  # Cycle through actions
                td_in = TensorDict({"action": torch.tensor(action)}, batch_size=())
                td_out = env._step(td_in)

                if td_out["done"].item():
                    break

            env.close()

    def test_set_seed(self, mock_observer, mock_trader):
        """Test that setting seed works."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10], seed=42)

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env._set_seed(42)
        # Should not raise

    def test_account_state_after_buy(self, mock_observer, mock_trader):
        """Test account state is correct after buy."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out = env._step(td_buy)

        account_state = td_out[env.account_state_key]

        # cash should be lower, position_size should be > 0
        assert account_state[0].item() < 10000.0  # cash
        assert account_state[1].item() > 0  # position_size
        assert account_state[2].item() > 0  # position_value
        assert account_state[3].item() > 0  # entry_price

    def test_account_state_after_sell(self, mock_observer, mock_trader):
        """Test account state is correct after sell."""
        config = self.create_config(symbol="BTC/USD", window_sizes=[10])

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Buy
        td_buy = TensorDict({"action": torch.tensor(2)}, batch_size=())
        env._step(td_buy)

        # Sell
        td_sell = TensorDict({"action": torch.tensor(0)}, batch_size=())
        td_out = env._step(td_sell)

        account_state = td_out[env.account_state_key]

        # position_size and position_value should be 0
        assert account_state[1].item() == 0  # position_size
        assert account_state[2].item() == 0  # position_value
