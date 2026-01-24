"""Tests for Alpaca environment price fetching fallback chain (Priority 9 gap)."""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from tensordict import TensorDict

from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from tests.envs.alpaca.mocks import (
    MockTradingClient,
    MockTrader,
    MockObserver,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env_config():
    """Create standard environment configuration."""
    return AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=["1Min"],
        window_sizes=[10],
        execute_on="1Min",
    )


@pytest.fixture
def mock_observer():
    """Create mock observer."""
    return MockObserver(window_sizes=[10])


@pytest.fixture
def mock_trader():
    """Create mock trader."""
    return MockTrader(initial_cash=10000.0)


@pytest.fixture
def env(env_config, mock_observer, mock_trader):
    """Create Alpaca environment with mocks."""
    env = AlpacaTorchTradingEnv(
        config=env_config,
        observer=mock_observer,
        trader=mock_trader,
    )

    return env


# ============================================================================
# Tests for _get_current_price() Fallback Chain
# ============================================================================


class TestGetCurrentPriceFallbackChain:
    """Tests for _get_current_price() method and its 3-level fallback chain."""

    def test_get_current_price_from_position_status_primary_path(self, env):
        """Test primary path: price from position_status when provided."""
        # Create mock position status with price
        mock_position = MagicMock()
        mock_position.current_price = 50000.0

        price = env._get_current_price(position_status=mock_position)

        assert price == 50000.0

    def test_get_current_price_fetches_position_status_when_none(self, env):
        """Test that position_status is fetched if not provided."""
        # Mock trader to return position status with price
        mock_position = MagicMock()
        mock_position.current_price = 51000.0

        env.trader.get_status = Mock(return_value={
            "position_status": mock_position
        })

        price = env._get_current_price(position_status=None)

        assert price == 51000.0
        env.trader.get_status.assert_called_once()

    def test_get_current_price_fallback1_trader_attribute(self, env):
        """Test fallback 1: Uses trader.current_price attribute when position_status unavailable."""
        # Mock trader with no position but has current_price attribute (for mocks)
        env.trader.get_status = Mock(return_value={"position_status": None})
        env.trader.current_price = 49000.0

        price = env._get_current_price(position_status=None)

        assert price == 49000.0

    def test_get_current_price_fallback2_observer_fetch(self, env):
        """Test fallback 2: Fetches from observer.get_current_price() when both previous sources fail."""
        # Mock trader with no position and no current_price attribute
        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        # Mock observer to return price
        env.observer.get_current_price = Mock(return_value=48000.0)

        price = env._get_current_price(position_status=None)

        assert price == 48000.0
        env.observer.get_current_price.assert_called_once()

    def test_get_current_price_all_sources_fail_returns_zero(self, env):
        """Test that when all 3 fallbacks fail, returns 0.0."""
        # Mock everything to fail
        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')
        env.observer.get_current_price = Mock(side_effect=ValueError("No data"))

        price = env._get_current_price(position_status=None)

        assert price == 0.0

    def test_get_current_price_position_status_zero_price_tries_fallback1(self, env):
        """Test that position_status with price <= 0 triggers fallback 1."""
        # Position exists but has price of 0
        mock_position = MagicMock()
        mock_position.current_price = 0.0

        env.trader.current_price = 52000.0

        price = env._get_current_price(position_status=mock_position)

        # Should use fallback 1 (trader attribute)
        assert price == 52000.0

    def test_get_current_price_negative_price_tries_fallback(self, env):
        """Test that negative price triggers fallback chain."""
        mock_position = MagicMock()
        mock_position.current_price = -1.0  # Invalid price

        env.trader.current_price = 53000.0

        price = env._get_current_price(position_status=mock_position)

        assert price == 53000.0


# ============================================================================
# Tests for Initial Trade Scenario (Bug Fix)
# ============================================================================


class TestInitialTradeWithoutPosition:
    """Tests for the bug fix: price fetching when no position exists (initial trade)."""

    def test_step_first_buy_no_position_fetches_price(self, env):
        """Test that first buy action correctly fetches price when no position exists."""
        env.reset()

        # Mock wait to avoid time.sleep in tests
        env._wait_for_next_timestamp = Mock()

        # Ensure no position exists
        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        # Remove trader.current_price attribute to force fallback 2
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        # Mock observer to return price
        env.observer.get_current_price = Mock(return_value=50000.0)

        # Action 2 = fully invested (buy)
        td_in = TensorDict({"action": torch.tensor(2)}, batch_size=())

        # Should not crash, should fetch price from observer
        td_out = env._step(td_in)

        # Verify observer was called
        env.observer.get_current_price.assert_called()

        # Verify step completed without error
        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()

    def test_execute_fractional_action_no_position_uses_observer(self, env):
        """Test that _execute_fractional_action fetches price from observer when no position."""
        env.reset()

        # No position
        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(return_value=48000.0)

        # Call the method directly
        trade_info = env._execute_fractional_action(action_value=0.5)

        # Should have called observer
        env.observer.get_current_price.assert_called()

    def test_get_current_price_logs_info_when_using_observer(self, env, caplog):
        """Test that info is logged when fetching from observer."""
        import logging
        caplog.set_level(logging.INFO)

        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(return_value=49000.0)

        price = env._get_current_price()

        assert price == 49000.0
        # Check that info log was emitted
        assert any("Fetched current price from market data" in record.message for record in caplog.records)

    def test_get_current_price_logs_warning_on_observer_exception(self, env, caplog):
        """Test that warning is logged when observer fetch fails."""
        import logging
        caplog.set_level(logging.WARNING)

        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(side_effect=Exception("API error"))

        price = env._get_current_price()

        assert price == 0.0
        # Check that warning log was emitted
        assert any("Could not fetch current price" in record.message for record in caplog.records)


# ============================================================================
# Tests for Trade Execution Price Validation
# ============================================================================


class TestTradeExecutionPriceValidation:
    """Tests for price validation in trade execution (Priority 6 gap)."""

    def test_execute_trade_rejects_when_no_price_available(self, env, caplog):
        """Test that trade is rejected when price cannot be determined."""
        import logging
        caplog.set_level(logging.ERROR)

        env.reset()

        # Force all price sources to fail
        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(side_effect=Exception("No data"))

        # Try to execute trade
        trade_info = env._execute_fractional_action(action_value=1.0)

        # Trade should not execute
        assert trade_info["executed"] is False
        assert trade_info["amount"] == 0
        assert trade_info["side"] is None

        # Error should be logged
        assert any("Cannot execute trade: invalid or missing price data" in record.message
                  for record in caplog.records)

    def test_execute_trade_proceeds_when_price_available(self, env):
        """Test that trade proceeds normally when price is available."""
        env.reset()

        # Mock position status with valid price
        mock_position = MagicMock()
        mock_position.qty = 0.0
        mock_position.current_price = 50000.0

        env.trader.get_status = Mock(return_value={
            "position_status": mock_position,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        # Mock trade execution to succeed
        env.trader.trade = Mock(return_value=True)

        trade_info = env._execute_fractional_action(action_value=1.0)

        # Trade should attempt to execute
        env.trader.trade.assert_called()

    def test_step_with_zero_price_does_not_crash(self, env):
        """Test that step with price=0 doesn't crash, just skips trade."""
        env.reset()

        # Mock wait to avoid time.sleep in tests
        env._wait_for_next_timestamp = Mock()

        # All price sources return 0
        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(return_value=0.0)

        # Action 2 = buy
        td_in = TensorDict({"action": torch.tensor(2)}, batch_size=())

        # Should not crash
        td_out = env._step(td_in)

        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()


# ============================================================================
# Tests for Exception Handling
# ============================================================================


class TestPriceFetchingExceptionHandling:
    """Tests for graceful exception handling in price fetching."""

    def test_observer_get_current_price_network_error(self, env):
        """Test graceful handling when observer raises network error."""
        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(side_effect=ConnectionError("Network timeout"))

        # Should not crash, returns 0
        price = env._get_current_price()
        assert price == 0.0

    def test_observer_get_current_price_value_error(self, env):
        """Test graceful handling when observer raises ValueError."""
        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(side_effect=ValueError("Invalid data"))

        price = env._get_current_price()
        assert price == 0.0

    def test_observer_get_current_price_generic_exception(self, env):
        """Test graceful handling of generic exceptions."""
        env.trader.get_status = Mock(return_value={"position_status": None})
        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(side_effect=RuntimeError("Unexpected error"))

        price = env._get_current_price()
        assert price == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestPriceFetchingIntegration:
    """Integration tests for price fetching in realistic scenarios."""

    def test_full_step_cycle_with_position(self, env):
        """Test full step cycle when position exists (normal trading)."""
        env.reset()

        # Mock wait to avoid time.sleep in tests
        env._wait_for_next_timestamp = Mock()

        # Position exists with valid price
        mock_position = MagicMock()
        mock_position.qty = 0.5
        mock_position.market_value = 25000.0
        mock_position.current_price = 50000.0
        mock_position.avg_entry_price = 48000.0
        mock_position.unrealized_plpc = 0.04

        env.trader.get_status = Mock(return_value={
            "position_status": mock_position,
            "account_status": MagicMock(cash=5000.0, portfolio_value=30000.0)
        })

        # Hold action
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        # Should complete successfully
        assert "reward" in td_out.keys()
        assert "done" in td_out.keys()

    def test_full_step_cycle_without_position_first_trade(self, env):
        """Test full step cycle for first trade when no position exists."""
        env.reset()

        # Mock wait to avoid time.sleep in tests
        env._wait_for_next_timestamp = Mock()

        # No position, need to fetch price from observer
        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": MagicMock(cash=10000.0, portfolio_value=10000.0)
        })

        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(return_value=50000.0)
        env.trader.trade = Mock(return_value=True)

        # Buy action
        td_in = TensorDict({"action": torch.tensor(2)}, batch_size=())
        td_out = env._step(td_in)

        # Observer should have been called
        env.observer.get_current_price.assert_called()

        # Step should complete
        assert "reward" in td_out.keys()

    def test_transition_from_position_to_no_position(self, env):
        """Test price fetching when transitioning from having position to no position."""
        env.reset()

        # Mock wait to avoid time.sleep in tests
        env._wait_for_next_timestamp = Mock()

        # Step 1: Have position
        mock_position = MagicMock()
        mock_position.qty = 0.5
        mock_position.current_price = 51000.0
        mock_position.market_value = 25500.0

        # Create account status mock that returns proper floats
        mock_account = MagicMock()
        mock_account.cash = 5000.0
        mock_account.portfolio_value = 30500.0

        # Mock trader.client.get_account() for portfolio value calculation
        env.trader.client.get_account = Mock(return_value=mock_account)

        env.trader.get_status = Mock(return_value={
            "position_status": mock_position,
            "account_status": mock_account
        })

        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        env._step(td_in)

        # Step 2: Sold everything, no position
        mock_account2 = MagicMock()
        mock_account2.cash = 30500.0
        mock_account2.portfolio_value = 30500.0

        # Mock trader.client.get_account() for portfolio value calculation
        env.trader.client.get_account = Mock(return_value=mock_account2)

        env.trader.get_status = Mock(return_value={
            "position_status": None,
            "account_status": mock_account2
        })

        if hasattr(env.trader, 'current_price'):
            delattr(env.trader, 'current_price')

        env.observer.get_current_price = Mock(return_value=51500.0)

        # Should still work, fetching from observer
        env._step(td_in)
        env.observer.get_current_price.assert_called()
