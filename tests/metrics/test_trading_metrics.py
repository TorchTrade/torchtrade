"""Tests for trading metrics module."""

import pytest
import torch
import numpy as np

from torchtrade.metrics import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_win_rate,
    compute_portfolio_returns,
)


class TestMaxDrawdown:
    """Test maximum drawdown computation."""

    def test_no_drawdown(self):
        """Test portfolio with no drawdown (monotonically increasing)."""
        portfolio = torch.tensor([100, 110, 120, 130, 140], dtype=torch.float32)
        result = compute_max_drawdown(portfolio)

        assert result['max_drawdown'] == 0.0
        assert result['max_drawdown_duration'] == 0
        assert result['current_drawdown'] == 0.0

    def test_simple_drawdown(self):
        """Test portfolio with a simple drawdown."""
        # Peak at 120, trough at 90, 25% drawdown
        portfolio = torch.tensor([100, 110, 120, 110, 100, 90, 95, 100], dtype=torch.float32)
        result = compute_max_drawdown(portfolio)

        expected_dd = (90 - 120) / 120  # -0.25
        assert abs(result['max_drawdown'] - expected_dd) < 1e-6
        assert result['max_drawdown_duration'] == 3  # From idx 2 (peak) to idx 5 (trough)
        assert result['peak_value'] == 120.0
        assert result['trough_value'] == 90.0

    def test_multiple_drawdowns(self):
        """Test portfolio with multiple drawdowns (takes the maximum)."""
        # First DD: 120->100 (16.67%)
        # Second DD: 140->70 (50%) <- this should be max
        portfolio = torch.tensor([100, 120, 100, 110, 140, 120, 100, 70, 80], dtype=torch.float32)
        result = compute_max_drawdown(portfolio)

        expected_dd = (70 - 140) / 140  # -0.5
        assert abs(result['max_drawdown'] - expected_dd) < 1e-6
        assert result['peak_value'] == 140.0
        assert result['trough_value'] == 70.0

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        portfolio = torch.tensor([], dtype=torch.float32)
        result = compute_max_drawdown(portfolio)

        assert result['max_drawdown'] == 0.0
        assert result['max_drawdown_duration'] == 0

    def test_single_value(self):
        """Test portfolio with single value."""
        portfolio = torch.tensor([100], dtype=torch.float32)
        result = compute_max_drawdown(portfolio)

        assert result['max_drawdown'] == 0.0
        assert result['max_drawdown_duration'] == 0


class TestSharpeRatio:
    """Test Sharpe ratio computation."""

    def test_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        # Consistent 1% returns per period
        returns = torch.tensor([0.01] * 100, dtype=torch.float32)
        periods_per_year = 252  # Daily

        sharpe = compute_sharpe_ratio(returns, periods_per_year, rf_annual=0.0)

        # With consistent returns, std should be 0, but in practice will be small
        # Sharpe should be very high (positive)
        assert sharpe > 0

    def test_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        returns = torch.tensor([-0.01] * 100, dtype=torch.float32)
        periods_per_year = 252

        sharpe = compute_sharpe_ratio(returns, periods_per_year, rf_annual=0.0)

        assert sharpe < 0

    def test_zero_returns(self):
        """Test Sharpe ratio with zero returns."""
        returns = torch.tensor([0.0] * 100, dtype=torch.float32)
        periods_per_year = 252

        sharpe = compute_sharpe_ratio(returns, periods_per_year, rf_annual=0.0)

        # With zero returns and zero std, should be 0 or nan
        assert sharpe == 0 or torch.isnan(torch.tensor(sharpe))

    def test_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = torch.tensor([0.01] * 100, dtype=torch.float32)
        periods_per_year = 252
        rf_annual = 0.02  # 2% annual risk-free rate

        sharpe_no_rf = compute_sharpe_ratio(returns, periods_per_year, rf_annual=0.0)
        sharpe_with_rf = compute_sharpe_ratio(returns, periods_per_year, rf_annual=rf_annual)

        # Sharpe with RF should be lower (less excess return)
        assert sharpe_with_rf < sharpe_no_rf


class TestSortinoRatio:
    """Test Sortino ratio computation."""

    def test_only_positive_returns(self):
        """Test Sortino ratio with only positive returns."""
        returns = torch.tensor([0.01, 0.02, 0.015, 0.01, 0.025], dtype=torch.float32)
        periods_per_year = 252

        sortino = compute_sortino_ratio(returns, periods_per_year)

        # With no downside and positive returns, Sortino should be very high (1000.0)
        assert sortino == 1000.0

    def test_mixed_returns(self):
        """Test Sortino ratio with mixed returns."""
        returns = torch.tensor([0.02, -0.01, 0.03, -0.015, 0.01], dtype=torch.float32)
        periods_per_year = 252

        sortino = compute_sortino_ratio(returns, periods_per_year)

        # Should produce a valid ratio
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_only_negative_returns(self):
        """Test Sortino ratio with only negative returns."""
        returns = torch.tensor([-0.01, -0.02, -0.015], dtype=torch.float32)
        periods_per_year = 252

        sortino = compute_sortino_ratio(returns, periods_per_year)

        assert sortino < 0


class TestCalmarRatio:
    """Test Calmar ratio computation."""

    def test_positive_return_with_drawdown(self):
        """Test Calmar ratio with positive return and drawdown."""
        # Portfolio goes up but has drawdown
        portfolio = torch.tensor([100, 120, 100, 130], dtype=torch.float32)
        periods_per_year = 252

        calmar = compute_calmar_ratio(portfolio, periods_per_year)

        # Should be positive (positive return despite drawdown)
        assert calmar > 0

    def test_no_drawdown(self):
        """Test Calmar ratio with no drawdown."""
        portfolio = torch.tensor([100, 110, 120, 130], dtype=torch.float32)
        periods_per_year = 252

        calmar = compute_calmar_ratio(portfolio, periods_per_year)

        # With no drawdown and positive returns, Calmar should be very high (1000.0)
        assert calmar == 1000.0

    def test_negative_return(self):
        """Test Calmar ratio with negative return."""
        portfolio = torch.tensor([100, 90, 80, 70], dtype=torch.float32)
        periods_per_year = 252

        calmar = compute_calmar_ratio(portfolio, periods_per_year)

        # Negative return with drawdown -> negative Calmar
        assert calmar < 0

    def test_empty_portfolio(self):
        """Test Calmar ratio with empty portfolio."""
        portfolio = torch.tensor([], dtype=torch.float32)
        periods_per_year = 252

        calmar = compute_calmar_ratio(portfolio, periods_per_year)

        assert calmar == 0.0


class TestWinRate:
    """Test win rate computation."""

    def test_all_wins(self):
        """Test win rate with all positive returns."""
        returns = torch.tensor([0.01, 0.02, 0.015, 0.03], dtype=torch.float32)
        result = compute_win_rate(returns)

        assert result['win_rate (reward>0)'] == 1.0
        assert result['avg_win'] > 0
        assert result['avg_loss'] == 0.0
        assert result['profit_factor'] == 0.0  # No losses

    def test_all_losses(self):
        """Test win rate with all negative returns."""
        returns = torch.tensor([-0.01, -0.02, -0.015], dtype=torch.float32)
        result = compute_win_rate(returns)

        assert result['win_rate (reward>0)'] == 0.0
        assert result['avg_win'] == 0.0
        assert result['avg_loss'] < 0
        assert result['profit_factor'] == 0.0  # No wins

    def test_mixed_returns(self):
        """Test win rate with mixed returns."""
        returns = torch.tensor([0.02, -0.01, 0.03, -0.015, 0.01], dtype=torch.float32)
        result = compute_win_rate(returns)

        assert result['win_rate (reward>0)'] == 0.6  # 3 wins out of 5
        assert result['avg_win'] > 0
        assert result['avg_loss'] < 0
        assert result['profit_factor'] > 0

    def test_empty_returns(self):
        """Test win rate with empty returns."""
        returns = torch.tensor([], dtype=torch.float32)
        result = compute_win_rate(returns)

        assert result['win_rate (reward>0)'] == 0.0
        assert result['avg_win'] == 0.0
        assert result['avg_loss'] == 0.0
        assert result['profit_factor'] == 0.0


class TestPortfolioReturns:
    """Test portfolio returns computation."""

    def test_increasing_portfolio(self):
        """Test returns with increasing portfolio."""
        portfolio = torch.tensor([100, 110, 121], dtype=torch.float32)
        returns = compute_portfolio_returns(portfolio)

        expected = torch.tensor([0.1, 0.1], dtype=torch.float32)
        torch.testing.assert_close(returns, expected)

    def test_decreasing_portfolio(self):
        """Test returns with decreasing portfolio."""
        portfolio = torch.tensor([100, 90, 81], dtype=torch.float32)
        returns = compute_portfolio_returns(portfolio)

        expected = torch.tensor([-0.1, -0.1], dtype=torch.float32)
        torch.testing.assert_close(returns, expected)

    def test_single_value(self):
        """Test returns with single value."""
        portfolio = torch.tensor([100], dtype=torch.float32)
        returns = compute_portfolio_returns(portfolio)

        assert len(returns) == 0

    def test_empty_portfolio(self):
        """Test returns with empty portfolio."""
        portfolio = torch.tensor([], dtype=torch.float32)
        returns = compute_portfolio_returns(portfolio)

        assert len(returns) == 0


class TestMetricsIntegration:
    """Integration tests for multiple metrics together."""

    def test_realistic_trading_scenario(self):
        """Test with a realistic trading scenario."""
        # Simulate a trading episode
        np.random.seed(42)
        n_steps = 100
        initial_value = 10000

        # Generate returns with some wins and losses
        returns = np.random.normal(0.001, 0.02, n_steps)
        portfolio_values = initial_value * np.cumprod(1 + returns)
        portfolio_tensor = torch.tensor(portfolio_values, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        periods_per_year = 525600  # 1-minute periods

        # Compute all metrics
        dd_metrics = compute_max_drawdown(portfolio_tensor)
        sharpe = compute_sharpe_ratio(returns_tensor, periods_per_year)
        sortino = compute_sortino_ratio(returns_tensor, periods_per_year)
        calmar = compute_calmar_ratio(portfolio_tensor, periods_per_year)
        win_metrics = compute_win_rate(returns_tensor)

        # Basic sanity checks
        assert isinstance(dd_metrics['max_drawdown'], float)
        assert dd_metrics['max_drawdown'] <= 0  # Drawdown is negative
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(calmar, float)
        assert 0 <= win_metrics['win_rate (reward>0)'] <= 1
        assert win_metrics['profit_factor'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
