import torch

def compute_sharpe_torch(returns: torch.Tensor, periods_per_year: float, rf_annual: float = 0.0):
    """
    Compute annualized Sharpe ratio using PyTorch.

    returns: 1D torch.Tensor of per-period returns
    periods_per_year: number of periods in a year
    rf_annual: annual risk-free rate (float)
    """
    # Remove NaNs if any
    returns = returns[~torch.isnan(returns)]

    # Convert annual RF to per-period
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1

    # Excess returns
    excess_returns = returns - rf_period

    # Compute mean and std
    mean_excess = torch.mean(excess_returns)
    std_excess = torch.std(excess_returns, unbiased=True) + 1e-9

    # Annualized Sharpe
    sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(periods_per_year, dtype=returns.dtype))

    return sharpe
