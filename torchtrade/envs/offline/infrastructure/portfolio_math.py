"""The portfolio envs' money math: one pure, batched, differentiable function.

Weights are fractions of portfolio value, cash at index 0, `w_cash + Σ|w_i| = 1`.
A trade pays `fee·|Δnotional|` (a perpetual swap's fee), so the post-trade value is the
fixed point `μ = 1 − fee·Σ_i |w'_i − μ·w_i|`, a contraction with factor at most `fee`.
"""

from typing import NamedTuple

import torch

MONEY_DTYPE = torch.float64
# Converges to 1e-12 within 7 iterations at a 2% fee; a fixed count keeps the function
# batched and differentiable.
MU_ITERS = 20


class PortfolioStep(NamedTuple):
    pv_factor: torch.Tensor
    weights: torch.Tensor
    drifted: torch.Tensor
    commission: torch.Tensor
    funding: torch.Tensor


def _with_cash(assets: torch.Tensor) -> torch.Tensor:
    return torch.cat([(1 - assets.abs().sum(-1))[..., None], assets], -1)


def normalise_request(request, force_close, max_gross, allow_short):
    """Any action -> valid target weights: cash ≥ 0, gross ≤ max_gross, all-zero -> cash."""
    cash = request[..., 0].clamp(min=0)
    assets = request[..., 1:] if allow_short else request[..., 1:].clamp(min=0)
    assets = torch.where(force_close, torch.zeros_like(assets), assets)
    total = cash + assets.abs().sum(-1)
    assets = torch.where(
        (total > 0)[..., None], assets / torch.where(total > 0, total, 1.0)[..., None], 0.0
    )
    gross = assets.abs().sum(-1)
    assets = assets * (max_gross / gross.clamp(min=1e-12)).clamp(max=1.0)[..., None]
    return _with_cash(assets)


def _target_weights(held, tradable, open_assets, open_mass, mu):
    # A closed asset's post-trade holding μ·w_i must equal what it already holds.
    pinned = torch.where(tradable, torch.zeros_like(held), held / mu[..., None])
    has_open = open_mass > 0
    budget = torch.where(
        has_open,
        (1 - pinned.abs().sum(-1)).clamp(min=0) / torch.where(has_open, open_mass, 1.0),
        0.0,
    )
    return _with_cash(pinned + open_assets * budget[..., None])


def portfolio_step(
    drifted, request, tradable, force_close, price_relative, funding_rate,
    fee, max_gross, allow_short,
) -> PortfolioStep:
    """Rebalance at bar n's close, then carry the portfolio to bar n+1."""
    target = normalise_request(request, force_close, max_gross, allow_short)
    open_assets = torch.where(tradable, target[..., 1:], torch.zeros_like(target[..., 1:]))
    open_mass = target[..., 0] + open_assets.abs().sum(-1)
    held = drifted[..., 1:]

    mu = torch.ones_like(open_mass)
    for _ in range(MU_ITERS):
        weights = _target_weights(held, tradable, open_assets, open_mass, mu)
        mu = 1 - fee * (held - mu[..., None] * weights[..., 1:]).abs().sum(-1)
    weights = _target_weights(held, tradable, open_assets, open_mass, mu)

    growth = 1 + (weights[..., 1:] * (price_relative - 1)).sum(-1)
    next_assets = weights[..., 1:] * price_relative / growth[..., None]
    funding_share = (next_assets * funding_rate).sum(-1)
    return PortfolioStep(
        pv_factor=mu * growth * (1 - funding_share),
        weights=weights,
        drifted=_with_cash(next_assets),
        commission=1 - mu,
        funding=mu * growth * funding_share,
    )
