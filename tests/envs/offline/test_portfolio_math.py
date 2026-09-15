"""portfolio_step is the only money math the portfolio envs have; these pin it."""

import pytest
import torch

from torchtrade.envs.offline.infrastructure.portfolio_math import (
    MONEY_DTYPE,
    normalise_request,
    portfolio_step,
)


def _t(rows):
    return torch.tensor(rows, dtype=MONEY_DTYPE)


def _step(drifted, request, tradable=None, force_close=None, y=None, rate=None,
          fee=0.0, max_gross=1.0, allow_short=False):
    drifted, request = _t(drifted), _t(request)
    n = drifted.shape[-1] - 1
    ones = torch.ones(drifted.shape[0], n, dtype=MONEY_DTYPE)
    return portfolio_step(
        drifted,
        request,
        torch.ones_like(ones, dtype=torch.bool) if tradable is None else torch.tensor(tradable),
        torch.zeros_like(ones, dtype=torch.bool) if force_close is None else torch.tensor(force_close),
        ones if y is None else _t(y),
        torch.zeros_like(ones) if rate is None else _t(rate),
        fee=fee, max_gross=max_gross, allow_short=allow_short,
    )


@pytest.mark.parametrize("request_,force_close,max_gross,allow_short,expected", [
    ([[0.5, 0.5, 0.0]], [[False, False]], 1.0, False, [[0.5, 0.5, 0.0]]),
    ([[2.0, 1.0, 1.0]], [[False, False]], 1.0, False, [[0.5, 0.25, 0.25]]),       # rescaled
    ([[0.0, 1.0, 1.0]], [[False, False]], 0.5, False, [[0.5, 0.25, 0.25]]),       # gross cap
    ([[0.0, -1.0, 1.0]], [[False, False]], 1.0, False, [[0.0, 0.0, 1.0]]),        # long-only clip
    ([[0.0, -1.0, 1.0]], [[False, False]], 1.0, True, [[0.0, -0.5, 0.5]]),        # short kept
    ([[-1.0, 0.0, 0.0]], [[False, False]], 1.0, True, [[1.0, 0.0, 0.0]]),         # all-zero -> cash
    ([[0.0, 1.0, 1.0]], [[True, False]], 1.0, False, [[0.0, 0.0, 1.0]]),          # force close
], ids=["valid", "rescale", "gross-cap", "long-only-clip", "short", "empty-to-cash", "force-close"])
def test_normalise_request(request_, force_close, max_gross, allow_short, expected):
    out = normalise_request(_t(request_), torch.tensor(force_close), max_gross, allow_short)
    torch.testing.assert_close(out, _t(expected))


@pytest.mark.parametrize("drifted,request_,allow_short,fee,mu", [
    ([[1.0, 0.0]], [[0.0, 1.0]], False, 0.01, 1 / 1.01),     # cash -> long pays fee on the new notional
    ([[1.0, 0.0]], [[0.0, -1.0]], True, 0.01, 1 / 1.01),     # cash -> short, same notional, same fee
    ([[1.0, 0.0]], [[0.0, -1.0]], False, 0.01, 1.0),         # long-only: short request clipped, no trade
    ([[0.0, 1.0]], [[1.0, 0.0]], False, 0.01, 0.99),         # long -> cash: fee on the held notional
    ([[0.0, 1.0]], [[0.0, -1.0]], True, 0.01, 0.99 / 1.01),  # flip pays both legs
], ids=["open-long", "open-short", "short-clipped", "close-long", "flip"])
def test_commission_matches_hand_computed(drifted, request_, allow_short, fee, mu):
    out = _step(drifted, request_, fee=fee, allow_short=allow_short)
    torch.testing.assert_close(1 - out.commission, _t([mu]), atol=1e-12, rtol=0)


@pytest.mark.parametrize("fee", [0.0, 0.001, 0.02])
@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("closed_share", [0.0, 0.3, 1.0])
def test_solution_satisfies_every_defining_condition(fee, allow_short, closed_share):
    """Any vector meeting all four conditions IS the answer; none of them re-derives μ."""
    gen = torch.Generator().manual_seed(0)
    B, N = 512, 12
    def raw():
        return torch.randn(B, N + 1, generator=gen, dtype=MONEY_DTYPE)

    no_close = torch.zeros(B, N, dtype=torch.bool)
    drifted = normalise_request(raw(), no_close, 1.0, allow_short)
    request = normalise_request(raw(), no_close, 1.0, allow_short)
    tradable = torch.rand(B, N, generator=gen) >= closed_share
    out = portfolio_step(drifted, request, tradable, no_close,
                         torch.ones(B, N, dtype=MONEY_DTYPE), torch.zeros(B, N, dtype=MONEY_DTYPE),
                         fee=fee, max_gross=1.0, allow_short=allow_short)
    mu, w = 1 - out.commission, out.weights

    # 1. μ is the fee fixed point, reached within MU_ITERS
    residual = mu - (1 - fee * (drifted[:, 1:] - mu[:, None] * w[:, 1:]).abs().sum(-1))
    assert residual.abs().max() < 1e-12
    # 2. closed assets are not traded
    closed_trade = torch.where(tradable, 0.0, drifted[:, 1:] - mu[:, None] * w[:, 1:])
    assert closed_trade.abs().max() < 1e-12
    # 3. gross identity
    assert (w[:, 0] + w[:, 1:].abs().sum(-1) - 1).abs().max() < 1e-12
    # 4. cash and tradable assets keep the requested proportions
    open_w = torch.cat([w[:, :1], torch.where(tradable, w[:, 1:], 0.0)], -1)
    open_req = torch.cat([request[:, :1], torch.where(tradable, request[:, 1:], 0.0)], -1)
    has_open = open_req.abs().sum(-1) > 1e-9
    cross = open_w[:, :, None] * open_req[:, None, :] - open_w[:, None, :] * open_req[:, :, None]
    assert cross[has_open].abs().max() < 1e-12


@pytest.mark.parametrize("weights,y,expected_growth,expected_drifted", [
    ([[0.5, 0.5]], [[2.0]], 1.5, [[1 / 3, 2 / 3]]),     # long doubles
    ([[0.5, -0.5]], [[2.0]], 0.5, [[-1.0, -2.0]]),      # short doubles: gross 2, cash -1
    ([[0.5, -0.5]], [[0.5]], 1.25, [[0.8, -0.2]]),      # short halves
], ids=["long-up", "short-up", "short-down"])
def test_drift(weights, y, expected_growth, expected_drifted):
    out = _step(weights, weights, y=y, allow_short=True)
    torch.testing.assert_close(out.pv_factor, _t([expected_growth]))
    torch.testing.assert_close(out.drifted, _t(expected_drifted))


@pytest.mark.parametrize("weights,rate,expected_factor", [
    ([[0.0, 1.0]], [[0.01]], 0.99),    # long pays a positive rate
    ([[0.0, -1.0]], [[0.01]], 1.01),   # short receives it
    ([[0.0, 1.0]], [[-0.01]], 1.01),   # long receives a negative rate
    ([[1.0, 0.0]], [[0.01]], 1.0),     # flat pays nothing
], ids=["long-pays", "short-receives", "negative-rate", "flat"])
def test_funding_sign(weights, rate, expected_factor):
    out = _step(weights, weights, rate=rate, allow_short=True)
    torch.testing.assert_close(out.pv_factor, _t([expected_factor]))
    torch.testing.assert_close(out.funding, _t([1 - expected_factor]))


def test_pv_factor_is_differentiable_in_the_request():
    request = torch.tensor([[0.2, 0.5, 0.3]], dtype=MONEY_DTYPE, requires_grad=True)
    out = portfolio_step(
        _t([[1.0, 0.0, 0.0]]), request,
        torch.tensor([[True, False]]), torch.tensor([[False, False]]),
        _t([[1.1, 0.9]]), _t([[0.0, 0.0]]), fee=0.001, max_gross=1.0, allow_short=False,
    )
    out.pv_factor.sum().backward()
    assert torch.isfinite(request.grad).all() and request.grad.abs().sum() > 0
