"""Portfolio baselines from the online portfolio selection literature, as `td -> td` policies.

Each writes `td["action"]` (target weights, cash first) from the portfolio env's observation,
so `env.rollout(policy=UCRP())` evaluates them exactly like a trained policy, on the scalar
or the vectorized env. All three are long-only and fully invested: the env pins the holding
of an asset that is not tradable at the decision bar and spreads the rest of the request
over the others, and caps gross exposure at `max_gross`.

The parameter is named `td` on purpose: torchrl passes a callable whose sole parameter is
`td` or `tensordict` the whole TensorDict, and wraps any other signature in a
TensorDictModule that hands over individual tensors.
"""

import torch
from tensordict import TensorDictBase


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of the last dim onto the probability simplex (Duchi et al., 2008).

    Done in float64: at float32 magnitudes above ~1.6e7 the cumulative sums lose the unit
    offset and the projection returns nothing usable.
    """
    v64 = v.double()
    u, _ = v64.sort(dim=-1, descending=True)
    css = u.cumsum(-1) - 1
    k = torch.arange(1, v.shape[-1] + 1, device=v.device)
    rho = ((u - css / k) > 0).sum(-1, keepdim=True)
    theta = css.gather(-1, rho - 1) / rho
    return (v64 - theta).clamp(min=0).to(v.dtype)


def _equal_weights(weights: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(weights, 1.0 / (weights.shape[-1] - 1))
    out[..., 0] = 0.0
    return out


class UBAH:
    """Uniform buy and hold: 1/N in each asset at its first tradable decision, then never rebalance.

    Cash is kept for the assets that are not yet tradable (a stock outside its session, an asset
    not yet listed), so the hold does not depend on which hour the window opens. A lane that is
    tradable and unheld is bought at 1/N, or at the cash left if that is less; nothing is sold.
    """

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        w = td["portfolio_weights"].clamp(min=0)  # the float32 cash weight can be -1e-16
        cash, assets = w[..., :1], w[..., 1:]
        unheld = (td["tradable"] > 0) & (assets <= 0)
        each = torch.minimum(torch.full_like(cash, 1.0 / assets.shape[-1]), cash / unheld.sum(-1, keepdim=True).clamp(min=1))
        buys = unheld * each
        td["action"] = torch.cat([cash - buys.sum(-1, keepdim=True), assets + buys], -1)
        return td


class UCRP:
    """Uniform constant rebalanced portfolio: back to equal weights every step."""

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        td["action"] = _equal_weights(td["portfolio_weights"])
        return td


class OLMAR:
    """On-line moving average reversion (Li & Hoi, 2012), the OLMAR-1 variant.

    Predicts next price relatives as the mean close over the latest close in the first
    `market_data_*` window, moves the drifted weights toward that prediction until the
    expected relative reaches `epsilon`, and projects back onto the simplex. Cash is the
    asset with a relative of 1. Assets that are not tradable get no signal; bars before an
    asset listed are left out of its mean.
    """

    def __init__(self, window: int = 5, epsilon: float = 10.0):
        self.window, self.epsilon = window, epsilon

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        key = next(k for k in td.keys() if k.startswith("market_data_"))
        closes = td[key][..., -self.window :, 0]  # (..., N, window) close over the latest close
        tradable = td["tradable"] > 0
        n_bars = (closes > 0).sum(-1)  # bars before an asset listed are zero, not a crash
        x_hat = torch.where(tradable & (n_bars > 0), closes.sum(-1) / n_bars.clamp(min=1), 1.0)
        x_hat = torch.cat([torch.ones_like(x_hat[..., :1]), x_hat], -1)

        b = td["portfolio_weights"].clamp(min=0)
        b = b / b.sum(-1, keepdim=True).clamp(min=1e-12)  # an all-short book has no long mass
        centered = x_hat - x_hat.mean(-1, keepdim=True)
        denom = (centered * centered).sum(-1, keepdim=True)  # zero exactly when centered is
        lam = ((self.epsilon - (b * x_hat).sum(-1, keepdim=True)) / denom.clamp(min=1e-12)).clamp(min=0)
        td["action"] = project_simplex(b + lam * centered)
        return td
