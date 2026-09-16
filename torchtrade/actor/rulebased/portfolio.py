"""Portfolio baselines from the online portfolio selection literature, as `td -> td` policies.

Each writes `td["action"]` (target weights, cash first) from the portfolio env's observation,
so `env.rollout(policy=UCRP())` evaluates them exactly like a trained policy, on the scalar
or the vectorized env. All three are long-only; the env redistributes weight requested on an
asset that is not tradable at the decision bar.

The parameter is named `td` on purpose: torchrl passes a callable whose sole parameter is
`td` or `tensordict` the whole TensorDict, and wraps any other signature in a
TensorDictModule that hands over individual tensors.
"""

import torch
from tensordict import TensorDictBase


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of the last dim onto the probability simplex (Duchi et al., 2008)."""
    u, _ = v.sort(dim=-1, descending=True)
    css = u.cumsum(-1) - 1
    k = torch.arange(1, v.shape[-1] + 1, device=v.device)
    rho = ((u - css / k) > 0).sum(-1, keepdim=True)
    theta = css.gather(-1, rho - 1) / rho
    return (v - theta).clamp(min=0)


def _equal_weights(weights: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(weights, 1.0 / (weights.shape[-1] - 1))
    out[..., 0] = 0.0
    return out


class UBAH:
    """Uniform buy and hold: equal weights once, then never rebalance."""

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        w = td["portfolio_weights"]
        invested = w[..., 1:].abs().sum(-1, keepdim=True) > 0
        td["action"] = torch.where(invested, w, _equal_weights(w))
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
    asset with a relative of 1; assets that are not tradable or have no bars yet get no
    signal and no weight.
    """

    def __init__(self, window: int = 5, epsilon: float = 10.0):
        self.window, self.epsilon = window, epsilon

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        key = next(k for k in td.keys() if k.startswith("market_data_"))
        closes = td[key][..., -self.window :, 0]  # (..., N, window) close over the latest close
        # Bars before an asset listed are zero and must not read as a crash.
        n_bars = (closes > 0).sum(-1)
        x_hat = torch.where(n_bars > 0, closes.sum(-1) / n_bars.clamp(min=1), 1.0)
        tradable = td["tradable"] > 0
        x_hat = torch.where(tradable, x_hat, 1.0)
        x_hat = torch.cat([torch.ones_like(x_hat[..., :1]), x_hat], -1)

        b = td["portfolio_weights"].clamp(min=0)
        b = b / b.sum(-1, keepdim=True)
        centered = x_hat - x_hat.mean(-1, keepdim=True)
        denom = (centered * centered).sum(-1, keepdim=True)
        lam = (self.epsilon - (b * x_hat).sum(-1, keepdim=True)) / denom.clamp(min=1e-12)
        lam = torch.where(denom > 0, lam.clamp(min=0), 0.0)
        target = project_simplex(b + lam * centered)
        target[..., 1:] = torch.where(tradable, target[..., 1:], 0.0)
        td["action"] = target
        return td
