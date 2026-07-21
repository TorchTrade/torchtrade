"""State-value critic V(s) over a numeric trading observation — the SAO baseline.

Used by the LLM SAO trainer: the LLM is the policy (it picks the action), while
this small critic supplies the variance-reducing baseline V(s) ≈ E[R | bar]. It
reads only the pre-decision market state (``market_data_*`` windows +
``account_state``), never the sampled action/tokens, so ``R − V(s)`` is an
unbiased advantage. Trained (smooth_l1/Huber) against the realized reward.

A plain LayerNorm MLP (no BatchNorm) so it works for any batch size, including 1;
the input width is inferred lazily on the first forward, so it adapts to whatever
timeframes/window sizes the env produces without wiring shapes by hand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDictBase


class ObservationCritic(nn.Module):
    """Scalar V(s) over ``market_data_*`` windows + ``account_state``.

    Args:
        market_data_keys: the env's market-data observation keys (each a
            ``(window, features)`` tensor, batched as ``(B, window, features)``).
        account_state_key: the account-state key. Default ``"account_state"``.
        hidden_size: MLP width. Default 128.
    """

    def __init__(
        self,
        market_data_keys,
        account_state_key: str = "account_state",
        hidden_size: int = 128,
    ):
        super().__init__()
        self.market_data_keys = list(market_data_keys)
        self.account_state_key = account_state_key
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def _flatten_features(self, td: TensorDictBase) -> torch.Tensor:
        # each market_data window is (..., window, features) -> (..., window*features)
        feats = [td.get(k).flatten(start_dim=-2) for k in self.market_data_keys]
        feats.append(td.get(self.account_state_key))
        return torch.cat(feats, dim=-1)

    def forward(self, td: TensorDictBase) -> torch.Tensor:
        """Return V(s) with shape ``(*batch, 1)``."""
        return self.net(self._flatten_features(td))
