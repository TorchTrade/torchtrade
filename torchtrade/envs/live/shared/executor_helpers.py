"""Order-executor logic that was verbatim across exchanges (#288).

Only genuinely identical bodies live here. `_round_price` is NOT one of them: bitget
rounds through CCXT's `price_to_precision`, while binance, bybit and okx do tick-size
arithmetic against a cached tick. Those are different mechanisms with different failure
modes, and folding them together to make the file shorter would be the same mistake as
leaving four copies of something that is one rule.
"""

import logging

from torchtrade.envs.core.state import POSITION_DUST_EPS

logger = logging.getLogger(__name__)


class ExecutorHelpersMixin:
    """Shared executor helpers. Mixed in ahead of the exchange class."""

    def _calculate_unrealized_pnl_pct(
        self, qty: float, entry_price: float, mark_price: float
    ) -> float:
        """Unrealized PnL as a fraction of entry, signed by the position's direction.

        Was three copies (bitget, bybit, okx) with identical bodies and differently
        sized docstrings. Divergence in a copy of this is expensive twice over: it feeds
        `account_state[2]`, which the policy reads every step, and it is the number a
        reward function sees.

        Direction goes through the dust rule rather than `qty > 0`: a float residual
        left by a full close is not a short, and reporting a PnL against one puts a
        phantom position's number in the observation (invariant 1).
        """
        if entry_price <= 0:
            return 0.0
        if qty > POSITION_DUST_EPS:
            return (mark_price - entry_price) / entry_price
        if qty < -POSITION_DUST_EPS:
            return (entry_price - mark_price) / entry_price
        return 0.0

    def _round_price_by_tick(self, price: float) -> float:
        """Round to the cached tick size. Three verbatim copies (binance, bybit, okx).

        bitget deliberately does not use this -- it rounds through CCXT.
        """
        if self._tick_size is not None:
            rounded = round(price / self._tick_size) * self._tick_size
            return round(rounded, self._tick_decimals)
        return price
