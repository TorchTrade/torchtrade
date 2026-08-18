"""Order-executor logic that was verbatim across exchanges (#288).

Only genuinely identical bodies live here. `_round_price` is NOT one of them: bitget
rounds through CCXT's `price_to_precision`, while binance, bybit and okx do tick-size
arithmetic against a cached tick. Those are different mechanisms with different failure
modes, and folding them together to make the file shorter would be the same mistake as
leaving four copies of something that is one rule.
"""

from torchtrade.envs.core.state import POSITION_DUST_EPS

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
        left by a full close is not a short, and the copies reported -10% against one.

        Defence in depth, not a live bug fix -- `futures_live_base._get_account_state`
        already hard-sets this to 0.0 when the direction is dust, so nothing downstream
        currently sees the wrong number. The earlier claim that it reached the
        observation was wrong. It is still worth having: canonicalising `qty > 0` into a
        SHARED helper would hand the next caller the bug pre-approved.
        """
        if entry_price <= 0:
            return 0.0
        if qty > POSITION_DUST_EPS:
            return (mark_price - entry_price) / entry_price
        if qty < -POSITION_DUST_EPS:
            return (entry_price - mark_price) / entry_price
        return 0.0


class TickSizeMixin:
    """Helpers that need a cached `_tick_size`/`_tick_decimals`.

    Separate from ExecutorHelpersMixin because BITGET HAS NEITHER -- it rounds through
    CCXT's `price_to_precision`. Handing it these would give it methods that raise
    AttributeError on call, and an override only covers the one you remember to write:
    `_round_price` had a bitget override, `_format_price` did not, so bitget silently
    inherited an always-raising method (#288 review). Not mixing them in at all is the
    version that cannot be half-done.
    """

    def _round_price(self, price: float) -> float:
        """Round to the cached tick size. Three verbatim copies (binance, bybit, okx)."""
        if self._tick_size is not None:
            rounded = round(price / self._tick_size) * self._tick_size
            return round(rounded, self._tick_decimals)
        return price

    def _format_price(self, price: float) -> str:
        """Tick-rounded price as a deterministic string. Two verbatim copies (bybit, okx).

        A string, not a float, because the venues parse the wire value and `repr` of a
        rounded float can carry more digits than the tick allows -- which the venue then
        rejects or silently re-rounds.
        """
        rounded = self._round_price(price)
        if self._tick_size is not None:
            return f"{rounded:.{self._tick_decimals}f}"
        return str(rounded)
