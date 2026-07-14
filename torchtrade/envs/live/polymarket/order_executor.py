"""Paper-trading order executor for Polymarket.

PAPER ONLY. ``dry_run=False`` is REFUSED here, not just at the env's config boundary: this
class is publicly exported (``torchtrade.envs.live``), so a caller reaching for the exported
API could otherwise construct a live CLOB client and post a real order while
:class:`PolymarketBetEnv` believed itself paper-only. The guard has to sit on the class that
can actually move money, not only on the env that happens to build it.

See ``LIVE_UNSUPPORTED`` in ``env.py`` for why live is unsupported, and for sources -- do not
restate it here.

The V1 py-clob-client machinery this module used to carry (``ClobClient`` construction,
``MarketOrderArgs``/``OrderType.FOK`` order posting) was deleted rather than left behind the
guard: it is unreachable once live is refused, and it would not have survived the CLOB V2 port
anyway (different package, different types, pUSD collateral, plus a redemption workflow that
does not exist yet). Recover it from git history if the port ever wants a reference.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

LIVE_EXECUTOR_UNSUPPORTED = (
    "PolymarketOrderExecutor is paper-only: dry_run=False is refused. py-clob-client is "
    "archived and no longer functional against Polymarket's CLOB V2, and resolved winnings "
    "are never redeemed -- without a redeem, a winning account's collateral drains to zero. "
    "See LIVE_UNSUPPORTED in torchtrade/envs/live/polymarket/env.py."
)


class PolymarketOrderExecutor:
    """Simulates Polymarket order submission. Never touches the network.

    Built automatically by :class:`PolymarketBetEnv` (which is paper-only and calls nothing
    here but :meth:`cancel_all`). This is where the CLOB V2 port begins.
    """

    def __init__(
        self,
        private_key: str = "",
        chain_id: int = 137,
        signature_type: int = 0,
        funder: Optional[str] = None,
        dry_run: bool = True,
    ):
        if not dry_run:
            raise NotImplementedError(LIVE_EXECUTOR_UNSUPPORTED)
        self._dry_run = True
        # Fully offline: no client, no API-cred derivation (which would hit the wallet RPC),
        # no dependency on py-clob-client being installed, no need for a funded key.
        self.client = None

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Simulate buying ``amount_usdc`` of ``token_id``. Submits nothing."""
        logger.info("DRY RUN: BUY %.4f of %s", amount_usdc, token_id)
        return {"success": True, "dry_run": True}

    def cancel_all(self) -> bool:
        """No-op: there are no live orders to cancel."""
        return True
