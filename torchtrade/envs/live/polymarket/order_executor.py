"""Order executor for Polymarket CLOB trading via py-clob-client.

Minimal surface — only the operations :class:`PolymarketBetEnv` needs:

- :meth:`buy` — submits a fill-or-kill market order for a single side.
- :meth:`cancel_all` — called from :meth:`PolymarketBetEnv.close`.

``dry_run=True`` skips the CLOB client entirely, so paper-trading the env
works without ``py-clob-client`` installed.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
except ImportError:
    ClobClient = None
    MarketOrderArgs = None
    OrderType = None
    BUY = "BUY"


class PolymarketOrderExecutor:
    """Buys YES/NO outcome shares on Polymarket via py-clob-client.

    Constructed automatically by :class:`PolymarketBetEnv`; you typically don't
    instantiate this directly.
    """

    def __init__(
        self,
        private_key: str,
        chain_id: int = 137,
        signature_type: int = 0,
        funder: Optional[str] = None,
        dry_run: bool = False,
    ):
        self._dry_run = dry_run
        # Dry-run is fully offline: never construct the live CLOB client, never
        # derive API creds. This keeps paper-trading independent of py-clob-client
        # availability AND of having a valid funded private key.
        if dry_run:
            self.client = None
            return
        if ClobClient is None:
            raise ImportError(
                "py-clob-client is required for live Polymarket trading. "
                "Install with: pip install py-clob-client. "
                "(Pass dry_run=True to skip the CLOB client.)"
            )

        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Buy ``amount_usdc`` worth of shares for ``token_id`` (FOK market order)."""
        if self._dry_run:
            logger.info("DRY RUN: BUY %.4f of %s", amount_usdc, token_id)
            return {"success": True, "dry_run": True}
        try:
            order_args = MarketOrderArgs(
                token_id=token_id, amount=amount_usdc, side=BUY
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("BUY failed for token %s: %s", token_id, e)
            return {"success": False, "error": str(e)}

    def cancel_all(self) -> bool:
        """Cancel all open orders (no-op when no client). Returns False on API failure."""
        if self.client is None:
            return True
        try:
            self.client.cancel_all()
            return True
        except Exception:
            logger.exception("Cancel all failed")
            return False
