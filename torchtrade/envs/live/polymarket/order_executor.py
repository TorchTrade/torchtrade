"""Order executor for Polymarket CLOB trading via py-clob-client."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:
    ClobClient = None
    MarketOrderArgs = None
    OrderType = None
    BUY = "BUY"
    SELL = "SELL"


class PolymarketOrderExecutor:
    """Executes trades on Polymarket via the CLOB API.

    Wraps py-clob-client for buying/selling YES/NO outcome shares.
    Supports dry-run mode for paper trading.
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
        if dry_run and ClobClient is None:
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

    def get_balance(self) -> float:
        """USDC balance in dollars (returns 0.0 in dry-run-without-client or on API failure)."""
        if self.client is None:
            return 0.0
        try:
            result = self.client.get_balance_allowance()
            return float(result.get("balance", 0)) / 1e6
        except Exception:
            logger.exception("Failed to get balance")
            return 0.0

    def _place_market(self, token_id: str, amount: float, side: str) -> dict:
        if self._dry_run:
            logger.info("DRY RUN: %s %.4f of %s", side, amount, token_id)
            return {"success": True, "dry_run": True}
        try:
            order_args = MarketOrderArgs(token_id=token_id, amount=amount, side=side)
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("%s failed for token %s: %s", side, token_id, e)
            return {"success": False, "error": str(e)}

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Buy ``amount_usdc`` worth of shares for ``token_id``."""
        return self._place_market(token_id, amount_usdc, BUY)

    def sell(self, token_id: str, amount_shares: float) -> dict:
        """Sell ``amount_shares`` shares for ``token_id``."""
        return self._place_market(token_id, amount_shares, SELL)

    def get_positions(self) -> list:
        """Current share holdings ([] in dry-run-without-client or on API failure)."""
        if self.client is None:
            return []
        try:
            return self.client.get_positions()
        except Exception:
            logger.exception("Failed to get positions")
            return []

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

    def close_position(self, token_id: str) -> dict:
        """Sell all shares held for ``token_id``."""
        if self.client is None:
            return {"success": True, "already_flat": True}
        try:
            for pos in self.client.get_positions():
                if pos.get("asset") == token_id:
                    size = float(pos.get("size", 0))
                    if size > 0:
                        return self.sell(token_id, size)
            return {"success": True, "already_flat": True}
        except Exception as e:
            logger.error("Close position failed for %s: %s", token_id, e)
            return {"success": False, "error": str(e)}
