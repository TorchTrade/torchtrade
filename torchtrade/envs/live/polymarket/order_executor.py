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
        if ClobClient is None:
            raise ImportError(
                "py-clob-client is required. Install with: pip install py-clob-client"
            )

        self._dry_run = dry_run
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def get_balance(self) -> float:
        """Get USDC balance in dollars."""
        try:
            result = self.client.get_balance_allowance()
            raw_balance = float(result.get("balance", 0))
            return raw_balance / 1e6
        except Exception:
            logger.exception("Failed to get balance")
            return 0.0

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Buy shares for a token.

        Args:
            token_id: The YES or NO token ID.
            amount_usdc: Dollar amount to spend.

        Returns:
            Dict with 'success' bool and order details.
        """
        if self._dry_run:
            logger.info("DRY RUN: buy %s for $%.2f", token_id, amount_usdc)
            return {"success": True, "dry_run": True}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id, amount=amount_usdc, side=BUY
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("Buy failed for token %s: %s", token_id, e)
            return {"success": False, "error": str(e)}

    def sell(self, token_id: str, amount_shares: float) -> dict:
        """Sell shares for a token.

        Args:
            token_id: The YES or NO token ID.
            amount_shares: Number of shares to sell.

        Returns:
            Dict with 'success' bool and order details.
        """
        if self._dry_run:
            logger.info("DRY RUN: sell %.4f of %s", amount_shares, token_id)
            return {"success": True, "dry_run": True}

        try:
            order_args = MarketOrderArgs(
                token_id=token_id, amount=amount_shares, side=SELL
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return {"success": True, "order": result}
        except Exception as e:
            logger.error("Sell failed for token %s: %s", token_id, e)
            return {"success": False, "error": str(e)}

    def get_positions(self) -> list:
        """Get current share holdings."""
        try:
            return self.client.get_positions()
        except Exception:
            logger.exception("Failed to get positions")
            return []

    def cancel_all(self) -> bool:
        """Cancel all open orders."""
        try:
            self.client.cancel_all()
            return True
        except Exception:
            logger.exception("Cancel all failed")
            return False

    def get_yes_price(self, token_id: str) -> float:
        """Get current midpoint price for a token."""
        try:
            return float(self.client.get_midpoint(token_id))
        except Exception:
            logger.exception("Failed to get price for %s", token_id)
            return 0.0

    def close_position(self, token_id: str) -> dict:
        """Close all shares for a token."""
        try:
            positions = self.client.get_positions()
            for pos in positions:
                if pos.get("asset") == token_id:
                    size = float(pos.get("size", 0))
                    if size > 0:
                        return self.sell(token_id, size)
            return {"success": True, "already_flat": True}
        except Exception as e:
            logger.error("Close position failed for %s: %s", token_id, e)
            return {"success": False, "error": str(e)}
