"""Paper-trading order executor for Polymarket.

PAPER ONLY, and it refuses to pretend otherwise:

- ``dry_run=False`` raises. The refusal lives HERE, not only on the env's config, because this
  class is publicly exported (``torchtrade.envs.live``) -- a caller reaching for the exported
  API could otherwise construct a live CLOB client and post a real order while
  :class:`PolymarketBetEnv` believed itself paper-only. The guard belongs on the class that
  can move money, not only on the env that happens to build it.
- ``private_key`` raises too. Accepting and ignoring a real key would tell the caller their
  key is configured when nothing here can ever sign anything. Refused, not swallowed.

The refusal message summarizes both blockers (archived client, no redeem); the sources for
them live in ``LIVE_UNSUPPORTED`` in ``env.py``.

The V1 py-clob-client machinery this module used to carry (``ClobClient`` construction,
``MarketOrderArgs``/``OrderType.FOK`` order posting) was deleted rather than parked behind the
guard: unreachable once live is refused, and it would not have survived the CLOB V2 port
anyway (different package, different types, pUSD collateral, plus a redeem that does not exist
yet). Recover it from git history if the port wants a reference.
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
    """Simulates Polymarket order submission. Never touches the network, never signs.

    Built automatically by :class:`PolymarketBetEnv`, which calls only :meth:`cancel_all`.
    This is where the CLOB V2 port begins.
    """

    def __init__(self, *, private_key: Optional[str] = None, dry_run: bool = True):
        # Keyword-only, and the live-config params (chain_id / signature_type / funder) are
        # gone: they fed the deleted CLOB client and no caller in the tree ever passed them.
        # Any leftover call site now gets a loud TypeError naming the argument.
        if not dry_run:
            raise NotImplementedError(LIVE_EXECUTOR_UNSUPPORTED)

        # `is not None`, NOT truthiness: the old example passed
        # os.getenv("POLYGON_PRIVATE_KEY", ""), and `if private_key:` would wave that through.
        if private_key is not None:
            raise TypeError(
                "PolymarketOrderExecutor no longer takes private_key: it is paper-only and "
                "can never sign or submit anything, so a key would do nothing. Accepting it "
                "silently would tell you it was configured. Remove the argument."
            )

    def buy(self, token_id: str, amount_usdc: float) -> dict:
        """Simulate buying ``amount_usdc`` of ``token_id``. Submits nothing."""
        logger.info("DRY RUN: BUY %.4f of %s", amount_usdc, token_id)
        return {"success": True, "dry_run": True}

    def cancel_all(self) -> bool:
        """No-op: a paper env has no live orders to cancel."""
        return True
