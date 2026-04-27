"""Polymarket observation class — fetches market state each time bar."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import requests
import torch
from torchrl.data import Bounded

from torchtrade.envs.live.polymarket.market_scanner import GAMMA_API_BASE

logger = logging.getLogger(__name__)

_MAX_HORIZON_SECONDS = 365 * 24 * 3600  # normalize time-to-resolution to (0, 1]


class PolymarketObservationClass:
    """Fetches Polymarket market state for a single market each time bar.

    Produces a 5-element ``market_state`` vector:
    ``[yes_price, spread, volume_24h, liquidity, time_to_resolution]``.
    """

    def __init__(
        self,
        yes_token_id: str,
        market_slug: str = "",
        condition_id: str = "",
        clob_client=None,
        feature_preprocessing_fn=None,
    ):
        self.yes_token_id = yes_token_id
        self.market_slug = market_slug
        self.condition_id = condition_id
        self.clob_client = clob_client
        self._feature_preprocessing_fn = feature_preprocessing_fn

        self._market_meta = self._fetch_market_metadata()
        token_ids = json.loads(self._market_meta.get("clobTokenIds") or "[]")
        self.no_token_id = token_ids[1] if len(token_ids) > 1 else ""

    def _fetch_market_metadata(self) -> dict:
        """Fetch market metadata from the Gamma API."""
        if self.market_slug:
            params = {"slug": self.market_slug}
        elif self.condition_id:
            params = {"condition_id": self.condition_id}
        else:
            params = {"clob_token_ids": self.yes_token_id}
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Failed to fetch market metadata")
            return {}
        if isinstance(data, list):
            return data[0] if data else {}
        return data

    def get_observations(self) -> dict:
        """Fetch current market state. Returns dict with ``market_state`` ndarray (5,)."""
        self._market_meta = self._fetch_market_metadata()

        market_state = np.array(
            [
                self.get_yes_price(),
                self._get_spread(),
                float(self._market_meta.get("volume24hr", 0)),
                float(self._market_meta.get("liquidity", 0)),
                self._get_time_to_resolution(),
            ],
            dtype=np.float32,
        )
        if self._feature_preprocessing_fn is not None:
            market_state = self._feature_preprocessing_fn(market_state)
        return {"market_state": market_state}

    def get_observation_spec(self) -> dict:
        """TorchRL spec for ``market_state`` (matches PolyTimeBarEnv)."""
        return {
            "market_state": Bounded(
                low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32
            ),
        }

    def get_yes_price(self) -> float:
        """Current YES midpoint from CLOB; falls back to Gamma metadata."""
        if self.clob_client is not None:
            try:
                return float(self.clob_client.get_midpoint(self.yes_token_id))
            except Exception:
                logger.warning("CLOB midpoint fetch failed; falling back to Gamma")
        try:
            prices = json.loads(self._market_meta.get("outcomePrices") or '["0.5","0.5"]')
            return float(prices[0])
        except (ValueError, IndexError, TypeError):
            return 0.5

    def _get_spread(self) -> float:
        """Bid-ask spread from CLOB order book; 0.0 if CLOB unavailable."""
        if self.clob_client is None:
            return 0.0
        try:
            book = self.clob_client.get_order_book(self.yes_token_id)
            best_bid = float(book.bids[0].price) if book.bids else 0.0
            best_ask = float(book.asks[0].price) if book.asks else 1.0
            return best_ask - best_bid
        except Exception:
            logger.warning("CLOB order-book fetch failed; reporting spread=0")
            return 0.0

    def _get_time_to_resolution(self) -> float:
        """Normalized time remaining until resolution (1.0 -> 0.0)."""
        end_date_str = self._market_meta.get("endDate", "")
        if not end_date_str:
            return 1.0
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return 1.0
        remaining = (end_dt - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return 0.0
        return min(remaining / _MAX_HORIZON_SECONDS, 1.0)

    def is_market_closed(self) -> bool:
        """True once the market has resolved/closed."""
        return bool(self._market_meta.get("closed", False))
