"""Polymarket observation class — fetches market state each time bar."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import torch
from torchrl.data import Bounded

from torchtrade.envs.live.polymarket.market_scanner import GAMMA_API_BASE

logger = logging.getLogger(__name__)


class PolymarketObservationClass:
    """Fetches Polymarket market state for a single market each time bar.

    Produces a 5-element market_state vector:
    [yes_price, spread, volume_24h, liquidity, time_to_resolution]
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

        # Fetch initial market metadata from Gamma API
        self._market_meta = self._fetch_market_metadata()

        # Resolve NO token ID from metadata
        try:
            token_ids = json.loads(self._market_meta.get("clobTokenIds", "[]"))
            self.no_token_id = token_ids[1] if len(token_ids) > 1 else ""
        except (ValueError, IndexError, TypeError):
            self.no_token_id = ""

    def _fetch_market_metadata(self) -> dict:
        """Fetch market metadata from Gamma API."""
        try:
            if self.market_slug:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"slug": self.market_slug},
                    timeout=15,
                )
            elif self.condition_id:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"condition_id": self.condition_id},
                    timeout=15,
                )
            else:
                resp = requests.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"clob_token_ids": self.yes_token_id},
                    timeout=15,
                )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data[0] if data else {}
            return data
        except Exception:
            logger.exception("Failed to fetch market metadata")
            return {}

    def _refresh_market_metadata(self):
        """Refresh market metadata (called each step for freshness)."""
        self._market_meta = self._fetch_market_metadata()

    def get_observations(self) -> dict:
        """Fetch current market state and return as dict.

        Returns:
            Dict with 'market_state' key -> np.ndarray of shape (5,).
        """
        self._refresh_market_metadata()

        yes_price = self.get_yes_price()
        spread = self._get_spread()
        volume_24h = float(self._market_meta.get("volume24hr", 0))
        liquidity = float(self._market_meta.get("liquidity", 0))
        time_to_resolution = self._get_time_to_resolution()

        market_state = np.array(
            [yes_price, spread, volume_24h, liquidity, time_to_resolution],
            dtype=np.float32,
        )

        if self._feature_preprocessing_fn is not None:
            market_state = self._feature_preprocessing_fn(market_state)

        return {"market_state": market_state}

    def get_observation_spec(self) -> dict:
        """Return TorchRL spec for market_state."""
        return {
            "market_state": Bounded(
                low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32
            ),
        }

    def get_yes_price(self) -> float:
        """Get current YES midpoint price from CLOB."""
        if self.clob_client is not None:
            try:
                return float(self.clob_client.get_midpoint(self.yes_token_id))
            except Exception:
                pass
        # Fallback to Gamma metadata
        try:
            prices = json.loads(self._market_meta.get("outcomePrices", '["0.5","0.5"]'))
            return float(prices[0])
        except (ValueError, IndexError, TypeError):
            return 0.5

    def _get_spread(self) -> float:
        """Get bid-ask spread from CLOB order book."""
        if self.clob_client is None:
            return 0.0
        try:
            book = self.clob_client.get_order_book(self.yes_token_id)
            best_bid = float(book.bids[0].price) if book.bids else 0.0
            best_ask = float(book.asks[0].price) if book.asks else 1.0
            return best_ask - best_bid
        except Exception:
            return 0.0

    def _get_time_to_resolution(self) -> float:
        """Normalized time remaining until market resolution (1.0 -> 0.0)."""
        end_date_str = self._market_meta.get("endDate", "")
        if not end_date_str:
            return 1.0
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = (end_dt - now).total_seconds()
            if remaining <= 0:
                return 0.0
            max_seconds = 365 * 24 * 3600
            return min(remaining / max_seconds, 1.0)
        except (ValueError, TypeError):
            return 1.0

    def is_market_closed(self) -> bool:
        """Check if the market has resolved/closed."""
        return bool(self._market_meta.get("closed", False))
