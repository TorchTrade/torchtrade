"""Polymarket market scanner — fetch and filter markets from Gamma API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class PolymarketMarket:
    """Parsed representation of a Polymarket prediction market."""

    market_id: str
    condition_id: str
    question: str
    description: str
    slug: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    volume_24h: float
    total_volume: float
    liquidity: float
    spread: float
    end_date: str
    tags: list
    neg_risk: bool


@dataclass
class MarketScannerConfig:
    """Configuration for market scanning and filtering."""

    min_volume_24h: float = 10_000
    min_liquidity: float = 5_000
    max_markets: int = 20
    categories: Optional[List[str]] = None
    min_time_to_resolution_hours: float = 24
    keyword: Optional[str] = None  # case-insensitive substring match on question or slug


class MarketScanner:
    """Fetches markets from the Gamma API and filters by configurable criteria."""

    def __init__(self, config: MarketScannerConfig | None = None):
        self.config = config or MarketScannerConfig()

    def _parse_market(self, raw: dict) -> PolymarketMarket:
        """Parse a raw Gamma API market JSON dict into a PolymarketMarket."""
        outcome_prices = json.loads(raw["outcomePrices"])
        clob_token_ids = json.loads(raw["clobTokenIds"])

        return PolymarketMarket(
            market_id=raw["id"],
            condition_id=raw["conditionId"],
            question=raw["question"],
            description=raw.get("description", ""),
            slug=raw["slug"],
            yes_token_id=clob_token_ids[0],
            no_token_id=clob_token_ids[1],
            yes_price=float(outcome_prices[0]),
            no_price=float(outcome_prices[1]),
            volume_24h=float(raw.get("volume24hr", 0)),
            total_volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            spread=float(raw.get("spread", 0)),
            end_date=raw.get("endDate", ""),
            tags=raw.get("tags", []),
            neg_risk=raw.get("negRisk", False),
        )

    def _filter_markets(self, markets: List[PolymarketMarket]) -> List[PolymarketMarket]:
        """Filter markets by volume, liquidity, category, and time to resolution."""
        cfg = self.config
        now = datetime.now(timezone.utc)
        filtered = []

        for m in markets:
            if m.volume_24h < cfg.min_volume_24h:
                continue
            if m.liquidity < cfg.min_liquidity:
                continue

            # Time to resolution check
            if cfg.min_time_to_resolution_hours > 0 and m.end_date:
                try:
                    end_dt = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
                    hours_remaining = (end_dt - now).total_seconds() / 3600
                    if hours_remaining < cfg.min_time_to_resolution_hours:
                        continue
                except (ValueError, TypeError):
                    pass  # Keep market if end_date can't be parsed

            # Category filter
            if cfg.categories is not None:
                market_labels = {tag.get("label", "") for tag in m.tags if isinstance(tag, dict)}
                if not market_labels.intersection(cfg.categories):
                    continue

            # Keyword filter (case-insensitive substring on question or slug)
            if cfg.keyword:
                needle = cfg.keyword.lower()
                if needle not in m.question.lower() and needle not in m.slug.lower():
                    continue

            filtered.append(m)

        # Sort by 24h volume descending and cap at max_markets
        filtered.sort(key=lambda m: m.volume_24h, reverse=True)
        return filtered[: cfg.max_markets]

    def scan(self) -> List[PolymarketMarket]:
        """Fetch active markets from Gamma API, parse and filter them."""
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={"active": "true", "closed": "false", "limit": 100},
                timeout=15,
            )
            resp.raise_for_status()
            raw_markets = resp.json()
        except Exception:
            logger.exception("Failed to fetch markets from Gamma API")
            return []

        markets = []
        for raw in raw_markets:
            if not raw.get("active", False) or raw.get("closed", False):
                continue
            try:
                markets.append(self._parse_market(raw))
            except (KeyError, json.JSONDecodeError, IndexError):
                logger.warning("Failed to parse market: %s", raw.get("id", "unknown"))
                continue

        return self._filter_markets(markets)
