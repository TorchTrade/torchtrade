"""Polymarket market scanner — fetch and filter markets from Gamma API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Union

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
    max_time_to_resolution_minutes: Optional[float] = None
    # Single keyword or list — any substring match (case-insensitive) on question or slug
    keyword: Optional[Union[str, List[str]]] = None
    # Case-sensitive prefix match on the market slug. Discover the right prefix
    # via the discovery flow (scan_markets.py) and use it as the stable identifier
    # for short-cadence recurring series (e.g. "btc-updown-5m-").
    slug_prefix: Optional[str] = None


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
        """Filter markets by volume, liquidity, category, time, slug prefix, keyword."""
        cfg = self.config
        now = datetime.now(timezone.utc)
        filtered = []

        for m in markets:
            if m.volume_24h < cfg.min_volume_24h:
                continue
            if m.liquidity < cfg.min_liquidity:
                continue

            # Resolution duration window — also drops markets already past their
            # endDate but not yet flagged closed (Polymarket leaves resolved
            # markets in this state for a while).
            if m.end_date:
                try:
                    end_dt = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
                    minutes_remaining = (end_dt - now).total_seconds() / 60
                except (ValueError, TypeError):
                    minutes_remaining = None
                if minutes_remaining is not None:
                    if minutes_remaining < 0:
                        continue
                    if minutes_remaining < cfg.min_time_to_resolution_hours * 60:
                        continue
                    if (
                        cfg.max_time_to_resolution_minutes is not None
                        and minutes_remaining > cfg.max_time_to_resolution_minutes
                    ):
                        continue

            # Category filter
            if cfg.categories is not None:
                market_labels = {tag.get("label", "") for tag in m.tags if isinstance(tag, dict)}
                if not market_labels.intersection(cfg.categories):
                    continue

            # Slug prefix (case-sensitive structural match) — used both for
            # discovery and as the env's stable series identifier.
            if cfg.slug_prefix and not m.slug.startswith(cfg.slug_prefix):
                continue

            # Keyword filter (case-insensitive substring on question or slug;
            # accepts a single string or a list — matches if ANY keyword hits)
            if cfg.keyword:
                needles = [cfg.keyword] if isinstance(cfg.keyword, str) else cfg.keyword
                haystack = (m.question + " " + m.slug).lower()
                if not any(n.lower() in haystack for n in needles if n):
                    continue

            filtered.append(m)

        # When the user is browsing for high-volume markets (no slug/duration
        # focus), sort by 24h volume desc. When they're targeting short-cadence
        # series, the API has already returned them in chronological order —
        # preserve that so the next-to-resolve markets surface first.
        if cfg.slug_prefix is None and cfg.max_time_to_resolution_minutes is None:
            filtered.sort(key=lambda m: m.volume_24h, reverse=True)
        return filtered[: cfg.max_markets]

    def scan(self) -> List[PolymarketMarket]:
        """Fetch active markets from Gamma API, parse and filter them.

        Two query strategies depending on configuration:

        * **No ``slug_prefix``** — sort by 24h volume descending so high-volume
          markets surface first (general discovery / browsing).
        * **``slug_prefix`` set** — sort by ``endDate`` ascending and use the
          ``end_date_min=now`` server-side filter so short-cadence series
          (e.g. ``btc-updown-5m-``, which typically have $0 volume until the
          last seconds before resolution) are surfaced reliably.

        For finding the single soonest-resolving match (used by
        :class:`PolymarketBetEnv` each step), call :meth:`next_active_market`.
        """
        # When the user is targeting upcoming/short-cadence markets (slug_prefix
        # set, or an upper resolution bound configured), sort by endDate
        # ascending and use the server-side end_date_min filter — short-cadence
        # markets (5m/15m crypto) typically have $0 volume so they never make
        # the volume-sorted top page.
        targeting_upcoming = bool(
            self.config.slug_prefix
            or self.config.max_time_to_resolution_minutes is not None
        )
        if targeting_upcoming:
            params = {
                "closed": "false",
                "limit": 500,
                "order": "endDate",
                "ascending": "true",
                "end_date_min": datetime.now(timezone.utc).isoformat(),
            }
        else:
            params = {
                "active": "true",
                "closed": "false",
                "limit": 500,
                "order": "volume24hr",
                "ascending": "false",
            }
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets", params=params, timeout=15
            )
            resp.raise_for_status()
            raw_markets = resp.json()
        except Exception:
            logger.exception("Failed to fetch markets from Gamma API")
            return []

        markets = []
        for raw in raw_markets:
            if raw.get("closed", False):
                continue
            try:
                markets.append(self._parse_market(raw))
            except (KeyError, json.JSONDecodeError, IndexError):
                logger.warning("Failed to parse market: %s", raw.get("id", "unknown"))
                continue

        return self._filter_markets(markets)

    def next_active_market(
        self, slug_prefix: str, lookahead: int = 500
    ) -> Optional[PolymarketMarket]:
        """Return the soonest-resolving active market whose slug starts with ``slug_prefix``.

        Used by :class:`PolymarketBetEnv` to pick the next bet each step.
        Queries Gamma sorted by ``endDate`` ascending with ``end_date_min=now``
        so already-resolved-but-still-flagged-active markets are skipped at
        the API level, then takes the first slug-matching entry.

        Returns ``None`` if no matching market exists in the lookahead window.
        """
        now = datetime.now(timezone.utc)
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={
                    "closed": "false",
                    "limit": lookahead,
                    "order": "endDate",
                    "ascending": "true",
                    "end_date_min": now.isoformat(),
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw_markets = resp.json()
        except Exception:
            logger.exception("Failed to fetch upcoming markets from Gamma API")
            return None

        for raw in raw_markets:
            slug = raw.get("slug", "")
            if not slug.startswith(slug_prefix):
                continue
            if raw.get("closed", False):
                continue
            try:
                return self._parse_market(raw)
            except (KeyError, json.JSONDecodeError, IndexError):
                logger.warning("Failed to parse market: %s", raw.get("id", "unknown"))
                continue
        return None
