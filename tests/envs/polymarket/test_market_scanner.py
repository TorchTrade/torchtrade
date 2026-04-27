"""Tests for Polymarket market scanner."""

import json
from unittest.mock import MagicMock, patch

import pytest

from torchtrade.envs.live.polymarket.market_scanner import (
    MarketScanner,
    MarketScannerConfig,
    PolymarketMarket,
)


def _make_raw_market(
    market_id="517310",
    question="Will Bitcoin exceed $100k by March 2026?",
    condition_id="0xcond1",
    slug="will-bitcoin-exceed-100k",
    yes_price="0.72",
    no_price="0.28",
    volume="1500000",
    volume_24hr=50000.0,
    liquidity=200000.0,
    active=True,
    closed=False,
    end_date="2027-03-01T00:00:00Z",
    yes_token="token_yes_1",
    no_token="token_no_1",
    description="Resolves YES if BTC price exceeds $100,000.",
    tags=None,
    neg_risk=False,
    spread=0.02,
):
    """Build a raw Gamma API market dict."""
    return {
        "id": market_id,
        "question": question,
        "conditionId": condition_id,
        "slug": slug,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": json.dumps([yes_price, no_price]),
        "volume": volume,
        "volume24hr": volume_24hr,
        "liquidity": liquidity,
        "active": active,
        "closed": closed,
        "endDate": end_date,
        "clobTokenIds": json.dumps([yes_token, no_token]),
        "description": description,
        "tags": tags or [],
        "negRisk": neg_risk,
        "spread": spread,
    }


class TestParseMarket:
    """Verify parsing raw Gamma API response into PolymarketMarket."""

    def test_parse_market_extracts_all_fields(self):
        raw = _make_raw_market()
        scanner = MarketScanner(MarketScannerConfig())
        market = scanner._parse_market(raw)

        assert market.market_id == "517310"
        assert market.condition_id == "0xcond1"
        assert market.question == "Will Bitcoin exceed $100k by March 2026?"
        assert market.slug == "will-bitcoin-exceed-100k"
        assert market.yes_token_id == "token_yes_1"
        assert market.no_token_id == "token_no_1"
        assert market.yes_price == pytest.approx(0.72)
        assert market.no_price == pytest.approx(0.28)
        assert market.volume_24h == pytest.approx(50000.0)
        assert market.total_volume == pytest.approx(1500000.0)
        assert market.liquidity == pytest.approx(200000.0)
        assert market.spread == pytest.approx(0.02)
        assert market.end_date == "2027-03-01T00:00:00Z"
        assert market.description == "Resolves YES if BTC price exceeds $100,000."
        assert market.neg_risk is False

    def test_parse_market_handles_string_volume(self):
        """volume field comes as string from API; volume24hr/liquidity as float."""
        raw = _make_raw_market(volume="999.5", volume_24hr=100.0, liquidity=50.0)
        scanner = MarketScanner(MarketScannerConfig())
        market = scanner._parse_market(raw)

        assert market.total_volume == pytest.approx(999.5)
        assert market.volume_24h == pytest.approx(100.0)
        assert market.liquidity == pytest.approx(50.0)

    def test_parse_market_handles_missing_optional_fields(self):
        """Markets may lack tags or spread."""
        raw = _make_raw_market()
        del raw["tags"]
        del raw["spread"]
        scanner = MarketScanner(MarketScannerConfig())
        market = scanner._parse_market(raw)

        assert market.tags == []
        assert market.spread == 0.0


class TestFilterMarkets:
    """Parametrized tests for volume, liquidity, category, and time filtering."""

    @pytest.mark.parametrize(
        "min_vol,min_liq,vol_24hr,liq,expected_count",
        [
            (10_000, 5_000, 50_000.0, 200_000.0, 1),   # passes both
            (100_000, 5_000, 50_000.0, 200_000.0, 0),   # fails volume
            (10_000, 500_000, 50_000.0, 200_000.0, 0),   # fails liquidity
            (100_000, 500_000, 50_000.0, 200_000.0, 0),  # fails both
            (0, 0, 0.0, 0.0, 1),                          # zero thresholds pass
        ],
        ids=[
            "passes-both",
            "fails-volume",
            "fails-liquidity",
            "fails-both",
            "zero-thresholds",
        ],
    )
    def test_volume_and_liquidity_thresholds(
        self, min_vol, min_liq, vol_24hr, liq, expected_count
    ):
        config = MarketScannerConfig(min_volume_24h=min_vol, min_liquidity=min_liq)
        scanner = MarketScanner(config)
        raw = _make_raw_market(volume_24hr=vol_24hr, liquidity=liq)
        markets = [scanner._parse_market(raw)]
        filtered = scanner._filter_markets(markets)
        assert len(filtered) == expected_count

    def test_filter_excludes_near_resolution(self):
        """Markets ending within min_time_to_resolution_hours are excluded."""
        config = MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            min_time_to_resolution_hours=24 * 365,  # 1 year minimum
        )
        scanner = MarketScanner(config)
        # End date is far in the past
        raw = _make_raw_market(end_date="2020-01-01T00:00:00Z")
        markets = [scanner._parse_market(raw)]
        filtered = scanner._filter_markets(markets)
        assert len(filtered) == 0

    def test_filter_keeps_distant_resolution(self):
        """Markets far from resolution pass the filter."""
        config = MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            min_time_to_resolution_hours=24,
        )
        scanner = MarketScanner(config)
        raw = _make_raw_market(end_date="2030-01-01T00:00:00Z")
        markets = [scanner._parse_market(raw)]
        filtered = scanner._filter_markets(markets)
        assert len(filtered) == 1

    @pytest.mark.parametrize(
        "categories,tags,expected_count",
        [
            (None, [{"label": "Crypto"}], 1),             # no filter => passes
            (["Crypto"], [{"label": "Crypto"}], 1),        # matches category
            (["Politics"], [{"label": "Crypto"}], 0),      # no match
            (["Crypto", "Politics"], [{"label": "Crypto"}], 1),  # one matches
            (["Crypto"], [], 0),                           # no tags => fails
        ],
        ids=[
            "no-category-filter",
            "matches-category",
            "no-match",
            "one-of-many-matches",
            "empty-tags-fails",
        ],
    )
    def test_category_filtering(self, categories, tags, expected_count):
        config = MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            min_time_to_resolution_hours=0,
            categories=categories,
        )
        scanner = MarketScanner(config)
        raw = _make_raw_market(tags=tags)
        markets = [scanner._parse_market(raw)]
        filtered = scanner._filter_markets(markets)
        assert len(filtered) == expected_count

    @pytest.mark.parametrize(
        "keyword,question,slug,expected_count",
        [
            (None, "Will Bitcoin hit $100k?", "btc-100k", 1),
            ("bitcoin", "Will Bitcoin hit $100k?", "btc-100k", 1),
            ("BITCOIN", "Will Bitcoin hit $100k?", "btc-100k", 1),
            ("btc", "Will Bitcoin hit $100k?", "btc-100k", 1),
            ("ethereum", "Will Bitcoin hit $100k?", "btc-100k", 0),
            ("", "Anything", "any-slug", 1),
            (["btc", "bitcoin", "crypto"], "Will Bitcoin hit $100k?", "btc-100k", 1),
            (["eth", "ethereum"], "Will Bitcoin hit $100k?", "btc-100k", 0),
            (["ethereum", "btc"], "Will Bitcoin hit $100k?", "btc-100k", 1),
            ([], "Anything", "any-slug", 1),
        ],
        ids=[
            "none-passes",
            "match-question",
            "case-insensitive",
            "match-slug-only",
            "no-match",
            "empty-string-passes",
            "list-any-matches",
            "list-no-match",
            "list-second-matches",
            "empty-list-passes",
        ],
    )
    def test_keyword_filtering(self, keyword, question, slug, expected_count):
        config = MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            min_time_to_resolution_hours=0,
            keyword=keyword,
        )
        scanner = MarketScanner(config)
        raw = _make_raw_market(question=question, slug=slug)
        markets = [scanner._parse_market(raw)]
        assert len(scanner._filter_markets(markets)) == expected_count

    def test_max_markets_limits_output(self):
        config = MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            min_time_to_resolution_hours=0,
            max_markets=2,
        )
        scanner = MarketScanner(config)
        raws = [
            _make_raw_market(market_id=str(i), volume_24hr=float(1000 - i))
            for i in range(5)
        ]
        markets = [scanner._parse_market(r) for r in raws]
        filtered = scanner._filter_markets(markets)
        assert len(filtered) == 2
        # Sorted by volume_24h descending, top 2
        assert filtered[0].volume_24h >= filtered[1].volume_24h


class TestScan:
    """Verify scan() calls Gamma API, parses, and filters."""

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_calls_gamma_api(self, mock_get):
        raw_market = _make_raw_market()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [raw_market]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0
        )
        scanner = MarketScanner(config)
        results = scanner.scan()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "gamma-api.polymarket.com" in call_args[0][0]
        # Without sorting by volume24hr at the API level, niche-topic markets
        # (e.g. crypto) get pushed past the limit window. Pin the sort params
        # so a refactor doesn't silently regress to the unsorted default.
        params = call_args.kwargs["params"]
        assert params["order"] == "volume24hr"
        assert params["ascending"] == "false"
        assert len(results) == 1
        assert isinstance(results[0], PolymarketMarket)
        assert results[0].market_id == "517310"

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_filters_inactive_and_closed(self, mock_get):
        active_market = _make_raw_market(market_id="1", active=True, closed=False)
        inactive_market = _make_raw_market(market_id="2", active=False, closed=False)
        closed_market = _make_raw_market(market_id="3", active=True, closed=True)

        mock_resp = MagicMock()
        mock_resp.json.return_value = [active_market, inactive_market, closed_market]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        config = MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0
        )
        scanner = MarketScanner(config)
        results = scanner.scan()

        assert len(results) == 1
        assert results[0].market_id == "1"

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_returns_empty_on_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        scanner = MarketScanner(MarketScannerConfig())
        results = scanner.scan()
        assert results == []
