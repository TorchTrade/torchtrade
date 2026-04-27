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
    def test_scan_filters_closed_markets(self, mock_get):
        """Closed markets are filtered out client-side. Inactive markets are kept
        because short-cadence markets (e.g. ``btc-updown-5m-``) sit listed-but-inactive
        until shortly before resolution and Gamma still includes them."""
        active_market = _make_raw_market(market_id="1", active=True, closed=False)
        inactive_market = _make_raw_market(market_id="2", active=False, closed=False)
        closed_market = _make_raw_market(market_id="3", active=True, closed=True)

        mock_resp = MagicMock()
        mock_resp.json.return_value = [active_market, inactive_market, closed_market]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0
        ))
        ids = sorted(m.market_id for m in scanner.scan())
        assert ids == ["1", "2"]  # closed filtered, inactive kept

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_returns_empty_on_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        scanner = MarketScanner(MarketScannerConfig())
        results = scanner.scan()
        assert results == []

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_switches_to_endDate_sort_when_targeting_upcoming(self, mock_get):
        """slug_prefix or max_resolution_minutes triggers chronological sort
        + end_date_min filter so short-cadence markets surface (they have $0
        volume and never make the volume-sorted top page)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = MarketScanner(MarketScannerConfig(slug_prefix="btc-updown-5m-"))
        scanner.scan()
        params = mock_get.call_args.kwargs["params"]
        assert params["order"] == "endDate"
        assert params["ascending"] == "true"
        assert "end_date_min" in params

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_scan_skips_malformed_markets_keeping_good_ones(self, mock_get):
        """A malformed entry (missing required fields) must not poison the batch,
        the rest of the response should still surface."""
        good = _make_raw_market(market_id="ok")
        malformed = {"id": "bad", "active": True, "closed": False}  # missing every field
        mock_resp = MagicMock()
        mock_resp.json.return_value = [malformed, good]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0
        ))
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].market_id == "ok"


class TestSlugPrefix:
    @pytest.mark.parametrize(
        "prefix,slug,expected_count",
        [
            (None, "btc-updown-5m-1234", 1),
            ("btc-updown-5m-", "btc-updown-5m-1234", 1),
            ("btc-updown-15m-", "btc-updown-5m-1234", 0),
            ("BTC-UPDOWN-5M-", "btc-updown-5m-1234", 0),  # case-sensitive
            ("", "btc-updown-5m-1234", 1),                 # empty falsy → no filter
        ],
        ids=["no-filter", "matches", "no-match", "case-sensitive", "empty"],
    )
    def test_slug_prefix_filtering(self, prefix, slug, expected_count):
        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0,
            slug_prefix=prefix,
        ))
        markets = [scanner._parse_market(_make_raw_market(slug=slug))]
        assert len(scanner._filter_markets(markets)) == expected_count


class TestAutoRelaxMinTimeWhenTargetingUpcoming:
    """Default ``min_time_to_resolution_hours=24`` would silently filter every
    short-cadence market out. When the user has opted into upcoming targeting
    (``slug_prefix`` set or ``max_time_to_resolution_minutes`` set), the
    scanner auto-relaxes the minimum to 0 so a one-line config still works.
    """

    @pytest.mark.parametrize(
        "config_kwargs,expected_count",
        [
            # No targeting → 24h floor enforced (5-min market filtered out).
            ({}, 0),
            # slug_prefix set → floor relaxed.
            ({"slug_prefix": "btc-updown-5m-"}, 1),
            # max_time_to_resolution_minutes set → floor relaxed.
            ({"max_time_to_resolution_minutes": 30}, 1),
            # Both set → still relaxed.
            ({"slug_prefix": "btc-updown-5m-", "max_time_to_resolution_minutes": 30}, 1),
        ],
        ids=["browse-default-filters-out", "slug-prefix-relaxes",
             "max-mins-relaxes", "both-relax"],
    )
    def test_min_time_auto_relaxes_for_upcoming_targeting(
        self, config_kwargs, expected_count
    ):
        from datetime import datetime, timedelta, timezone
        # 5-min market resolving in 5 minutes (well under the default 24h floor).
        end = datetime.now(timezone.utc) + timedelta(minutes=5)
        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0,
            min_liquidity=0,
            **config_kwargs,
        ))
        raw = _make_raw_market(
            slug="btc-updown-5m-1234",
            end_date=end.isoformat().replace("+00:00", "Z"),
        )
        markets = [scanner._parse_market(raw)]
        assert len(scanner._filter_markets(markets)) == expected_count


class TestMaxResolutionMinutes:
    @pytest.mark.parametrize(
        "max_minutes,end_offset_seconds,expected_count",
        [
            (None, 600, 1),    # no upper bound
            (60, 600, 1),      # under cap (10 min)
            (60, 3700, 0),     # over cap (62 min)
            (5, 600, 0),       # tighter cap excludes
        ],
        ids=["no-cap", "under-cap", "over-cap", "tight-cap"],
    )
    def test_upper_bound_filtering(self, max_minutes, end_offset_seconds, expected_count):
        from datetime import datetime, timedelta, timezone
        end = (datetime.now(timezone.utc) + timedelta(seconds=end_offset_seconds))
        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0,
            max_time_to_resolution_minutes=max_minutes,
        ))
        markets = [scanner._parse_market(
            _make_raw_market(end_date=end.isoformat().replace("+00:00", "Z"))
        )]
        assert len(scanner._filter_markets(markets)) == expected_count

    def test_drops_markets_already_past_end_date(self):
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(minutes=5))
        scanner = MarketScanner(MarketScannerConfig(
            min_volume_24h=0, min_liquidity=0, min_time_to_resolution_hours=0,
        ))
        markets = [scanner._parse_market(
            _make_raw_market(end_date=past.isoformat().replace("+00:00", "Z"))
        )]
        assert scanner._filter_markets(markets) == []


class TestNextActiveMarket:
    """Verify next_active_market hits the correct endpoint and matches by prefix."""

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_returns_first_matching_market(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            _make_raw_market(slug="other-market-1234", market_id="1"),
            _make_raw_market(slug="btc-updown-5m-1111", market_id="2"),
            _make_raw_market(slug="btc-updown-5m-2222", market_id="3"),
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        scanner = MarketScanner()
        result = scanner.next_active_market("btc-updown-5m-")
        assert result is not None
        assert result.market_id == "2"  # first matching, not first in list

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_returns_none_when_no_match(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [_make_raw_market(slug="something-else-1234")]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        assert MarketScanner().next_active_market("btc-updown-5m-") is None

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_returns_none_on_api_error(self, mock_get):
        mock_get.side_effect = Exception("503")
        assert MarketScanner().next_active_market("btc-updown-5m-") is None

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_skips_closed_markets(self, mock_get):
        """A closed market with a matching slug should be skipped, the env
        only wants markets that have not yet resolved."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            _make_raw_market(slug="btc-updown-5m-1111", market_id="1", closed=True),
            _make_raw_market(slug="btc-updown-5m-2222", market_id="2", closed=False),
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = MarketScanner().next_active_market("btc-updown-5m-")
        assert result is not None
        assert result.market_id == "2"

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_pins_query_params_for_upcoming_lookup(self, mock_get):
        """next_active_market is hit every step by the env; pin its outgoing
        query params so a refactor that drops end_date_min or flips the sort
        doesn't silently regress."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        MarketScanner().next_active_market("btc-updown-5m-")
        params = mock_get.call_args.kwargs["params"]
        assert params["closed"] == "false"
        assert params["order"] == "endDate"
        assert params["ascending"] == "true"
        assert "end_date_min" in params

    @patch("torchtrade.envs.live.polymarket.market_scanner.requests.get")
    def test_skips_malformed_then_returns_next_match(self, mock_get):
        """Malformed prefix-matching market must not abort the lookup, keep going."""
        malformed = {
            "id": "bad",
            "slug": "btc-updown-5m-bad",
            "active": True,
            "closed": False,
        }  # missing outcomePrices etc.
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            malformed,
            _make_raw_market(slug="btc-updown-5m-good", market_id="good"),
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = MarketScanner().next_active_market("btc-updown-5m-")
        assert result is not None
        assert result.market_id == "good"
