"""Tests for LLM actor tools."""
import pytest

from torchtrade.actor.tools import GoogleNewsTool, PolymarketTool, symbol_to_query


@pytest.mark.parametrize("symbol,expected", [
    ("BTC/USD", "Bitcoin"), ("ETH/USD", "Ethereum"), ("DOGE/USD", "DOGE"),
], ids=["btc", "eth", "fallback"])
def test_symbol_to_query(symbol, expected):
    assert symbol_to_query(symbol) == expected


def _entries(n):
    return [{"title": f"headline {i}", "source": "Reuters", "published": "2h ago"} for i in range(n)]


def test_google_news_formats_top_n(monkeypatch):
    tool = GoogleNewsTool(symbol="BTC/USD", top_n=2)
    monkeypatch.setattr(tool, "_fetch", lambda query: _entries(5))
    out = tool.run()
    assert "headline 0" in out and "headline 1" in out
    assert "headline 2" not in out            # capped at top_n
    assert "Reuters" in out


def test_google_news_default_query_uses_symbol(monkeypatch):
    tool = GoogleNewsTool(symbol="ETH/USD")
    seen = {}
    monkeypatch.setattr(tool, "_fetch", lambda query: (seen.update({"q": query}), [])[1])
    tool.run()
    assert seen["q"] == "Ethereum"


def test_google_news_empty_results(monkeypatch):
    tool = GoogleNewsTool(symbol="BTC/USD")
    monkeypatch.setattr(tool, "_fetch", lambda query: [])
    assert "no recent news" in tool.run().lower()


def test_google_news_fetch_failure_returns_error_string(monkeypatch):
    tool = GoogleNewsTool(symbol="BTC/USD")
    def boom(query):
        raise ConnectionError("network down")
    monkeypatch.setattr(tool, "_fetch", boom)
    out = tool.run()                          # must NOT raise
    assert "error" in out.lower()


def _fake_feedparser(monkeypatch):
    """Inject a stub feedparser so _fetch's lazy import works without the real dep."""
    import sys
    import types
    fake = types.ModuleType("feedparser")
    fake.parse = lambda data: types.SimpleNamespace(entries=[])
    monkeypatch.setitem(sys.modules, "feedparser", fake)


def test_google_news_fetch_enforces_timeout(monkeypatch):
    """_fetch must bound the request with self.timeout so a hung RSS connection
    cannot block a live trading decision indefinitely."""
    _fake_feedparser(monkeypatch)
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<rss></rss>"

    def fake_urlopen(url, timeout=None):
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    GoogleNewsTool(symbol="BTC/USD", timeout=3.0)._fetch("Bitcoin")
    assert captured["timeout"] == 3.0


def test_google_news_timeout_returns_error_string(monkeypatch):
    """A network stall (urlopen raises after the timeout) degrades to an error
    string via run()'s guard — never hangs or raises into forward()."""
    _fake_feedparser(monkeypatch)

    def stall(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", stall)
    out = GoogleNewsTool(symbol="BTC/USD", timeout=0.01).run()
    assert "error" in out.lower()


def _market(question="Will Bitcoin exceed $100k by March 2026?", yes_price=0.72, volume_24h=50_000.0):
    """Build a PolymarketMarket the way MarketScanner.scan() would return it."""
    from torchtrade.envs.live.polymarket.market_scanner import PolymarketMarket

    return PolymarketMarket(
        market_id="1", condition_id="0x1", question=question, description="",
        slug="slug", yes_token_id="y", no_token_id="n",
        yes_price=yes_price, no_price=1.0 - yes_price,
        volume_24h=volume_24h, total_volume=1_500_000.0, liquidity=200_000.0,
        spread=0.02, end_date="2027-03-01T00:00:00Z", tags=[], neg_risk=False,
    )


def _fake_scanner(monkeypatch, result):
    """Swap MarketScanner for a stub; capture the config the tool builds.

    Mocks the scanner rather than HTTP: the Gamma client, its retry policy and
    its filtering are already covered by tests/envs/polymarket/.
    """
    captured = {}

    class _Scanner:
        def __init__(self, config):
            captured["config"] = config

        def scan(self):
            if isinstance(result, Exception):
                raise result
            return result

    import torchtrade.envs.live.polymarket.market_scanner as ms
    monkeypatch.setattr(ms, "MarketScanner", _Scanner)
    return captured


@pytest.mark.parametrize("symbol,query,expected", [
    ("BTC/USD", None, "Bitcoin"),          # symbol routed through symbol_to_query
    ("ETH/USD", None, "Ethereum"),
    ("BTC/USD", "Fed rate cut", "Fed rate cut"),  # explicit query wins over the symbol
])
def test_polymarket_keyword_comes_from_symbol_unless_query_given(
    monkeypatch, symbol, query, expected
):
    """The model can steer the search, but defaults to the traded asset."""
    captured = _fake_scanner(monkeypatch, [_market()])
    PolymarketTool(symbol=symbol).run(query=query)
    assert captured["config"].keyword == expected


def test_polymarket_forwards_caps_and_spam_thresholds_to_scanner(monkeypatch):
    """Volume/liquidity floors are the spam filter keeping junk markets out of
    the model's context — they must reach the scanner, not be dropped."""
    captured = _fake_scanner(monkeypatch, [])
    PolymarketTool(
        symbol="BTC/USD", top_n=3, min_volume_24h=1234.0, min_liquidity=567.0
    ).run()
    assert captured["config"].max_markets == 3
    assert captured["config"].min_volume_24h == 1234.0
    assert captured["config"].min_liquidity == 567.0


def test_polymarket_reports_probability_not_raw_price(monkeypatch):
    """A 0.72 YES price is a 72% probability — the model reads percentages."""
    _fake_scanner(monkeypatch, [_market(question="Will BTC hit 100k?", yes_price=0.72)])
    out = PolymarketTool(symbol="BTC/USD").run()
    assert "Will BTC hit 100k?" in out
    assert "72%" in out


def test_polymarket_truncates_long_question(monkeypatch):
    """Market questions are user-authored on Polymarket and land in the model's
    context verbatim — cap their length so one market can't flood the prompt."""
    long_question = "Q" * 400
    _fake_scanner(monkeypatch, [_market(question=long_question)])
    out = PolymarketTool(symbol="BTC/USD", question_chars=60).run()
    assert long_question not in out
    assert "Q" * 60 in out
    # Mark the cut: a question clipped mid-number ("...September 202") otherwise
    # reads to the model as a complete, and wrong, statement.
    assert "Q…" in out


def test_polymarket_short_question_is_not_marked_as_truncated(monkeypatch):
    """The ellipsis must mean something — only clipped questions carry it."""
    _fake_scanner(monkeypatch, [_market(question="Short one?")])
    out = PolymarketTool(symbol="BTC/USD", question_chars=60).run()
    assert "…" not in out


def test_polymarket_no_markets_returns_message(monkeypatch):
    """An empty result is not an error, and must not read as one."""
    _fake_scanner(monkeypatch, [])
    out = PolymarketTool(symbol="BTC/USD").run()
    assert "error" not in out.lower()
    assert "Bitcoin" in out


def test_polymarket_failure_returns_error_string(monkeypatch):
    """Fail-open: a scanner blow-up degrades to a string, never into a live step."""
    _fake_scanner(monkeypatch, RuntimeError("gamma down"))
    out = PolymarketTool(symbol="BTC/USD").run()
    assert out.startswith("error: polymarket")


def test_polymarket_does_not_hide_markets_resolving_within_a_day(monkeypatch):
    """MarketScannerConfig defaults to a 24h minimum time-to-resolution, built
    for slow discovery. An intraday agent cares most about markets resolving
    soon, so the tool must not inherit that floor."""
    captured = _fake_scanner(monkeypatch, [])
    PolymarketTool(symbol="BTC/USD").run()
    assert captured["config"].min_time_to_resolution_hours == 0
