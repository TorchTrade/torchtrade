"""Tests for LLM actor tools."""
import pytest

from torchtrade.actor.tools import AdanosSentimentTool, GoogleNewsTool, symbol_to_query


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


class _SentimentResult:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


def _fake_adanos(monkeypatch, result):
    import sys
    import types

    calls = {}

    class _Namespace:
        def __init__(self, source):
            self.source = source

        def stock(self, symbol, **period):
            calls["stock"] = (self.source, symbol, period)
            if isinstance(result, Exception):
                raise result
            return result

        def token(self, symbol, **period):
            calls["token"] = (self.source, symbol, period)
            if isinstance(result, Exception):
                raise result
            return result

    class _Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.reddit = _Namespace("reddit")
            self.x = _Namespace("x")
            self.news = _Namespace("news")
            self.polymarket = _Namespace("polymarket")
            self.crypto = _Namespace("crypto")

        def close(self):
            calls["closed"] = True

    module = types.ModuleType("adanos")
    module.AdanosClient = _Client
    monkeypatch.setitem(sys.modules, "adanos", module)
    return calls


def test_adanos_stock_uses_source_dates_and_closes_client(monkeypatch):
    result = _SentimentResult({"ticker": "AAPL", "sentiment_score": 0.4, "top_mentions": ["omit"]})
    calls = _fake_adanos(monkeypatch, result)
    tool = AdanosSentimentTool(
        symbol="AAPL/USD",
        asset_type="stock",
        source="news",
        api_key="test-key",
        timeout=2.5,
    )

    out = tool.run(from_date="2026-08-01", to_date="2026-08-08")

    assert calls["client"] == {"api_key": "test-key", "timeout": 2.5}
    assert calls["stock"] == (
        "news", "AAPL", {"from_": "2026-08-01", "to": "2026-08-08"}
    )
    assert calls["closed"] is True
    assert '"sentiment_score": 0.4' in out
    assert "top_mentions" not in out


def test_adanos_crypto_uses_token_endpoint(monkeypatch):
    calls = _fake_adanos(monkeypatch, _SentimentResult({"symbol": "BTC", "buzz_score": 72.0}))
    tool = AdanosSentimentTool(
        symbol="BTC/USD",
        asset_type="crypto",
        api_key="test-key",
    )

    out = tool.run()

    assert calls["token"] == ("crypto", "BTC", {})
    assert '"asset_type": "crypto"' in out


def test_adanos_missing_key_returns_error_without_fetch(monkeypatch):
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)
    tool = AdanosSentimentTool(symbol="AAPL", asset_type="stock")
    monkeypatch.setattr(tool, "_fetch", lambda *args: pytest.fail("must not fetch"))

    assert "requires ADANOS_API_KEY" in tool.run()


def test_adanos_failure_returns_error_string(monkeypatch):
    calls = _fake_adanos(monkeypatch, TimeoutError("timed out"))
    tool = AdanosSentimentTool(symbol="AAPL", asset_type="stock", api_key="test-key")

    assert "error" in tool.run().lower()
    assert calls["closed"] is True


def test_adanos_rejects_non_reddit_crypto_source():
    with pytest.raises(ValueError, match="Reddit"):
        AdanosSentimentTool(symbol="BTC/USD", asset_type="crypto", source="news")
