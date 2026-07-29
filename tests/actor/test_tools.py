"""Tests for LLM actor tools."""
import pytest

from torchtrade.actor.tools import GoogleNewsTool, symbol_to_query


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
