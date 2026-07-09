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
