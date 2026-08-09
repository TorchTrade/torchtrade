"""External information tools the LLM trading actor can call mid-reasoning."""
import json
import os
from typing import Literal, Optional
from urllib.parse import quote_plus

SYMBOL_QUERY_MAP = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
    "XRP": "XRP", "ADA": "Cardano",
}


def symbol_to_query(symbol: str) -> str:
    """Map a trading symbol to a news search term ('BTC/USD' -> 'Bitcoin')."""
    base = symbol.split("/")[0].split("-")[0].upper()
    return SYMBOL_QUERY_MAP.get(base, base)


class Tool:
    """Minimal tool interface: a name, a one-line description, and run(**args)->str."""

    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> str:
        raise NotImplementedError


class GoogleNewsTool(Tool):
    """Top-N recent Google News headlines for the traded symbol (free RSS)."""

    name = "google_news"
    description = "google_news(query?: str): recent news headlines (defaults to the traded symbol)"

    def __init__(self, symbol: str, top_n: int = 5, timeout: float = 5.0):
        self.symbol = symbol
        self.top_n = top_n
        self.timeout = timeout

    def _fetch(self, query: str) -> list[dict]:
        """Fetch + normalize Google News RSS entries. Thin network seam (mocked in tests)."""
        import feedparser  # lazy: torchtrade.actor.tools imports without feedparser
        from urllib.request import urlopen

        url = (
            "https://news.google.com/rss/search?q="
            + quote_plus(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        # Bound the network request: feedparser.parse(url) would fetch with no
        # timeout, so a hung RSS connection could block a live trading decision
        # indefinitely. Fetch the bytes ourselves with self.timeout — a stall then
        # raises and run()'s guard degrades it to an error string.
        with urlopen(url, timeout=self.timeout) as resp:
            raw = resp.read()
        feed = feedparser.parse(raw)
        entries = []
        for e in feed.entries:
            entries.append({
                "title": getattr(e, "title", ""),
                "source": getattr(getattr(e, "source", None), "title", "") or "",
                "published": getattr(e, "published", ""),
            })
        return entries

    def run(self, query: Optional[str] = None) -> str:
        q = query or symbol_to_query(self.symbol)
        try:
            entries = self._fetch(q)
        except Exception as exc:  # never raise into a live trading step
            return f"error: google_news unavailable ({exc})"
        if not entries:
            return f"No recent news for '{q}'."
        lines = [f"Top news for '{q}':"]
        for i, e in enumerate(entries[: self.top_n], 1):
            title = e.get("title", "")
            source = e.get("source", "")
            published = e.get("published", "")
            lines.append(f"{i}. {title} — {source} · {published}".rstrip(" ·"))
        return "\n".join(lines)


class AdanosSentimentTool(Tool):
    """Structured Adanos sentiment context for the traded stock or crypto asset."""

    name = "adanos_sentiment"
    description = (
        "adanos_sentiment(from_date?: YYYY-MM-DD, to_date?: YYYY-MM-DD): "
        "market sentiment evidence for the traded asset; not a trade signal"
    )
    _STOCK_SOURCES = {"reddit", "x", "news", "polymarket"}
    _OUTPUT_FIELDS = (
        "ticker",
        "symbol",
        "company_name",
        "name",
        "found",
        "sentiment_score",
        "bullish_pct",
        "bearish_pct",
        "buzz_score",
        "mentions",
        "trend",
        "period_days",
    )

    def __init__(
        self,
        symbol: str,
        *,
        asset_type: Literal["stock", "crypto"],
        source: Literal["reddit", "x", "news", "polymarket"] = "reddit",
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ):
        if asset_type not in {"stock", "crypto"}:
            raise ValueError("asset_type must be 'stock' or 'crypto'")
        if source not in self._STOCK_SOURCES:
            raise ValueError(f"unsupported Adanos source: {source}")
        if asset_type == "crypto" and source != "reddit":
            raise ValueError("Adanos crypto sentiment currently uses the Reddit source")

        self.symbol = symbol.split("/")[0].split("-")[0].upper()
        self.asset_type = asset_type
        self.source = source
        self.api_key = api_key
        self.timeout = timeout

    def _fetch(self, api_key: str, from_date: Optional[str], to_date: Optional[str]):
        """Fetch one asset summary through the official SDK."""
        from adanos import AdanosClient

        period = {key: value for key, value in {"from_": from_date, "to": to_date}.items() if value}
        client = AdanosClient(api_key=api_key, timeout=self.timeout)
        try:
            if self.asset_type == "crypto":
                return client.crypto.token(self.symbol, **period)
            return getattr(client, self.source).stock(self.symbol, **period)
        finally:
            client.close()

    def run(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> str:
        api_key = self.api_key or os.getenv("ADANOS_API_KEY")
        if not api_key:
            return "error: adanos_sentiment requires ADANOS_API_KEY"

        try:
            result = self._fetch(api_key, from_date, to_date)
        except Exception as exc:  # never raise into a live trading step
            return f"error: adanos_sentiment unavailable ({exc})"

        if result is None:
            return f"No Adanos sentiment data for '{self.symbol}'."
        data = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(data, dict):
            return "error: adanos_sentiment returned an unexpected response"

        summary = {field: data[field] for field in self._OUTPUT_FIELDS if field in data}
        summary["asset_type"] = self.asset_type
        summary["source"] = self.source
        return "Adanos market sentiment: " + json.dumps(summary, ensure_ascii=True)
