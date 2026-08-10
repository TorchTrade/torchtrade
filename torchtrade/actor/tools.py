"""External information tools the LLM trading actor can call mid-reasoning."""
from typing import Optional
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


class PolymarketTool(Tool):
    """Prediction-market odds related to the traded symbol (free Gamma API)."""

    name = "polymarket"
    description = (
        "polymarket(query?: str): prediction-market odds related to the traded asset "
        "(crowd probabilities, not a trade signal)"
    )

    def __init__(
        self,
        symbol: str,
        top_n: int = 5,
        min_volume_24h: float = 10_000.0,
        min_liquidity: float = 5_000.0,
        question_chars: int = 140,
    ):
        self.symbol = symbol
        self.top_n = top_n
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity
        self.question_chars = question_chars

    def _scan(self, keyword: str) -> list:
        """Fetch matching markets. Thin seam over the live-env Gamma scanner,
        which already owns the retry policy, parsing and filtering."""
        # lazy: torchtrade.actor.tools imports without the live-env stack
        from torchtrade.envs.live.polymarket import market_scanner as ms

        return ms.MarketScanner(ms.MarketScannerConfig(
            keyword=keyword,
            max_markets=self.top_n,
            min_volume_24h=self.min_volume_24h,
            min_liquidity=self.min_liquidity,
            # The scanner defaults to a 24h floor, tuned for slow discovery. A
            # market resolving in six hours is the most decision-relevant thing
            # we can show an intraday agent, so keep everything unresolved.
            min_time_to_resolution_hours=0,
        )).scan()

    def run(self, query: Optional[str] = None) -> str:
        q = query or symbol_to_query(self.symbol)
        try:
            markets = self._scan(q)
        except Exception as exc:  # never raise into a live trading step
            return f"error: polymarket unavailable ({exc})"
        if not markets:
            return f"No prediction markets for '{q}'."
        lines = [f"Prediction markets for '{q}':"]
        for i, m in enumerate(markets, 1):
            question = m.question
            if len(question) > self.question_chars:
                question = question[: self.question_chars] + "…"
            lines.append(
                f"{i}. {question} — "
                f"YES {m.yes_price * 100:.0f}% · 24h vol ${m.volume_24h:,.0f}"
            )
        return "\n".join(lines)
