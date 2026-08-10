"""External information tools the LLM trading actor can call mid-reasoning."""
from typing import Optional
from urllib.parse import quote_plus

SYMBOL_QUERY_MAP = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
    "XRP": "XRP", "ADA": "Cardano",
}

# Longest free-text field rendered into the model's context, per field.
_MAX_TEXT_CHARS = 140


def _one_line(text: str) -> str:
    """Collapse text to a single capped line before it enters the model context.

    Applied to both free-text fields PolymarketTool renders. A newline lets one
    field occupy two numbered rows and fabricate a market the model then treats
    as tool-verified; the cap stops one field flooding the prompt.

    Scoped to row forgery and length only -- it does not neutralise inline
    markup such as a literal </tool_results>, which would still close the
    results block early. GoogleNewsTool does not use this yet (see #308).
    """
    text = " ".join(text.split())
    return text[:_MAX_TEXT_CHARS] + "…" if len(text) > _MAX_TEXT_CHARS else text


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
        timeout: float = 5.0,
    ):
        self.symbol = symbol
        self.top_n = top_n
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity
        self.timeout = timeout

    def _scan(self, keyword: str) -> list:
        """Fetch matching markets. Thin seam over the live-env Gamma scanner,
        which already owns the fetch, retry, parsing and filtering machinery."""
        # lazy: torchtrade.actor.tools imports without the live-env stack
        from torchtrade.envs.live.polymarket import market_scanner as ms

        config = ms.MarketScannerConfig(
            keyword=keyword,
            max_markets=self.top_n,
            min_volume_24h=self.min_volume_24h,
            min_liquidity=self.min_liquidity,
            # Scanner defaults to a 24h floor; an intraday agent wants exactly
            # the soon-resolving markets that floor hides.
            min_time_to_resolution_hours=0,
            # Far below the scanner's default budget: this call blocks a
            # collector step, not a 5-minute live loop.
            timeout=self.timeout,
            retry_attempts=2,
        )
        return ms.MarketScanner(config).scan()

    def run(self, query: Optional[str] = None) -> str:
        try:
            # Inside the guard: `query` arrives unvalidated from the model, so
            # normalising it can itself raise on a non-string.
            q = _one_line(query or symbol_to_query(self.symbol))
            markets = self._scan(q)
        except Exception as exc:  # never raise into a live trading step
            return f"error: polymarket unavailable ({exc})"
        if not markets:
            # MarketScanner.scan() logs and returns [] on fetch failure, so we
            # cannot tell an outage from a genuinely empty result. Never assert
            # an absence we did not verify -- "no markets on this asset" is
            # itself a signal the model will trade on.
            return f"No Polymarket markets matched '{q}' (none open, or Gamma unavailable)."
        lines = [f"Prediction markets for '{q}':"]
        for i, m in enumerate(markets[: self.top_n], 1):
            lines.append(
                f"{i}. {_one_line(m.question)} — "
                f"YES {m.yes_price * 100:.1f}% · 24h vol ${m.volume_24h:,.0f}"
            )
        return "\n".join(lines)
