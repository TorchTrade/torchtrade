"""External information tools the LLM trading actor can call mid-reasoning."""
import re
from typing import Optional
from urllib.parse import quote_plus

SYMBOL_QUERY_MAP = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
    "XRP": "XRP", "ADA": "Cardano",
}

# Longest free-text field rendered into the model's context, per field.
_MAX_TEXT_CHARS = 140

# Every tag the actor's protocol gives meaning to, not just the block delimiters. A
# forged <answer>N</answer> in a headline sits in context as a completed trade -- it
# cannot reach a Python parser (those run on responses only), so it is persuasion rather
# than parser bypass, but a tool has no legitimate reason to emit any of these.
#
# Matched the way the CONSUMER reads them rather
# than the way we emit them. The consumer is an LLM, so it is maximally lenient: case,
# stray whitespace and trailing attributes all still read as a closing tag. This repo
# already concedes the point -- parsers.py compiles extract_action with re.IGNORECASE.
# Escaping the two exact literals left </TOOL_RESULTS> and </tool_results > working.
# `[\s/]*` as ONE class, not `\s*/?\s*`: two adjacent \s* around an optional slash is
# ambiguous, so a whitespace run has O(n) split points and the match goes quadratic --
# 52 SECONDS on a 60k run, inside a live trading step. `result` deliberately bypasses
# _one_line, so a tool can emit one.
# The trailing `>` is optional: the consumer is an LLM, and `</tool_results` at a line
# end reads as a close to it.
_TAG = r"<[\s/]*(?:%s)\b[^>\n]*>?"

# Tool output: every tag the protocol gives meaning to. A hostile RSS headline has no
# legitimate reason to carry any of them.
_PROTOCOL_TAG_RE = re.compile(_TAG % "tool_results|answer|tool|think", re.IGNORECASE)

# The model's OWN reply: delimiters only. <think> and <tool ...> are what the prompt
# instructs it to emit, and escaping those showed the model its prior turn in a mangled
# convention on every tool round -- few-shot imitation then yields &lt;answer&gt;N...,
# which extract_action cannot parse, so it warns and returns 0. On futures action_levels
# that is a FULL SHORT. Only the delimiters can move the trusted boundary.
_BLOCK_MARKER_RE = re.compile(_TAG % "tool_results", re.IGNORECASE)


def neutralise_block_markers(text: str) -> str:
    """Defuse any literal tool_results delimiter appearing inside tool output (#330).

    A forged closing tag is strictly worse than the forged ROW #308 fixed: a fake row
    adds an entry inside the trusted region, while a closing tag moves the boundary of
    the region itself, so everything after it reads to the model as its own reasoning.

    Applied once to the assembled body rather than per field, because the field-level
    helper deliberately does not see `result` -- and GoogleNewsTool renders titles
    straight from an RSS feed authored by whoever gets a headline indexed.
    """
    return _PROTOCOL_TAG_RE.sub(lambda m: "&lt;" + m.group(0)[1:], text)


def neutralise_boundary_markers(text: str) -> str:
    """Defuse only the block delimiters -- for text the MODEL authored (#330).

    Narrower than `neutralise_block_markers` on purpose: the model is instructed to emit
    <think> and <tool ...>, so escaping those in its own reply corrupts the convention it
    is being shown, on every tool round.
    """
    return _BLOCK_MARKER_RE.sub(lambda m: "&lt;" + m.group(0)[1:], text)


def _one_line(text: str) -> str:
    """Collapse text to a single capped line before it enters the model context.

    Applied to every free-text field PolymarketTool and GoogleNewsTool render. A
    newline lets one field occupy two numbered rows and fabricate an entry the
    model cannot distinguish from genuine tool output; the cap stops one field flooding the
    prompt.

    Scoped to row forgery and length only. Inline markup is handled once at the
    assembly seam by `neutralise_block_markers`, not per field: `result` is the
    tool's own string and never passes through here (#330).
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
        # Every feed field rendered below is collapsed to one line (#308). A newline in
        # any of them lets one entry occupy two numbered rows and fabricate a headline the
        # model cannot distinguish from genuine tool output. Unlike PolymarketTool's
        # query, title/source/published come from an RSS feed -- authored by whoever gets
        # a headline indexed by Google News -- so this is a third-party boundary, and the
        # news path has no volume floor or other content filter.
        try:
            # Inside the guard: `query` arrives unvalidated from the model, so
            # normalising it can itself raise on a non-string.
            q = _one_line(query or symbol_to_query(self.symbol))
            entries = self._fetch(q)
        except Exception as exc:  # never raise into a live trading step
            return f"error: google_news unavailable ({exc})"
        if not entries:
            return f"No recent news for '{q}'."
        lines = [f"Top news for '{q}':"]
        for i, e in enumerate(entries[: self.top_n], 1):
            title = _one_line(e.get("title", ""))
            source = _one_line(e.get("source", ""))
            published = _one_line(e.get("published", ""))
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
