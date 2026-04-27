"""List active Polymarket prediction markets that pass volume/liquidity/duration filters.

This script does not place any orders — it only reads from the public Gamma API.
It is the **discovery** half of the workflow: find a market series that interests
you, copy its slug stem, paste that into ``PolymarketBetEnvConfig.market_slug_prefix``.

Run with:
    # Top markets by 24h volume
    python examples/broker/polymarket/scan_markets.py

    # Filter by keyword (fuzzy substring match)
    python examples/broker/polymarket/scan_markets.py --keyword bitcoin
    python examples/broker/polymarket/scan_markets.py --keyword btc bitcoin crypto

    # Filter by slug prefix (exact, structural — same primitive the env uses)
    python examples/broker/polymarket/scan_markets.py --slug-prefix btc-updown-5m-

    # Find short-cadence markets (e.g. resolving in <30 min)
    python examples/broker/polymarket/scan_markets.py --max-resolution-minutes 30 --min-volume 0
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig


def _format_eta(end_date: str) -> str:
    """Format the resolution time as either a date or a `Xm` / `Xh` countdown."""
    if not end_date:
        return ""
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    except ValueError:
        return end_date[:16]
    minutes = (end_dt - datetime.now(timezone.utc)).total_seconds() / 60
    if minutes < 0:
        return f"{end_date[:16]} (past)"
    if minutes < 60:
        return f"in {minutes:>4.0f}m"
    if minutes < 60 * 24:
        return f"in {minutes / 60:>4.1f}h"
    return f"{end_date[:10]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keyword",
        nargs="+",
        default=None,
        metavar="WORD",
        help=(
            "One or more case-insensitive substrings to match against the market "
            "question or slug. A market passes if ANY keyword hits "
            "(e.g. --keyword btc bitcoin crypto)."
        ),
    )
    parser.add_argument(
        "--slug-prefix",
        type=str,
        default=None,
        metavar="STEM",
        help=(
            "Case-sensitive prefix match on the market slug — the same identifier "
            "PolymarketBetEnv uses. Use this once you know which series you want "
            "(e.g. --slug-prefix btc-updown-5m-)."
        ),
    )
    parser.add_argument(
        "--min-volume", type=float, default=0.0, help="Minimum 24h volume (USD)."
    )
    parser.add_argument(
        "--min-liquidity", type=float, default=0.0, help="Minimum liquidity (USD)."
    )
    parser.add_argument(
        "--min-resolution-hours",
        type=float,
        default=0.0,
        help="Minimum hours to resolution (default 0; raise to exclude very-short markets).",
    )
    parser.add_argument(
        "--max-resolution-minutes",
        type=float,
        default=None,
        help="Maximum minutes to resolution (default unset; set to e.g. 30 to find short-cadence markets).",
    )
    parser.add_argument(
        "--max", type=int, default=20, help="Maximum number of markets to print."
    )
    args = parser.parse_args()

    scanner = MarketScanner(
        MarketScannerConfig(
            min_volume_24h=args.min_volume,
            min_liquidity=args.min_liquidity,
            min_time_to_resolution_hours=args.min_resolution_hours,
            max_time_to_resolution_minutes=args.max_resolution_minutes,
            max_markets=args.max,
            keyword=args.keyword,
            slug_prefix=args.slug_prefix,
        )
    )
    markets = scanner.scan()
    if not markets:
        print("No markets matched the filters.")
        return

    print(
        f"{'YES':>5} | {'24h vol':>11} | {'liquidity':>11} | {'resolves':>14} | slug"
    )
    print("-" * 110)
    for m in markets:
        print(
            f"{m.yes_price:5.2f} | "
            f"${m.volume_24h:10,.0f} | "
            f"${m.liquidity:10,.0f} | "
            f"{_format_eta(m.end_date):>14} | "
            f"{m.slug}"
        )


if __name__ == "__main__":
    main()
