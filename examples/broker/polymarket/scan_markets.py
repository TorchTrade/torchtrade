"""List active Polymarket prediction markets that pass volume/liquidity filters.

This script does not place any orders — it only reads from the public Gamma API.

Run with:
    python examples/broker/polymarket/scan_markets.py
    python examples/broker/polymarket/scan_markets.py --keyword bitcoin
    python examples/broker/polymarket/scan_markets.py --keyword "world cup" --max 5
"""

from __future__ import annotations

import argparse

from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Case-insensitive substring match on the market question or slug.",
    )
    parser.add_argument(
        "--min-volume", type=float, default=10_000.0, help="Minimum 24h volume (USD)."
    )
    parser.add_argument(
        "--min-liquidity", type=float, default=5_000.0, help="Minimum liquidity (USD)."
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Maximum number of markets to print."
    )
    args = parser.parse_args()

    scanner = MarketScanner(
        MarketScannerConfig(
            min_volume_24h=args.min_volume,
            min_liquidity=args.min_liquidity,
            min_time_to_resolution_hours=24,
            max_markets=args.max,
            keyword=args.keyword,
        )
    )
    markets = scanner.scan()
    if not markets:
        suffix = f" matching '{args.keyword}'" if args.keyword else ""
        print(f"No markets{suffix} matched the filters.")
        return

    print(f"{'YES':>5} | {'24h vol':>12} | {'liquidity':>12} | resolves     | question")
    print("-" * 100)
    for m in markets:
        print(
            f"{m.yes_price:5.2f} | "
            f"${m.volume_24h:11,.0f} | "
            f"${m.liquidity:11,.0f} | "
            f"{m.end_date[:10]:>12} | "
            f"{m.question[:60]}"
        )


if __name__ == "__main__":
    main()
