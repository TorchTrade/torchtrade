"""List active Polymarket prediction markets that pass volume/liquidity filters.

This script does not place any orders — it only reads from the public Gamma API.

Run with:
    python examples/broker/polymarket/scan_markets.py
"""

from __future__ import annotations

from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig


def main():
    scanner = MarketScanner(
        MarketScannerConfig(
            min_volume_24h=10_000.0,
            min_liquidity=5_000.0,
            min_time_to_resolution_hours=24,
            max_markets=10,
        )
    )
    markets = scanner.scan()
    if not markets:
        print("No markets matched the filters.")
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
