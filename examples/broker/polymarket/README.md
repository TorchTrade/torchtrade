# Polymarket Broker Examples

Examples demonstrating `PolyTimeBarEnv` — the live trading environment for
[Polymarket](https://polymarket.com/) prediction markets.

All examples default to `dry_run=True`, so you can run them without a funded
Polygon wallet or a real `py-clob-client` install.

## Examples

### Discover markets

List the most-liquid active markets matching a filter (volume, liquidity,
time-to-resolution, optional keyword). No environment, no orders.

```bash
# Top 10 by 24h volume
python examples/broker/polymarket/scan_markets.py

# Filter by keyword (case-insensitive substring on question or slug)
python examples/broker/polymarket/scan_markets.py --keyword bitcoin

# Tweak filters
python examples/broker/polymarket/scan_markets.py --keyword "world cup" --min-volume 50000 --max 5
```

Example output:

```
  YES |      24h vol |    liquidity | resolves     | question
----------------------------------------------------------------------------------------------------
 0.01 | $  1,303,675 | $  2,565,119 |   2026-07-20 | Will USA win the 2026 FIFA World Cup?
 0.00 | $    858,344 | $    999,236 |   2026-07-01 | Will the Portland Trail Blazers win the 2026 NBA Finals?
 0.00 | $    751,378 | $  5,583,069 |   2026-07-20 | Will Iraq win the 2026 FIFA World Cup?
 0.16 | $    652,216 | $    362,115 |   2026-07-01 | Will the San Antonio Spurs win the 2026 NBA Finals?
 0.14 | $    513,394 | $    111,446 |   2026-07-01 | Will the Boston Celtics win the 2026 NBA Finals?
 0.04 | $    484,678 | $    378,936 |   2026-07-01 | Will the Los Angeles Lakers win the 2026 NBA Finals?
 0.01 | $    451,391 | $    731,710 |   2026-07-01 | Will the Minnesota Timberwolves win the 2026 NBA Finals?
 0.03 | $    450,139 | $    508,460 |   2026-07-01 | Will the New York Knicks win the 2026 NBA Finals?
 0.01 | $    425,376 | $    531,424 |   2026-07-01 | Will the Atlanta Hawks win the 2026 NBA Finals?
 0.09 | $    409,781 | $  1,402,387 |   2026-07-20 | Will Brazil win the 2026 FIFA World Cup?
```

The output reflects the live Gamma API at run time — your numbers and rows will differ.

#### Column reference

| Column      | Meaning |
|-------------|---------|
| `YES`       | Current market price of the YES outcome token in USDC, in [0, 1]. Read it as the market's implied probability of YES resolving true (e.g. `0.16` ≈ 16 % implied probability). |
| `24h vol`   | Total USDC traded against this market in the trailing 24 hours. Higher numbers mean tighter spreads and easier fills. |
| `liquidity` | USDC currently resting in the order book (sum of bid + ask depth). Higher numbers mean less slippage on entry/exit. |
| `resolves`  | Date the market is scheduled to resolve, in `YYYY-MM-DD` (UTC). Markets close to resolution have less time for moves but tighter pricing. |
| `question`  | The human-readable question the market resolves on. Truncated to 60 characters in the table. |

Rows are sorted by `24h vol` descending and capped at `--max` (default 10).

#### CLI flags

| Flag             | Default     | Description |
|------------------|-------------|-------------|
| `--keyword`      | *(none)*    | Case-insensitive substring match against the market `question` **or** `slug`. Quote multi-word terms (`--keyword "world cup"`). |
| `--min-volume`   | `10000`     | Minimum 24-hour volume in USDC. Use this to skip illiquid markets. |
| `--min-liquidity`| `5000`      | Minimum resting order-book liquidity in USDC. |
| `--max`          | `10`        | Maximum number of markets to print. |

The same fields are available on `MarketScannerConfig` if you want to use the scanner programmatically:

```python
from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig

scanner = MarketScanner(MarketScannerConfig(
    keyword="bitcoin",
    min_volume_24h=50_000.0,
    min_liquidity=10_000.0,
    max_markets=5,
))
for market in scanner.scan():
    print(market.slug, market.yes_price)
```

### End-to-end dry run

Pick the top market via the scanner, build `PolyTimeBarEnv` in dry-run mode,
and run a 3-step random rollout.

```bash
python examples/broker/polymarket/run_dry_run.py
```

### With a supplementary observer

Augment the env's `market_state` with an external feature window — useful when
a prediction market is correlated with another asset (crypto OHLCV, news embeddings,
etc.). The example uses a tiny stub observer; swap it for a production source.

```bash
python examples/broker/polymarket/run_with_supplementary.py
```

## Running for real

To trade real funds, set:

- `POLYGON_PRIVATE_KEY` in `.env` — Polygon wallet with USDC.e
- Install the optional CLOB client: `pip install py-clob-client`
- Set `dry_run=False` in `PolyTimeBarEnvConfig`

Always start with `dry_run=True` and verify the action loop and accounting
before flipping the switch.

## See Also

- [Online Environments docs](../../../docs/environments/online.md#polymarket-environment)
- Source: `torchtrade/envs/live/polymarket/`
