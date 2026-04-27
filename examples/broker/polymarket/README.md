# Polymarket Broker Examples

Examples demonstrating `PolyTimeBarEnv` — the live trading environment for
[Polymarket](https://polymarket.com/) prediction markets.

All examples default to `dry_run=True`, so you can run them without a funded
Polygon wallet or a real `py-clob-client` install.

## End-to-end walkthrough

A complete tour of the env in ~30 lines: pick a market, configure, step.

### 1. Find a market

Use `MarketScanner` to query the public Gamma API and return the most-liquid
active markets matching your filters:

```python
from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig

scanner = MarketScanner(MarketScannerConfig(
    keyword=["btc", "bitcoin"],   # any-match across question + slug
    min_volume_24h=50_000.0,
    min_liquidity=10_000.0,
    max_markets=1,
))
market = scanner.scan()[0]
print(market.slug, market.question, market.yes_price)
# will-bitcoin-hit-150k-by-june-30-2026  Will Bitcoin hit $150k by June 30, 2026?  0.01
```

Each `PolymarketMarket` carries the fields you need to construct an env —
`slug`, `condition_id`, `yes_token_id`, `no_token_id`, plus pricing/liquidity
metadata.

### 2. Configure the env

`PolyTimeBarEnvConfig` is a dataclass — only `market_slug` (or `condition_id` /
`yes_token_id`) is required. The most important knob is `action_levels`: a list
of portfolio fractions in `[-1, 1]` that the agent's discrete action indexes
into. Negative = NO, positive = YES, zero = flat.

```python
from torchtrade.envs.live.polymarket import PolyTimeBarEnv, PolyTimeBarEnvConfig

config = PolyTimeBarEnvConfig(
    market_slug=market.slug,
    execute_on="1Hour",                    # one RL step per hour bar
    action_levels=[-1.0, 0.0, 1.0],        # buy NO / flat / buy YES
    max_steps=24,                          # 1-day episode
    dry_run=True,                          # no real orders, no wallet needed
    close_position_on_init=False,
)
env = PolyTimeBarEnv(
    config=config,
    private_key="",                        # only needed when dry_run=False
    reward_function=lambda h: 0.0,         # replace with log_return_reward etc.
)
```

### 3. Inspect the observation

`reset()` returns a TensorDict with two universal keys:

```python
td = env.reset()
td["market_state"]   # tensor([yes_price, spread, vol_24h, liquidity, time_to_resolution])  shape (5,)
td["account_state"]  # tensor([exposure, direction, unrealized_pnl, hold_time, leverage=1, liq_dist=1])  shape (6,)
```

`account_state` follows TorchTrade's universal layout, so a policy trained on
offline crypto envs can be plugged in here without rewiring. `leverage` and
`distance_to_liquidation` are constant 1.0 on Polymarket (no leverage).

### 4. Step the env

Each `step()` does seven things in order, all in one bar:

```
1. Resolve action_idx → desired fraction (e.g. 2 → +1.0 = "100% YES")
2. Read live YES price from CLOB
3. Diff target vs current position; place delta order via the trader
4. Sleep until the next bar boundary (skipped in tests / dry-run-fast)
5. Bump the hold counter
6. Build the next observation TensorDict
7. Compute reward; check market-closed / bankruptcy / max_steps
```

Run it with any TorchRL-compatible loop:

```python
import torch
from tensordict import TensorDict

td = env.reset()
for step in range(config.max_steps):
    action_idx = torch.randint(0, env.action_spec.n, ())   # replace with policy(td)
    td = env._step(td.set("action", action_idx))
    if td["done"].item():
        break
env.close()
```

### 5. What the trader actually does

In `dry_run=True` the order executor short-circuits — every `buy/sell` returns
`{"success": True, "dry_run": True}` without touching the network. Flip
`dry_run=False`, set `POLYGON_PRIVATE_KEY`, and `pip install py-clob-client`,
and the same code submits real fill-or-kill market orders to
`clob.polymarket.com`. The agent never knows the difference.

The position state machine inside `_execute_trade_if_needed` always trades the
**delta** between target and current exposure:

| Current → Target              | Action                            |
|-------------------------------|-----------------------------------|
| flat → +0.5 YES               | buy YES with 50% of portfolio     |
| +0.5 YES → +1.0 YES           | top up: buy more YES              |
| +1.0 YES → -1.0 NO            | close YES, then buy NO            |
| anything → 0.0                | close current position            |
| same target as current        | no-op (delta < $1 floor)          |

### Termination

`step()` sets one of these flags:

- `terminated=True` — underlying market resolved (`observer.is_market_closed()`)
  or wallet dropped below `bankrupt_threshold * initial_balance`.
- `truncated=True` — hit `config.max_steps`.

`done = terminated or truncated`, following the standard TorchRL convention.

The runnable equivalent of the snippets above is
[`run_dry_run.py`](run_dry_run.py) — exactly the same flow, with `argparse` and
print statements stripped out.

## Examples

### Discover markets

List the most-liquid active markets matching a filter (volume, liquidity,
time-to-resolution, optional keyword). No environment, no orders.

```bash
# Top 10 by 24h volume
python examples/broker/polymarket/scan_markets.py

# Filter by a single keyword (case-insensitive substring on question or slug)
python examples/broker/polymarket/scan_markets.py --keyword bitcoin

# Filter by multiple keywords — a market passes if ANY of them matches
python examples/broker/polymarket/scan_markets.py --keyword btc bitcoin crypto

# Multi-word terms must be quoted; combine with other filters as needed
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
| `--keyword`      | *(none)*    | One or more case-insensitive substrings matched against the market `question` **or** `slug`. A market passes if **any** keyword hits — `--keyword btc bitcoin crypto` matches anything containing `btc` OR `bitcoin` OR `crypto`. Quote multi-word terms: `--keyword "world cup"`. |
| `--min-volume`   | `10000`     | Minimum 24-hour volume in USDC. Use this to skip illiquid markets. |
| `--min-liquidity`| `5000`      | Minimum resting order-book liquidity in USDC. |
| `--max`          | `10`        | Maximum number of markets to print. |

The same fields are available on `MarketScannerConfig` if you want to use the scanner programmatically:

```python
from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig

scanner = MarketScanner(MarketScannerConfig(
    keyword=["btc", "bitcoin", "crypto"],  # or just a single string, e.g. "bitcoin"
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
