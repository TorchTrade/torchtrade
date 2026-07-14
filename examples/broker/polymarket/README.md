# Polymarket Broker Examples

Examples demonstrating `PolymarketBetEnv`, a one-shot betting environment for
short-cadence binary markets on [Polymarket](https://polymarket.com/).

Polymarket runs continuous **5-minute, 15-minute, 1-hour, and 4-hour** crypto
"up/down" markets (BTC, ETH, SOL) plus daily markets. Each market is a binary
bet: did the asset go up or down over the bar? The env rolls through the
series, bet → wait for resolution → collect payoff → next market, without
holding any position across steps.

All examples default to `dry_run=True`, so you can run them without a funded
Polygon wallet or `py-clob-client` installed.

## End-to-end walkthrough

A complete tour in five steps: **discover → identify → configure → run**.

### 1. Discover what's available

Use `scan_markets.py` to query the live Gamma API. With no flags it returns
the highest-volume open markets. Add `--slug-prefix` once you know what you
want, same primitive the env uses, so what you see here is what the env will
trade:

```bash
# Browse high-volume markets
python examples/broker/polymarket/scan_markets.py

# Look at upcoming short-cadence crypto markets
python examples/broker/polymarket/scan_markets.py --max-resolution-minutes 30 --max 8

# Lock to the BTC 5-minute series
python examples/broker/polymarket/scan_markets.py --slug-prefix btc-updown-5m- --max 5
```

Sample output for the BTC 5m query:

```
  YES |     24h vol |   liquidity |       resolves | slug
--------------------------------------------------------------------------------------
 0.51 | $     1,690 | $    18,110 |       in    3m | btc-updown-5m-1777284300
 0.51 | $        72 | $    21,010 |       in    8m | btc-updown-5m-1777284600
 0.51 | $        15 | $    18,054 |       in   53m | btc-updown-5m-1777287300
 0.51 | $        15 | $    14,635 |       in  1.7h | btc-updown-5m-1777290300
 0.51 | $        10 | $    14,670 |       in  1.4h | btc-updown-5m-1777289100
```

#### Column reference

| Column | Meaning |
|--------|---------|
| `YES`       | Implied probability of "Up" (the YES outcome), in `[0, 1]`. `0.51` ≈ 51 % chance. |
| `24h vol`   | USDC traded against this market in the trailing 24 h. Short-cadence markets typically show ~$0 until minutes before resolution. |
| `liquidity` | USDC currently resting on the order book. Higher = lower slippage. |
| `resolves`  | Time until resolution, formatted as `Xm` / `Xh` / date. |
| `slug`      | The market's stable identifier on Polymarket. The **prefix** (everything before the trailing timestamp / strike) is what the env locks onto. |

#### CLI flags

| Flag                       | Default | Description |
|----------------------------|---------|-------------|
| `--slug-prefix`            | *(none)* | Case-sensitive prefix match on the market slug, the env-side primitive. Use it once you know what you want. |
| `--keyword`                | *(none)* | One or more case-insensitive substrings; ANY-match against question/slug. Useful for fuzzy discovery. |
| `--min-volume`             | `0`      | Minimum 24 h volume (USD). |
| `--min-liquidity`          | `0`      | Minimum order-book liquidity (USD). |
| `--min-resolution-hours`   | `0`      | Lower bound on time-to-resolution. |
| `--max-resolution-minutes` | *(none)* | Upper bound on time-to-resolution. Set to e.g. `30` to surface short-cadence markets that volume-sorted browsing would miss. |
| `--max`                    | `20`     | Cap on rows printed. |

### 2. Identify the slug prefix

The `slug` column above shows individual markets like `btc-updown-5m-1777284300`.
The trailing number is a per-market epoch; the **prefix** `btc-updown-5m-` is
the stable series identifier. Currently active short-cadence series include:

| Series | Cadence | Slug prefix |
|--------|---------|-------------|
| BTC up/down (5 min)   | 5 min  | `btc-updown-5m-` |
| BTC up/down (15 min)  | 15 min | `btc-updown-15m-` |
| BTC up/down (1 hour)  | 1 h    | `bitcoin-up-or-down-` |
| BTC up/down (4 hour)  | 4 h    | `btc-updown-4h-` |
| ETH up/down (5 min)   | 5 min  | `eth-updown-5m-` |
| ETH up/down (15 min)  | 15 min | `eth-updown-15m-` |
| SOL up/down (5 min)   | 5 min  | `sol-updown-5m-` |
| BTC daily up/down     | 1 day  | `bitcoin-up-or-down-on-` |

Polymarket's slug naming isn't perfectly consistent (5m / 15m / 4h use `btc`
abbreviations; 1h and daily use full `bitcoin`), so re-running `scan_markets.py`
once before configuring the env is the safe move.

### 3. Configure the env

Plug the prefix into `PolymarketBetEnvConfig`:

```python
from torchtrade.envs.live.polymarket import PolymarketBetEnv, PolymarketBetEnvConfig

config = PolymarketBetEnvConfig(
    market_slug_prefix="btc-updown-5m-",   # the only required field
    bet_fraction=0.01,                     # stake 1 % of cash per bet
    max_steps=10,                          # 10 bets per episode
    initial_cash=1_000.0,                  # simulated paper balance
    dry_run=True,                          # the only supported mode (the default)
)
env = PolymarketBetEnv(config)
```

> **Paper trading only.** `dry_run=False` raises `NotImplementedError`. Two reasons, either
> fatal on its own:
> 1. `py-clob-client`, which the order executor is built on, was **archived** in May 2026
>    ("no longer functional"). Polymarket's CLOB V2 uses new contracts and replaced USDC.e
>    collateral with pUSD, so no order this env submits can reach production.
> 2. This env **buys and holds every bet through resolution**, and Polymarket does *not*
>    release collateral on resolution — a winning share stays an ERC-1155 token until it is
>    explicitly **redeemed** on-chain, which no Polymarket client exposes. A live bot that
>    never redeems would watch its balance drain to zero *while winning*.
>
> Reviving live trading needs the CLOB V2 port **and** a redemption workflow. Everything
> else about the env is real: real market data, real resolution, real payoffs — booked
> against a simulated balance.


Swapping to ETH 15-minute is a one-line change:

```python
config = PolymarketBetEnvConfig(market_slug_prefix="eth-updown-15m-")
```

### 4. Inspect the observation

`reset()` returns a `TensorDict` with a single key, `market_state` of the
freshly-picked next market:

```python
td = env.reset()
td["market_state"]
# tensor([yes_price, spread, vol_24h, liquidity])  shape (4,)
```

There is **no** `account_state`. By the time the next `step()` runs, the
previous bet has already resolved, there is no carried position to encode.
Cumulative P&L is captured directly in the per-step rewards.

### 5. Step the env

Each `step()` does five things:

```
1. Submit the bet on the current market (skipped in dry_run)
2. Sleep until the market's endDate + grace
3. Poll the CLOB midpoint for each outcome token; resolved when YES mid >= 0.99
   and NO mid <= 0.01 (Up won) or vice versa (Down won)
4. Compute realized payoff:  win → stake * (1 - fill) / fill, loss → -stake
5. Pick the next active market matching market_slug_prefix and return its market_state
```

```python
import torch

td = env.reset()
for _ in range(config.max_steps):
    action = torch.randint(0, env.action_spec.n, ())  # 0 = Down, 1 = Up
    td = env.step(td.set("action", action))["next"]
    if td["done"]: break
env.close()
```

The runnable equivalent, with a random policy and progress prints, is
[`run_dry_run.py`](run_dry_run.py).

## Examples

### `scan_markets.py`: discover

See section 1 above.

### `run_dry_run.py`: bet end-to-end

```bash
python examples/broker/polymarket/run_dry_run.py
python examples/broker/polymarket/run_dry_run.py --slug-prefix btc-updown-15m- --max-steps 4
```

Note: the script blocks until each market resolves, so default 5-minute / 2-step
takes ~10–15 minutes wall-clock. Use `--slug-prefix btc-updown-5m-` and
`--max-steps 1` for the quickest end-to-end test.

## Action-space convention

`Categorical(2)` over `{0: Down, 1: Up}`, indexing into the market's `["Up",
"Down"]` outcomes. An LLM actor can prompt over the question text and emit a
discrete index; an RL policy gets logits over two classes.

## Running for real

**You can't, and you shouldn't try.** `dry_run=False` raises `NotImplementedError`. The
previous instructions here (set `POLYGON_PRIVATE_KEY`, `pip install py-clob-client`, flip
`dry_run`) were all wrong, and following them would have lost money quietly. Two blockers:

1. **The client is dead.** `py-clob-client` was archived in May 2026 — *"no longer functional
   and should not be used for new or existing integrations."* Polymarket's CLOB V2 uses new
   contracts and replaced USDC.e collateral with pUSD. No order this env submits can reach
   production.
2. **There is no redemption.** This env buys and holds every bet through resolution, and
   Polymarket does **not** release collateral when a market resolves — a winning share is
   worth $1 but stays an ERC-1155 token until an explicit on-chain `redeemPositions` call,
   which no Polymarket client exposes (it has to go through their Relayer, because the
   proxy wallet owns the position, not your EOA). A bot that never redeems watches its
   spendable balance **drain to zero while it is winning.**

Blocker 2 is why this can't be fixed by "just reading the real wallet": while winnings sit
unredeemed, the collateral balance isn't the account's worth, so a wallet-backed balance
would make a *winning* agent declare itself bankrupt.

Reviving live trading needs the CLOB V2 port **and** a redemption workflow.

## See Also

- [Online Environments docs](../../../docs/environments/online.md#polymarket-environment)
- Source: `torchtrade/envs/live/polymarket/`
