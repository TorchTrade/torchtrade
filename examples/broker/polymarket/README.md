# Polymarket Broker Examples

Examples demonstrating `PolyTimeBarEnv` — the live trading environment for
[Polymarket](https://polymarket.com/) prediction markets.

All examples default to `dry_run=True`, so you can run them without a funded
Polygon wallet or a real `py-clob-client` install.

## Examples

### Discover markets

List the most-liquid active markets matching a filter (volume, liquidity,
time-to-resolution). No environment, no orders.

```bash
python examples/broker/polymarket/scan_markets.py
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
