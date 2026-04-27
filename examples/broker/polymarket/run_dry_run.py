"""Dry-run a few rolling Polymarket bets on a short-cadence series.

Picks the next active market matching ``market_slug_prefix`` (e.g.
``btc-updown-5m-``), bets a random direction, *waits for resolution*, collects
the realized payoff, and rolls to the next market. ``dry_run=True`` skips real
CLOB orders so no funded wallet is required.

Run with:
    python examples/broker/polymarket/run_dry_run.py
    python examples/broker/polymarket/run_dry_run.py --slug-prefix btc-updown-15m- --max-steps 4

By default ``--max-steps 2`` and 5-minute markets, so the script blocks for
roughly 10–15 minutes (two bar resolutions plus a 30 s grace each).
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from torchtrade.envs.live.polymarket import (
    PolymarketBetEnv,
    PolymarketBetEnvConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slug-prefix", default="btc-updown-5m-")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--bet-fraction", type=float, default=0.01)
    parser.add_argument("--initial-cash", type=float, default=1_000.0)
    args = parser.parse_args()

    # Surface env-level progress logs (waiting for endDate, polling for
    # resolution, etc.) so the long-blocking phases don't look hung.
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("torchtrade.envs.live.polymarket").setLevel(logging.INFO)

    config = PolymarketBetEnvConfig(
        market_slug_prefix=args.slug_prefix,
        max_steps=args.max_steps,
        bet_fraction=args.bet_fraction,
        initial_cash=args.initial_cash,
        dry_run=True,
    )
    env = PolymarketBetEnv(config, private_key=os.getenv("POLYGON_PRIVATE_KEY", ""))

    td = env.reset()
    print(f"slug_prefix:        {config.market_slug_prefix}")
    print(f"initial cash:       ${env.cash:,.2f}")
    print(
        f"first market_state: yes_price={td['market_state'][0].item():.3f}  "
        f"liq=${td['market_state'][3].item():,.0f}"
    )

    # Loop until the env says we're done (max_steps truncation or bankruptcy);
    # no external counter, the env owns termination.
    step = 0
    while not bool(td.get("done", torch.zeros(1, dtype=torch.bool)).item()):
        step += 1
        action = torch.randint(0, env.action_spec.n, ())
        side = "UP" if action.item() == 1 else "DOWN"
        print(f"\nstep {step}: betting {side} (waiting for resolution)...")
        td = env.step(td.set("action", action))["next"]
        print(
            f"  resolved → reward={td['reward'].item():+.4f}  "
            f"cash=${env.cash:,.2f}  done={bool(td['done'].item())}"
        )

    env.close()
    print(f"\nfinal cash: ${env.cash:,.2f}  (started ${args.initial_cash:,.2f})")


if __name__ == "__main__":
    main()
