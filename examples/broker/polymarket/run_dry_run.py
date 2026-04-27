"""End-to-end Polymarket dry-run example.

Steps:
    1. Discover an active prediction market via the Gamma API.
    2. Construct PolyTimeBarEnv in dry-run mode (no real orders).
    3. Run a short random-policy rollout and print the trajectory.

Run with:
    python examples/online_rl/polymarket/run_dry_run.py
"""

from __future__ import annotations

import os

import torch
from tensordict import TensorDict

from torchtrade.envs.live.polymarket import (
    MarketScanner,
    MarketScannerConfig,
    PolyTimeBarEnv,
    PolyTimeBarEnvConfig,
)


def pick_market(min_volume: float = 10_000.0, min_liquidity: float = 5_000.0):
    """Return the most-liquid active market matching the filters, or None."""
    scanner = MarketScanner(
        MarketScannerConfig(
            min_volume_24h=min_volume,
            min_liquidity=min_liquidity,
            max_markets=5,
        )
    )
    markets = scanner.scan()
    if not markets:
        return None
    return markets[0]  # already sorted by 24h volume desc


def main():
    market = pick_market()
    if market is None:
        print(
            "No active markets matched the scanner filters. "
            "Lower the volume/liquidity thresholds and retry."
        )
        return

    print(f"Selected market: {market.question}")
    print(f"  slug:        {market.slug}")
    print(f"  yes_price:   {market.yes_price:.3f}")
    print(f"  volume_24h:  ${market.volume_24h:,.0f}")
    print(f"  liquidity:   ${market.liquidity:,.0f}")
    print(f"  resolves:    {market.end_date}")

    config = PolyTimeBarEnvConfig(
        market_slug=market.slug,
        execute_on="1Hour",
        action_levels=[-1.0, 0.0, 1.0],  # buy NO / flat / buy YES
        max_steps=3,
        dry_run=True,                    # no real orders
        close_position_on_init=False,    # nothing to close in dry run
    )
    # Dry-run with no funded wallet → portfolio value is 0, so the default
    # log-return reward would raise. Replace with a simple no-op for demo
    # purposes; production code should pick a reward suited to its market.
    env = PolyTimeBarEnv(
        config=config,
        private_key=os.getenv("POLYGON_PRIVATE_KEY", ""),
        reward_function=lambda history: 0.0,
    )
    # Skip the bar wait so this script returns quickly.
    env._wait_for_next_timestamp = lambda: None

    td = env.reset()
    print(f"\nInitial market_state:  {td['market_state'].tolist()}")
    print(f"Initial account_state: {td['account_state'].tolist()}")

    for step in range(config.max_steps):
        action_idx = torch.randint(0, env.action_spec.n, ())
        td_in = td.set("action", action_idx)
        td = env._step(td_in)
        print(
            f"\nstep={step + 1}  action_idx={action_idx.item()}  "
            f"reward={td['reward'].item():+.5f}  "
            f"done={td['done'].item()}  "
            f"position_dir={td['account_state'][1].item():+.0f}"
        )
        if td["done"].item():
            break

    env.close()


if __name__ == "__main__":
    main()
