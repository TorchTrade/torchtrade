"""Demonstrate the supplementary-observer pattern.

PolyTimeBarEnv accepts any object exposing ``get_observation_spec()`` and
``get_observations()`` as a supplementary data source. This is useful when a
prediction market is correlated with another asset (e.g., a "BTC > $100k" market
correlates with BTC OHLCV).

To keep this example self-contained, we build a tiny ``DemoOHLCVObserver`` that
emits random data with a fixed shape. Replace it with e.g. a ``BinanceObservationClass``
in production — but note that observers from other live envs may not implement
``get_observation_spec`` directly; you'll need a small adapter that returns the
TorchRL ``Bounded`` spec for whatever feature window you're feeding in.

Run with:
    python examples/broker/polymarket/run_with_supplementary.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded

from torchtrade.envs.live.polymarket import (
    MarketScanner,
    MarketScannerConfig,
    PolyTimeBarEnv,
    PolyTimeBarEnvConfig,
)


class DemoOHLCVObserver:
    """Emits a random ``(window, 4)`` OHLCV-shaped feature window each step.

    Replace with a real broker observer (e.g., BinanceObservationClass wrapped
    in an adapter) to feed live OHLCV alongside Polymarket market state.
    """

    def __init__(self, window: int = 12, key: str = "btc_ohlcv"):
        self.window = window
        self.key = key

    def get_observation_spec(self):
        return {
            self.key: Bounded(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.window, 4),
                dtype=torch.float32,
            )
        }

    def get_observations(self):
        return {self.key: np.random.randn(self.window, 4).astype(np.float32)}


def main():
    scanner = MarketScanner(MarketScannerConfig(max_markets=1))
    markets = scanner.scan()
    if not markets:
        print("No active markets — try lowering the scanner thresholds.")
        return
    market = markets[0]
    print(f"Selected market: {market.question}\n")

    config = PolyTimeBarEnvConfig(
        market_slug=market.slug,
        execute_on="1Hour",
        action_levels=[-1.0, 0.0, 1.0],
        max_steps=2,
        dry_run=True,
        close_position_on_init=False,
    )
    env = PolyTimeBarEnv(
        config=config,
        private_key=os.getenv("POLYGON_PRIVATE_KEY", ""),
        supplementary_observers=[DemoOHLCVObserver()],
        reward_function=lambda history: 0.0,
    )
    env._wait_for_next_timestamp = lambda: None

    td = env.reset()
    print(f"observation keys: {list(td.keys())}")
    print(f"  market_state shape:  {td['market_state'].shape}")
    print(f"  account_state shape: {td['account_state'].shape}")
    print(f"  btc_ohlcv shape:     {td['btc_ohlcv'].shape}")

    for step in range(config.max_steps):
        td = env._step(td.set("action", torch.tensor(2)))  # buy YES each step
        print(
            f"\nstep={step + 1}  "
            f"position_dir={td['account_state'][1].item():+.0f}  "
            f"done={td['done'].item()}"
        )
    env.close()


if __name__ == "__main__":
    main()
