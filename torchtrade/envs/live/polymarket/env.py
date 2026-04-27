"""PolymarketBetEnv — rolling one-shot bets on a Polymarket market series.

Pattern B (contextual-bandit shape): each step is an independent bet on a
fresh, short-cadence binary market — bet on direction, wait for resolution,
collect the realized payoff, then move to the next market in the series.

Concrete example (5-minute Bitcoin up/down):

    config = PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")
    env = PolymarketBetEnv(config, dry_run=True)

    td = env.reset()
    while True:
        action = policy(td)              # 0 = Down, 1 = Up
        td = env.step(td.set("action", action))["next"]
        if td["done"]: break
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Bounded, Categorical
from torchrl.data.tensor_specs import Composite
from torchrl.envs import EnvBase

from torchtrade.envs.live.polymarket.market_scanner import (
    GAMMA_API_BASE,
    MarketScanner,
    MarketScannerConfig,
    PolymarketMarket,
)

logger = logging.getLogger(__name__)


@dataclass
class PolymarketBetEnvConfig:
    """Configuration for :class:`PolymarketBetEnv`.

    Attributes:
        market_slug_prefix: Stable identifier for the market series, discovered
            via ``scan_markets.py`` (e.g. ``"btc-updown-5m-"`` for 5-minute BTC
            up/down markets).
        bet_fraction: Fraction of current cash to stake per bet (default 1%).
        max_steps: Maximum bets per episode.
        initial_cash: Starting USDC balance used for dry-run accounting; live
            mode reads the real wallet via the trader.
        done_on_bankruptcy: If True, terminate the episode when cash drops below
            ``bankrupt_threshold * initial_cash``.
        bankrupt_threshold: Bankruptcy cutoff as a fraction of initial cash.
        dry_run: If True, skip CLOB order submission but still wait for
            resolution and compute the would-have-been payoff.
        resolution_grace_seconds: Extra wait after the market's endDate before
            polling for the resolved outcome (Polymarket settles on-chain).
    """

    market_slug_prefix: str = ""
    bet_fraction: float = 0.01
    max_steps: int = 50
    initial_cash: float = 1_000.0
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1
    dry_run: bool = False
    resolution_grace_seconds: float = 30.0


class PolymarketBetEnv(EnvBase):
    """One-shot betting environment for short-cadence Polymarket up/down markets.

    Each step is an independent bet:

    1. The current market_state is observed
       (``[yes_price, spread, volume_24h, liquidity]``).
    2. The agent picks a side (0 = Down, 1 = Up).
    3. The trader submits a market order (skipped in ``dry_run``).
    4. The env sleeps until the market's endDate plus a small grace period.
    5. The resolved outcome is fetched from Gamma; the realized payoff is
       computed and returned as the step's reward.
    6. The scanner picks the next active market matching ``market_slug_prefix``
       and its market_state becomes the next observation.

    Episode ends when the scanner finds no next market (terminated), the wallet
    drops below the bankruptcy threshold (terminated), or ``max_steps`` is hit
    (truncated). The observation deliberately omits any ``account_state`` — by
    the time the next decision is made, the previous bet has already resolved
    and there is no carried position to encode.
    """

    batch_locked = False

    def __init__(
        self,
        config: PolymarketBetEnvConfig,
        private_key: str = "",
        scanner: Optional[MarketScanner] = None,
        trader=None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device, batch_size=())

        if not config.market_slug_prefix:
            raise ValueError(
                "PolymarketBetEnvConfig.market_slug_prefix is required. "
                "Discover it via examples/broker/polymarket/scan_markets.py "
                "(e.g. --slug-prefix btc-updown-5m-)."
            )
        self.config = config

        self.scanner = scanner or MarketScanner(MarketScannerConfig())
        if trader is not None:
            self.trader = trader
        else:
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )
            self.trader = PolymarketOrderExecutor(
                private_key=private_key, dry_run=config.dry_run
            )

        # Specs — Binary for bool flags (per TorchRL spec semantics) and
        # reward inside a Composite so RewardSum-style transforms work.
        self.observation_spec = Composite(
            market_state=Bounded(
                low=0.0, high=float("inf"), shape=(4,), dtype=torch.float32
            ),
            shape=(),
        )
        self.action_spec = Categorical(2)
        self.reward_spec = Composite(
            reward=Bounded(
                low=-float("inf"), high=float("inf"), shape=(1,), dtype=torch.float32
            ),
            shape=(),
        )
        # Declare only terminated/truncated; TorchRL derives `done` automatically
        # via _complete_done so the two cannot drift apart.
        self.full_done_spec = Composite(
            terminated=Binary(n=1, shape=(1,), dtype=torch.bool),
            truncated=Binary(n=1, shape=(1,), dtype=torch.bool),
            shape=(),
        )

        self.cash = config.initial_cash
        self._step_count = 0
        self._current_market: Optional[PolymarketMarket] = None

    # ------------------------------------------------------------------ #
    #  TorchRL entry points                                               #
    # ------------------------------------------------------------------ #

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self.cash = self.config.initial_cash
        self._step_count = 0

        market = self._fetch_next_market()
        if market is None:
            raise RuntimeError(
                f"No active markets found for slug_prefix='{self.config.market_slug_prefix}'. "
                "Run scan_markets.py to verify the prefix and market availability."
            )
        self._current_market = market

        return TensorDict(
            {
                "market_state": self._market_state(market),
                "terminated": torch.zeros(1, dtype=torch.bool),
                "truncated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        market = self._current_market
        if market is None:
            raise RuntimeError("step() called before reset()")

        action_idx = int(tensordict.get("action").item())

        fill_price = market.yes_price if action_idx == 1 else market.no_price
        stake = self.cash * self.config.bet_fraction
        token_id = market.yes_token_id if action_idx == 1 else market.no_token_id

        if stake > 0 and not self.config.dry_run:
            result = self.trader.buy(token_id=token_id, amount_usdc=stake)
            if not result.get("success"):
                # Order failed (FOK rejection, insufficient USDC, network glitch).
                # Do NOT book a payoff against an order we never filled — set the
                # effective stake to zero so _compute_payoff returns 0.0.
                logger.warning(
                    "Order failed for %s: %s — recording as no-bet",
                    market.slug,
                    result.get("error", "unknown"),
                )
                stake = 0.0

        self._wait_for_resolution(market.end_date)
        outcome = self._fetch_resolved_outcome(market.condition_id)

        if outcome is None:
            logger.warning("Market %s not yet resolved after wait; reward=0", market.slug)
            payoff = 0.0
        else:
            payoff = self._compute_payoff(action_idx, fill_price, outcome, stake)

        self.cash += payoff
        self._step_count += 1

        next_market = self._fetch_next_market()
        terminated = bool(next_market is None) or self._is_bankrupt()
        truncated = self._step_count >= self.config.max_steps and not terminated

        if next_market is not None:
            self._current_market = next_market
            obs = self._market_state(next_market)
        else:
            # Terminal: agent should not bootstrap from a stale observation.
            obs = torch.zeros(
                self.observation_spec["market_state"].shape, dtype=torch.float32
            )

        return TensorDict(
            {
                "market_state": obs,
                "reward": torch.tensor([payoff], dtype=torch.float32),
                "terminated": torch.tensor([terminated], dtype=torch.bool),
                "truncated": torch.tensor([truncated], dtype=torch.bool),
            },
            batch_size=self.batch_size,
        )

    def _set_seed(self, seed: Optional[int]):
        # Env outcomes come from the live exchange; nothing to seed locally.
        return None

    # ------------------------------------------------------------------ #
    #  Helpers (overridable in tests)                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _market_state(market: PolymarketMarket) -> torch.Tensor:
        """4-element market state: [yes_price, spread, volume_24h, liquidity]."""
        return torch.tensor(
            [market.yes_price, market.spread, market.volume_24h, market.liquidity],
            dtype=torch.float32,
        )

    def _fetch_next_market(self) -> Optional[PolymarketMarket]:
        return self.scanner.next_active_market(self.config.market_slug_prefix)

    def _wait_for_resolution(self, end_date_iso: str) -> None:
        """Sleep until ``end_date_iso + grace``. Tests should override this."""
        if not end_date_iso:
            return
        try:
            end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        except ValueError:
            return
        target = end_dt + timedelta(seconds=self.config.resolution_grace_seconds)
        sleep_seconds = (target - datetime.now(timezone.utc)).total_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _fetch_resolved_outcome(self, condition_id: str) -> Optional[int]:
        """Return 1 (Up won), 0 (Down won), or None if still unresolved."""
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={"condition_id": condition_id},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            market = data[0] if isinstance(data, list) and data else data
            up, down = (float(p) for p in json.loads(market["outcomePrices"])[:2])
        except (requests.RequestException, ValueError, TypeError, KeyError, IndexError):
            return None
        if up >= 0.99 and down <= 0.01:
            return 1
        if down >= 0.99 and up <= 0.01:
            return 0
        return None

    @staticmethod
    def _compute_payoff(
        action: int, fill_price: float, outcome: int, stake: float
    ) -> float:
        """Realized USDC payoff. Win pays ``stake * (1 - fill) / fill``; loss returns ``-stake``."""
        if stake <= 0:
            return 0.0
        if action == outcome:
            if fill_price <= 0:
                return 0.0
            return stake * (1.0 - fill_price) / fill_price
        return -stake

    def _is_bankrupt(self) -> bool:
        initial = self.config.initial_cash
        return (
            self.config.done_on_bankruptcy
            and initial > 0
            and self.cash < self.config.bankrupt_threshold * initial
        )

    def close(self):
        self.trader.cancel_all()
