"""PolymarketBetEnv, rolling one-shot bets on a Polymarket market series.

Pattern B (contextual-bandit shape): each step is an independent bet on a
fresh, short-cadence binary market, bet on direction, wait for resolution,
collect the modelled payoff, then move to the next market in the series.

PAPER TRADING ONLY. The env runs against live Polymarket market data but never submits an
order: ``dry_run=False`` is refused at the config boundary. See ``LIVE_UNSUPPORTED`` below
for the two reasons (archived CLOB client; no redemption workflow, without which a winning
account's on-chain balance drains to zero).

Concrete example (5-minute Bitcoin up/down):

    config = PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")  # dry_run=True
    env = PolymarketBetEnv(config)

    td = env.reset()
    while True:
        action = policy(td)              # 0 = Down, 1 = Up
        td = env.step(td.set("action", action))["next"]
        if td["done"]: break
"""

from __future__ import annotations

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
    MarketScanner,
    MarketScannerConfig,
    PolymarketMarket,
)

from torchtrade.envs.utils.termination import is_bankrupt

logger = logging.getLogger(__name__)

# Polymarket's public CLOB endpoint. Used for resolution detection — see
# ``_fetch_resolved_outcome`` for why CLOB midpoint is preferred over
# Gamma's outcomePrices.
CLOB_API_BASE = "https://clob.polymarket.com"

# For whoever does the CLOB V2 port -- the part LIVE_UNSUPPORTED does not say.
#
# The redemption blocker is the deep one, and porting to the V2 SDK does NOT fix it. This env
# never sells: it buys and holds every position through resolution. Polymarket resolution does
# not move collateral, so a winning share is worth $1 but stays an ERC-1155 token until an
# explicit redeemPositions call -- which must go through their Relayer, because the proxy/Safe
# wallet owns the position, not the EOA.
#
# The corollary is the whole reason this env is refused rather than repaired: you CANNOT fix
# the simulated `cash` by reading the real wallet. While winnings sit unredeemed the collateral
# balance is not the account's worth, so wiring it in would make a WINNING agent watch its
# balance drain and declare itself bankrupt -- strictly worse than the simulation it replaced.
LIVE_UNSUPPORTED = (
    "Live Polymarket trading is not supported. `dry_run=False` is refused rather than "
    "silently mis-trading:\n"
    "  - py-clob-client is archived and non-functional (Polymarket shipped CLOB V2, with "
    "pUSD collateral replacing USDC.e); no order can reach production.\n"
    "  - This env holds every bet through resolution, and Polymarket does not auto-credit "
    "collateral on resolution. Winning shares must be redeemed on-chain via the Relayer, "
    "which no Polymarket client exposes. Without that, a winning account's balance drains "
    "to zero.\n"
    "Use dry_run=True (paper trading against live market data). Reviving live mode needs "
    "the CLOB V2 port plus a redemption workflow."
)


@dataclass(frozen=True)
class PolymarketBetEnvConfig:
    """Configuration for :class:`PolymarketBetEnv`.

    Attributes:
        market_slug_prefix: Stable identifier for the market series, discovered
            via ``scan_markets.py`` (e.g. ``"btc-updown-5m-"`` for 5-minute BTC
            up/down markets).
        bet_fraction: Fraction of current cash to stake per bet (default 1%).
        max_steps: Maximum bets per episode.
        initial_cash: Starting paper balance. This is SIMULATED bookkeeping, by design,
            and that is correct: the env is paper-only (see ``dry_run``), so there is no
            real wallet for it to disagree with. ``_reset()`` restores it each episode.
            It is NOT a claim about any real USDC/pUSD balance.
        done_on_bankruptcy: If True, terminate the episode when cash drops below
            ``bankrupt_threshold * initial_cash``.
        bankrupt_threshold: Bankruptcy cutoff as a fraction of initial cash.
        dry_run: Must be True. Paper-trades against live Polymarket market data: no order
            is submitted, but the env still waits for real resolution and books the
            would-have-been payoff. ``dry_run=False`` raises -- see LIVE_UNSUPPORTED.
        resolution_grace_seconds: Extra wait after the market's endDate before
            polling for the resolved outcome (Polymarket settles on-chain).
    """

    market_slug_prefix: str = ""
    bet_fraction: float = 0.01
    max_steps: int = 50
    initial_cash: float = 1_000.0
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1
    dry_run: bool = True
    # Initial wait after a market's endDate before the FIRST resolution poll.
    resolution_grace_seconds: float = 30.0
    # If the first poll finds the market still pre-settlement on the CLOB
    # (YES + NO midpoints not yet snapped to ~$1 / ~$0), keep polling at
    # this interval...
    resolution_poll_interval_seconds: float = 15.0
    # ... up to this total budget after endDate. Polymarket on-chain settlement
    # typically snaps the CLOB midpoints within 1-5 minutes of endDate;
    # 10 min is a safe ceiling.
    resolution_max_wait_seconds: float = 600.0

    def __post_init__(self):
        # Validated at the boundary, not guarded inside the rule. _is_bankrupt() used to carry
        # an `initial > 0` guard, which silently turned a nonsense config into "never bankrupt"
        # instead of saying so -- the silent-default pattern this repo keeps getting bitten by.
        # Refuse the config instead.
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {self.initial_cash}")

        # bet_fraction > 1 stakes more than the account holds, drives cash negative, and then
        # only the `stake <= 0` guard inside _compute_payoff stands between that and INVERTED
        # P&L (a loss paying out). Same boundary rule; 0 is allowed (a no-bet agent).
        if not 0.0 <= self.bet_fraction <= 1.0:
            raise ValueError(f"bet_fraction must be in [0, 1], got {self.bet_fraction}")

        # frozen=True above is load-bearing, not style: __post_init__ runs ONCE, at
        # construction, so on a mutable dataclass `config.dry_run = False` afterwards sails
        # straight past this check and the guard becomes a speed bump rather than a boundary.
        # (Safe for TorchRL: pickle/deepcopy restore via __dict__ without re-running
        # __post_init__, so ParallelEnv workers and env.clone() cannot spuriously raise, while
        # dataclasses.replace() DOES re-validate.)
        if not self.dry_run:
            raise NotImplementedError(LIVE_UNSUPPORTED)

    # When the scanner returns no upcoming market, retry at the env level after
    # this sleep. The scanner's internal retry budget (~48s) covers brief blips;
    # this layer covers minutes-long Gamma outages where the underlying series
    # is still running. For continuous short-cadence series (e.g. 5-min crypto),
    # "no next market" is almost always "API unreachable" rather than "series
    # ended", so a few sleep+retry rounds before terminating saves the run.
    next_market_max_attempts: int = 3
    next_market_retry_sleep_seconds: float = 30.0


class PolymarketBetEnv(EnvBase):
    """One-shot betting environment for short-cadence Polymarket up/down markets.

    Each step is an independent bet:

    1. The current market_state is observed
       (``[yes_price, spread, volume_24h, liquidity]``).
    2. The agent picks a side (0 = Down, 1 = Up).
    3. No order is submitted (paper-only; see ``LIVE_UNSUPPORTED``).
    4. The env sleeps until the market's endDate plus a small grace period.
    5. The resolved outcome is polled from the Polymarket CLOB
       (``/midpoint`` per outcome token); the realized payoff is computed
       and returned as the step's reward.
    6. The scanner picks the next active market matching ``market_slug_prefix``
       and its market_state becomes the next observation.

    Paper trading only (``dry_run=True``, enforced by the config): step 3 never submits an
    order. The market data and the resolution are real, polled live from Gamma and the CLOB.
    The PAYOFF IS MODELLED, not real: ``_compute_payoff`` fills at the quoted outcome price
    with no spread crossing and no fees, so it is optimistic relative to the fill-or-kill
    market order a live bot would actually pay. See ``LIVE_UNSUPPORTED``.

    Episode ends when the scanner finds no next market across
    ``next_market_max_attempts`` env-level retries (terminated), the simulated
    cash balance drops below the bankruptcy threshold (terminated), or ``max_steps``
    is hit (truncated). The observation deliberately omits any ``account_state``, by
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
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device or torch.device("cpu"), batch_size=())

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

        # Specs, Binary for bool flags (per TorchRL spec semantics) and
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

        # No order is submitted -- this env is paper-only (LIVE_UNSUPPORTED). The CLOB V2 port
        # re-enters HERE: submit before _wait_for_resolution, redeem the winnings after it.
        # The order-submission branch that used to sit here, and the four tests that pinned its
        # safety contract (a failed order must book zero, not phantom, P&L), were removed
        # together -- recover both with `git show 1a83d54^ -- torchtrade/ tests/`. They are not
        # a head start: the V1 call shape (buy() -> {"success": bool}) does not survive the
        # port, and reading the real wallet is NOT sufficient without the redeem.
        self._wait_for_resolution(market.end_date)
        outcome = self._poll_for_resolution(market)

        if outcome is None:
            logger.warning(
                "Market %s did not resolve within %s s of endDate; reward=0",
                market.slug, self.config.resolution_max_wait_seconds,
            )
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
        """Get the next upcoming market, sleeping and retrying on ``None``.

        Complements the scanner's per-call retry (which handles brief blips
        within a ~48 s budget), a longer Gamma outage during a 24h unattended
        run would still terminate the episode without this layer. For
        continuous short-cadence series, ``None`` is almost always "API
        unreachable" rather than "series ended", so we sleep and retry a few
        times before giving up.
        """
        for attempt in range(1, self.config.next_market_max_attempts + 1):
            market = self.scanner.next_active_market(self.config.market_slug_prefix)
            if market is not None:
                return market
            if attempt < self.config.next_market_max_attempts:
                logger.warning(
                    "No upcoming market for slug_prefix=%r (attempt %d/%d), "
                    "sleeping %.0fs before retrying",
                    self.config.market_slug_prefix, attempt,
                    self.config.next_market_max_attempts,
                    self.config.next_market_retry_sleep_seconds,
                )
                time.sleep(self.config.next_market_retry_sleep_seconds)
        return None

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
            logger.info(
                "Waiting %.0fs for market endDate %s + %.0fs grace",
                sleep_seconds, end_date_iso, self.config.resolution_grace_seconds,
            )
            time.sleep(sleep_seconds)

    def _poll_for_resolution(self, market: PolymarketMarket) -> Optional[int]:
        """Repeatedly fetch the resolved outcome, retrying until the market's
        CLOB midpoint snaps to the winning side or the wait budget is exhausted.

        Polymarket on-chain settlement typically takes 1-5 minutes after a
        market's endDate to fully snap on the CLOB; the initial single-shot
        check after the 30 s grace usually finds the market still mid-market.
        Tests should override this method to side-step the polling loop.
        """
        start = time.monotonic()
        deadline = start + self.config.resolution_max_wait_seconds
        attempt = 0
        while True:
            attempt += 1
            outcome = self._fetch_resolved_outcome(market)
            if outcome is not None:
                logger.info(
                    "Market resolved on attempt %d after %.0fs: %s won",
                    attempt, time.monotonic() - start, "Up" if outcome == 1 else "Down",
                )
                return outcome
            if time.monotonic() >= deadline:
                return None
            logger.info(
                "Resolution not yet on CLOB (attempt %d, elapsed %.0fs); "
                "retrying in %.0fs",
                attempt, time.monotonic() - start,
                self.config.resolution_poll_interval_seconds,
            )
            time.sleep(self.config.resolution_poll_interval_seconds)

    def _fetch_resolved_outcome(self, market: PolymarketMarket) -> Optional[int]:
        """Return 1 (Up won), 0 (Down won), or None if still unresolved.

        Reads CLOB midpoint per outcome token directly. The CLOB is the
        authoritative price source — its midpoint snaps to ~$1.00 / ~$0.00
        on the winning / losing side once the market resolves. Gamma's
        ``outcomePrices`` field is unreliable (we observed an in-flight
        market reporting ``[0.305, 0.695]`` on Gamma while CLOB midpoint
        was ``0.985``), and Gamma evicts short-cadence markets from
        ``/markets`` within minutes of endDate.
        """
        try:
            yes_mid = self._fetch_clob_midpoint(market.yes_token_id)
            no_mid = self._fetch_clob_midpoint(market.no_token_id)
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return None
        if yes_mid is None or no_mid is None:
            return None
        if yes_mid >= 0.99 and no_mid <= 0.01:
            return 1
        if no_mid >= 0.99 and yes_mid <= 0.01:
            return 0
        return None

    @staticmethod
    def _fetch_clob_midpoint(token_id: str) -> Optional[float]:
        """One-shot CLOB midpoint query for a single outcome token."""
        resp = requests.get(
            f"{CLOB_API_BASE}/midpoint",
            params={"token_id": token_id},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        mid = body.get("mid")
        return float(mid) if mid is not None else None

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
        """Same rule as the exchange envs, on this env's own terms.

        `cash`, not a portfolio value: a resolved bet has already been paid into cash by the
        time this runs, and there is no carried position to mark.
        """
        return is_bankrupt(
            current=self.cash,
            initial=self.config.initial_cash,
            threshold=self.config.bankrupt_threshold,
            enabled=self.config.done_on_bankruptcy,
        )

    def close(self):
        self.trader.cancel_all()
