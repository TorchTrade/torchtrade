"""PolyTimeBarEnv — Polymarket prediction market TorchRL trading environment."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, runtime_checkable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical
from torchrl.data.tensor_specs import Composite

from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import HistoryTracker

logger = logging.getLogger(__name__)


@runtime_checkable
class SupplementaryObserver(Protocol):
    """Protocol for supplementary data sources (e.g., Binance OHLCV)."""

    def get_observation_spec(self) -> dict:
        """Return dict mapping key names to TorchRL TensorSpec."""
        ...

    def get_observations(self) -> dict:
        """Return current observations as dict of arrays/tensors."""
        ...


# Map execute_on strings like "1Hour", "5Minute", "1Day" to (value, unit)
_EXECUTE_ON_PATTERN = re.compile(r"^(\d+)(Minute|Hour|Day)$")


@dataclass
class PolyTimeBarEnvConfig:
    """Configuration for PolyTimeBarEnv."""

    # Market identification — priority: yes_token_id > condition_id > market_slug
    # At least one must be non-empty
    market_slug: str = ""
    condition_id: str = ""
    yes_token_id: str = ""

    # Stepping
    execute_on: str = "1Hour"
    max_steps: Optional[int] = None

    # Actions
    action_levels: List[float] = field(default_factory=lambda: [-1, 0, 1])

    # Capital
    initial_cash: float = 10_000.0

    # Termination
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Position management
    close_position_on_init: bool = True
    close_position_on_reset: bool = False

    # Mode
    dry_run: bool = False

    seed: Optional[int] = 42


class PolyTimeBarEnv(TorchTradeLiveEnv):
    """TorchRL environment for Polymarket prediction market trading.

    Steps on regular time bars. Manages a single market's YES/NO position.
    Supports supplementary observers for augmenting observations with
    external data sources (e.g., Binance OHLCV).
    """

    ACCOUNT_STATE_LABELS = [
        "exposure_pct",
        "position_direction",
        "unrealized_pnl_pct",
        "holding_time",
        "leverage",
        "distance_to_liquidation",
    ]

    def __init__(
        self,
        config: PolyTimeBarEnvConfig,
        private_key: str = "",
        observer=None,
        trader=None,
        supplementary_observers: Optional[List[SupplementaryObserver]] = None,
        reward_function: Optional[Callable] = None,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        # Validate market identification
        if not (config.yes_token_id or config.condition_id or config.market_slug):
            raise ValueError(
                "At least one market identifier must be provided: "
                "yes_token_id, condition_id, or market_slug"
            )

        self._private_key = private_key
        self._supplementary_observers = supplementary_observers or []
        self.reward_function = reward_function or log_return_reward
        self._feature_preprocessing_fn = feature_preprocessing_fn

        self.action_levels = config.action_levels
        self.history = HistoryTracker()
        self._step_count = 0
        self._initial_balance = config.initial_cash

        # Call parent — passes empty strings for api_key/api_secret
        # Our _init_trading_clients override uses self._private_key instead
        super().__init__(
            config=config,
            api_key="",
            api_secret="",
            observer=observer,
            trader=trader,
            timezone="UTC",
        )

        # Parse execute_on into value + unit for _wait_for_next_timestamp()
        match = _EXECUTE_ON_PATTERN.match(config.execute_on)
        if match:
            self.execute_on_value = int(match.group(1))
            self.execute_on_unit = match.group(2)
        else:
            self.execute_on_value = 1
            self.execute_on_unit = "Hour"

        # Resolve token IDs from observer (observer resolves from Gamma API)
        self._yes_token_id = config.yes_token_id or getattr(self.observer, "yes_token_id", "")
        self._no_token_id = getattr(self.observer, "no_token_id", "")

        # Build specs after parent init (observer/trader are now set)
        self._build_observation_specs()
        self.action_spec = Categorical(len(self.action_levels))

        # Close position on init if configured
        if config.close_position_on_init and self.trader is not None:
            self.trader.cancel_all()
            if self._yes_token_id:
                self.trader.close_position(self._yes_token_id)
            if self._no_token_id:
                self.trader.close_position(self._no_token_id)

    def _init_trading_clients(self, api_key, api_secret, observer, trader):
        """Initialize observer and trader using private_key (ignores api_key/api_secret)."""
        self.observer = observer
        self.trader = trader

        if self.observer is None:
            from torchtrade.envs.live.polymarket.observation import (
                PolymarketObservationClass,
            )

            self.observer = PolymarketObservationClass(
                yes_token_id=self.config.yes_token_id,
                market_slug=self.config.market_slug,
                condition_id=self.config.condition_id,
                feature_preprocessing_fn=self._feature_preprocessing_fn,
            )

        if self.trader is None:
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )

            self.trader = PolymarketOrderExecutor(
                private_key=self._private_key,
                dry_run=self.config.dry_run,
            )

    def _build_observation_specs(self):
        """Build observation spec from market_state + account_state + supplementary."""
        self.observation_spec = Composite(shape=())

        # Market state (5,)
        self.observation_spec.set(
            "market_state",
            Bounded(low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32),
        )

        # Account state (6,) — standard across all TorchTrade envs
        self.observation_spec.set(
            "account_state",
            Bounded(
                low=-float("inf"), high=float("inf"), shape=(6,), dtype=torch.float32
            ),
        )

        # Supplementary observer specs
        reserved_keys = set(self.observation_spec.keys())
        for supp in self._supplementary_observers:
            for key, spec in supp.get_observation_spec().items():
                if key in reserved_keys:
                    raise ValueError(
                        f"Supplementary observer key collision: '{key}' "
                        f"already exists in observation spec"
                    )
                self.observation_spec.set(key, spec)
                reserved_keys.add(key)

    def _get_observation(self) -> TensorDictBase:
        """Build observation TensorDict for current time bar."""
        # Market state from observer
        obs = self.observer.get_observations()
        market_state = torch.tensor(obs["market_state"], dtype=torch.float32)

        # Cache yes_price from the observation to avoid redundant CLOB call
        self._last_yes_price = float(market_state[0])

        # Account state
        account_state = self._build_account_state(self._last_yes_price)

        td = TensorDict(
            {"market_state": market_state, "account_state": account_state},
            batch_size=(),
        )

        # Supplementary observations
        for supp in self._supplementary_observers:
            for key, val in supp.get_observations().items():
                td.set(key, torch.tensor(val, dtype=torch.float32))

        return td

    def _build_account_state(self, current_price: float) -> torch.Tensor:
        """Build 6-element account state tensor."""
        portfolio_value = self._get_portfolio_value()

        if portfolio_value > 0 and self.position.position_value > 0:
            exposure_pct = self.position.position_value / portfolio_value
        else:
            exposure_pct = 0.0

        # Unrealized PnL
        if self.position.entry_price > 0 and self.position.current_position != 0:
            unrealized_pnl = (
                (current_price - self.position.entry_price)
                / self.position.entry_price
                * self.position.current_position
            )
        else:
            unrealized_pnl = 0.0

        return torch.tensor(
            [
                exposure_pct,
                self.position.current_position,
                unrealized_pnl,
                float(self.position.hold_counter),
                1.0,  # leverage (always 1.0)
                1.0,  # distance_to_liquidation (always 1.0)
            ],
            dtype=torch.float32,
        )

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value: USDC balance + position market value."""
        cash = self.trader.get_balance()
        current_price = getattr(self, "_last_yes_price", 0.0) or self.observer.get_yes_price()
        position_value = abs(self.position.position_size) * current_price
        return cash + position_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset: clear history, optionally close positions, return initial obs."""
        self.history.reset()
        self.position.reset()
        self._step_count = 0
        self._last_yes_price = 0.0

        if self.config.close_position_on_reset and self.trader is not None:
            self.trader.cancel_all()
            if self._yes_token_id:
                self.trader.close_position(self._yes_token_id)
            if self._no_token_id:
                self.trader.close_position(self._no_token_id)

        obs = self._get_observation()
        self._initial_balance = self._get_portfolio_value()
        return obs

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step: process action, wait for next bar, return obs."""
        action_idx = tensordict.get("action", torch.tensor(0))
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()

        desired_action = self.action_levels[action_idx]
        yes_price = self.observer.get_yes_price()
        self._last_yes_price = yes_price

        # Execute trade
        self._execute_trade_if_needed(desired_action, yes_price)

        # Wait for next time bar (blocking — same as all other live envs)
        self._wait_for_next_timestamp()

        # Update position hold counter
        if self.position.current_position != 0:
            self.position.hold_counter += 1
        else:
            self.position.hold_counter = 0

        self._step_count += 1

        # Get updated observation (refreshes prices from exchange)
        td_out = self._get_observation()

        # Get updated portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # Record history
        action_label = "hold"
        if desired_action > 0:
            action_label = "buy_yes"
        elif desired_action < 0:
            action_label = "buy_no"

        self.history.record_step(
            price=self._last_yes_price,
            action=desired_action,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=self.position.current_position,
            action_type=action_label,
        )

        # Calculate reward
        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        # Check termination
        terminated = False
        truncated = False

        if self.observer.is_market_closed():
            terminated = True

        if (
            self.config.done_on_bankruptcy
            and self._initial_balance > 0
            and new_portfolio_value
            < self.config.bankrupt_threshold * self._initial_balance
        ):
            terminated = True

        if (
            self.config.max_steps is not None
            and self._step_count >= self.config.max_steps
        ):
            truncated = True

        done = terminated or truncated

        td_out.set("reward", torch.tensor([reward], dtype=torch.float32))
        td_out.set("done", torch.tensor([done], dtype=torch.bool))
        td_out.set("terminated", torch.tensor([terminated], dtype=torch.bool))
        td_out.set("truncated", torch.tensor([truncated], dtype=torch.bool))

        return td_out

    def _execute_trade_if_needed(self, desired_action: float, current_price: float):
        """Execute trade based on desired fractional action.

        Action mapping (portfolio-fraction, Alpaca pattern):
        - positive -> allocate fraction to YES position
        - negative -> allocate fraction to NO position
        - zero -> go flat
        """
        current_direction = self.position.current_position

        if desired_action == 0.0:
            if current_direction != 0:
                token_id = (
                    self._yes_token_id if current_direction > 0 else self._no_token_id
                )
                self.trader.close_position(token_id)
                self.position.reset()
            return

        target_direction = 1.0 if desired_action > 0 else -1.0
        target_fraction = abs(desired_action)

        # If direction changed, close existing position first
        if current_direction != 0 and current_direction != target_direction:
            token_id = (
                self._yes_token_id if current_direction > 0 else self._no_token_id
            )
            self.trader.close_position(token_id)
            self.position.reset()

        # Calculate target position value
        portfolio_value = self._get_portfolio_value()
        target_value = portfolio_value * target_fraction
        current_value = abs(self.position.position_value)

        delta_value = target_value - current_value
        if delta_value < 1.0:
            return

        # Buy the appropriate token
        token_id = self._yes_token_id if target_direction > 0 else self._no_token_id
        result = self.trader.buy(token_id=token_id, amount_usdc=delta_value)

        if result.get("success"):
            shares_bought = delta_value / current_price if current_price > 0 else 0
            self.position.current_position = target_direction
            self.position.position_size += shares_bought
            self.position.position_value = (
                abs(self.position.position_size) * current_price
            )
            if self.position.entry_price == 0:
                self.position.entry_price = current_price
            self.position.current_action_level = desired_action

    def close(self):
        """Clean up: cancel orders and warn about open positions."""
        if self.trader is not None:
            self.trader.cancel_all()
            if self.position.current_position != 0:
                logger.warning(
                    "PolyTimeBarEnv closed with open position (direction=%.0f). "
                    "Position was NOT automatically closed.",
                    self.position.current_position,
                )
