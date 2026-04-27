"""PolyTimeBarEnv — Polymarket prediction market TorchRL trading environment."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical
from torchrl.data.tensor_specs import Composite

from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import HistoryTracker

logger = logging.getLogger(__name__)

_TIME_UNITS = ("Minute", "Hour", "Day")


def _parse_execute_on(value: str) -> tuple[int, str]:
    """Parse strings like '1Hour', '5Minute', '1Day' into (count, unit)."""
    for unit in _TIME_UNITS:
        if value.endswith(unit):
            try:
                return int(value[: -len(unit)]), unit
            except ValueError:
                break
    return 1, "Hour"


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

    # Termination
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Position management
    close_position_on_init: bool = True
    close_position_on_reset: bool = False

    # Mode
    dry_run: bool = False


class PolyTimeBarEnv(TorchTradeLiveEnv):
    """TorchRL environment for Polymarket prediction market trading.

    Steps on regular time bars. Manages a single market's YES/NO position.
    Supports supplementary observers for augmenting observations with
    external data sources (e.g., Binance OHLCV) — any object exposing
    ``get_observation_spec()`` and ``get_observations()`` works.

    The 6-element ``account_state`` follows TorchTrade's universal layout:
    ``[exposure_pct, position_direction, unrealized_pnl_pct, holding_time,
    leverage, distance_to_liquidation]``. Polymarket has no leverage or
    liquidation, so those slots are constant 1.0.
    """

    def __init__(
        self,
        config: PolyTimeBarEnvConfig,
        private_key: str = "",
        observer=None,
        trader=None,
        supplementary_observers: Optional[List] = None,
        reward_function: Optional[Callable] = None,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        if not (config.yes_token_id or config.condition_id or config.market_slug):
            raise ValueError(
                "At least one market identifier must be provided: "
                "yes_token_id, condition_id, or market_slug"
            )

        # Build observer/trader BEFORE super().__init__ so we can pass them
        # through the standard injection slot rather than stashing private state.
        if observer is None:
            from torchtrade.envs.live.polymarket.observation import (
                PolymarketObservationClass,
            )
            observer = PolymarketObservationClass(
                yes_token_id=config.yes_token_id,
                market_slug=config.market_slug,
                condition_id=config.condition_id,
                feature_preprocessing_fn=feature_preprocessing_fn,
            )
        if trader is None:
            from torchtrade.envs.live.polymarket.order_executor import (
                PolymarketOrderExecutor,
            )
            trader = PolymarketOrderExecutor(
                private_key=private_key,
                dry_run=config.dry_run,
            )

        self._supplementary_observers = supplementary_observers or []
        self.reward_function = reward_function or log_return_reward

        self.action_levels = config.action_levels
        self.history = HistoryTracker()
        self._step_count = 0
        self._last_yes_price = 0.0
        self._initial_balance = 0.0  # populated by _reset() once the wallet is known

        super().__init__(
            config=config,
            observer=observer,
            trader=trader,
            timezone="UTC",
        )

        self.execute_on_value, self.execute_on_unit = _parse_execute_on(
            config.execute_on
        )

        # Resolve token IDs (observer resolves NO from Gamma metadata)
        self._yes_token_id = config.yes_token_id or getattr(
            self.observer, "yes_token_id", ""
        )
        self._no_token_id = getattr(self.observer, "no_token_id", "")

        self._build_specs()

        if config.close_position_on_init:
            self._close_all_positions()

    def _init_trading_clients(self, api_key, api_secret, observer, trader):
        """Polymarket uses a Polygon private key, not api_key/api_secret pairs."""
        self.observer = observer
        self.trader = trader

    def _build_specs(self):
        """Build observation/action/done specs."""
        self.observation_spec = Composite(shape=())
        self.observation_spec.set(
            "market_state",
            Bounded(low=0.0, high=float("inf"), shape=(5,), dtype=torch.float32),
        )
        self.observation_spec.set(
            "account_state",
            Bounded(
                low=-float("inf"), high=float("inf"), shape=(6,), dtype=torch.float32
            ),
        )

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

        self.action_spec = Categorical(len(self.action_levels))

        # Declare done/terminated/truncated so check_env_specs accepts _step output
        self.full_done_spec = Composite(
            done=Categorical(2, dtype=torch.bool, shape=(1,)),
            terminated=Categorical(2, dtype=torch.bool, shape=(1,)),
            truncated=Categorical(2, dtype=torch.bool, shape=(1,)),
            shape=(),
        )

    def _close_all_positions(self):
        """Cancel open orders and close YES/NO positions."""
        if self.trader is None:
            return
        self.trader.cancel_all()
        if self._yes_token_id:
            self.trader.close_position(self._yes_token_id)
        if self._no_token_id:
            self.trader.close_position(self._no_token_id)

    def _get_observation(self) -> TensorDictBase:
        """Build observation TensorDict for current time bar."""
        obs = self.observer.get_observations()
        market_state = torch.tensor(obs["market_state"], dtype=torch.float32)
        self._last_yes_price = float(market_state[0])

        account_state = self._build_account_state(self._last_yes_price)

        td = TensorDict(
            {"market_state": market_state, "account_state": account_state},
            batch_size=(),
        )
        for supp in self._supplementary_observers:
            for key, val in supp.get_observations().items():
                td.set(key, torch.tensor(val, dtype=torch.float32))
        return td

    def _build_account_state(self, current_price: float) -> torch.Tensor:
        """Build 6-element account state tensor (universal TorchTrade layout)."""
        portfolio_value = self._get_portfolio_value()
        # Use current price (not stale entry-time position_value) so exposure
        # tracks live mark-to-market.
        current_position_value = abs(self.position.position_size) * current_price

        if portfolio_value > 0 and current_position_value > 0:
            exposure_pct = current_position_value / portfolio_value
        else:
            exposure_pct = 0.0

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
                1.0,  # leverage
                1.0,  # distance_to_liquidation
            ],
            dtype=torch.float32,
        )

    def _get_portfolio_value(self) -> float:
        """USDC balance + position market value at the latest cached price."""
        cash = self.trader.get_balance()
        price = self._last_yes_price or self.observer.get_yes_price()
        position_value = abs(self.position.position_size) * price
        return cash + position_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset: clear history, optionally close positions, return initial obs."""
        self.history.reset()
        self.position.reset()
        self._step_count = 0
        self._last_yes_price = 0.0

        if self.config.close_position_on_reset:
            self._close_all_positions()

        obs = self._get_observation()
        self._initial_balance = self._get_portfolio_value()
        return obs

    def _resolve_action_idx(self, raw) -> int:
        """Coerce a tensordict action to a valid index, mirroring OKX env."""
        if isinstance(raw, torch.Tensor):
            raw = raw.item()
        if isinstance(raw, float) and math.isfinite(raw):
            raw = int(raw)
        if not isinstance(raw, int):
            logger.warning("Invalid action %r — defaulting to flat", raw)
            return self._flat_action_idx
        if raw < 0 or raw >= len(self.action_levels):
            logger.warning(
                "Action index %d out of range [0, %d] — clamping",
                raw, len(self.action_levels) - 1,
            )
            return max(0, min(raw, len(self.action_levels) - 1))
        return raw

    @property
    def _flat_action_idx(self) -> int:
        """Index of the 'go flat' action (0.0); falls back to the middle index."""
        try:
            return self.action_levels.index(0)
        except ValueError:
            return len(self.action_levels) // 2

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step: process action, wait for next bar, return obs."""
        action_idx = self._resolve_action_idx(tensordict.get("action"))
        desired_action = self.action_levels[action_idx]

        yes_price = self.observer.get_yes_price()
        self._last_yes_price = yes_price

        self._execute_trade_if_needed(desired_action, yes_price)

        self._wait_for_next_timestamp()

        if self.position.current_position != 0:
            self.position.hold_counter += 1
        else:
            self.position.hold_counter = 0

        self._step_count += 1

        td_out = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()

        if desired_action > 0:
            action_label = "buy_yes"
        elif desired_action < 0:
            action_label = "buy_no"
        else:
            action_label = "hold"
        self.history.record_step(
            price=self._last_yes_price,
            action=desired_action,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=self.position.current_position,
            action_type=action_label,
        )
        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        terminated = bool(self.observer.is_market_closed())
        if (
            self.config.done_on_bankruptcy
            and self._initial_balance > 0
            and new_portfolio_value
            < self.config.bankrupt_threshold * self._initial_balance
        ):
            terminated = True

        truncated = bool(
            self.config.max_steps is not None
            and self._step_count >= self.config.max_steps
        )

        done = terminated or truncated

        td_out.set("reward", torch.tensor([reward], dtype=torch.float32))
        td_out.set("done", torch.tensor([done], dtype=torch.bool))
        td_out.set("terminated", torch.tensor([terminated], dtype=torch.bool))
        td_out.set("truncated", torch.tensor([truncated], dtype=torch.bool))
        return td_out

    def _execute_trade_if_needed(self, desired_action: float, current_price: float):
        """Allocate a portfolio fraction to YES (positive) or NO (negative)."""
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

        # Direction flip — close existing leg first
        if current_direction != 0 and current_direction != target_direction:
            token_id = (
                self._yes_token_id if current_direction > 0 else self._no_token_id
            )
            self.trader.close_position(token_id)
            self.position.reset()

        portfolio_value = self._get_portfolio_value()
        target_value = portfolio_value * target_fraction
        current_value = abs(self.position.position_size) * current_price

        delta_value = target_value - current_value
        if delta_value < 1.0:
            return

        token_id = self._yes_token_id if target_direction > 0 else self._no_token_id
        result = self.trader.buy(token_id=token_id, amount_usdc=delta_value)

        if result.get("success"):
            shares_bought = delta_value / current_price if current_price > 0 else 0
            self.position.current_position = target_direction
            self.position.position_size += shares_bought
            if self.position.entry_price == 0:
                self.position.entry_price = current_price

    def close(self):
        """Cancel orders; warn if a position is still open."""
        if self.trader is None:
            return
        self.trader.cancel_all()
        if self.position.current_position != 0:
            logger.warning(
                "PolyTimeBarEnv closed with open position (direction=%.0f). "
                "Position was NOT automatically closed.",
                self.position.current_position,
            )
