"""Sequential Trading Environment with automatic mode detection.

A TorchRL-compatible environment for algorithmic trading that automatically
adapts based on leverage and action_levels configuration.

Key Features:
    - 6-element account state for all configurations
    - leverage > 1: Enables liquidation mechanics
    - leverage == 1: No liquidation (spot-like)
    - negative action_levels: Allows short positions
    - all positive action_levels: Long-only
    - Fractional position sizing with configurable action levels
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.core.offline_base import TorchTradeOfflineEnv
from torchtrade.envs.core.state import (
    advance_hold_counter_from_size,
    binarize_action_type,
)
from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.utils.timeframe import TimeFrame, normalize_timeframe_config
from torchtrade.envs.utils.sltp_helpers import stop_fill_price
from torchtrade.envs.utils.fractional_sizing import (
    calculate_fractional_position,
    PositionCalculationParams,
    validate_action_levels,
    AFFORDABILITY_REL_TOL,
    POSITION_TOLERANCE_PCT,
    POSITION_TOLERANCE_NOTIONAL,
)
from torchtrade.envs.core.common_types import MarginType
from torchtrade.envs.utils.liquidation import (
    DEFAULT_MAINTENANCE_MARGIN_RATE,
    isolated_liquidation_price,
)


@dataclass
class SequentialTradingEnvConfig:
    """Configuration for sequential trading environment.

    Trading behavior is automatically determined by parameters:
    - leverage > 1: Enables futures mode (liquidation mechanics)
    - negative action_levels: Enables short positions
    - leverage == 1 and no negative actions: Spot mode (long-only, no liquidation)
    """
    # Common parameters
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"
    initial_cash: Union[Tuple[int, int], int, float] = 10000
    transaction_fee: float = 0.0
    slippage: float = 0.0
    bankrupt_threshold: float = 0.1

    # Environment settings
    seed: Optional[int] = 42
    include_base_features: bool = False
    max_traj_length: Optional[int] = None
    random_start: bool = True

    # Action space configuration
    action_levels: Optional[List[float]] = field(default_factory=lambda: [-1, 0, 1])

    # Trading parameters
    # leverage > 1: Enables futures mode with liquidation mechanics
    # leverage == 1: Spot mode (no liquidation)
    leverage: int = 1
    margin_type: MarginType = MarginType.ISOLATED
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE

    def __post_init__(self):
        """Validate configuration and normalize timeframes."""
        # Normalize timeframe configuration
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Validate leverage
        if not (1 <= self.leverage <= 125):
            raise ValueError(
                f"Leverage must be between 1 and 125, got {self.leverage}"
            )

        # Termination thresholds, range mirrors the vectorized config.
        # bankrupt_threshold=0.0 removes the positive termination floor (spot positions
        # never hit it; leveraged positions can still terminate on portfolio_value <= 0).
        # maintenance_margin_rate=0.0 = no maintenance-margin buffer. >=1 is nonsensical.
        if not (0 <= self.bankrupt_threshold < 1):
            raise ValueError(
                f"bankrupt_threshold must be in [0, 1), got {self.bankrupt_threshold}"
            )
        if not (0 <= self.maintenance_margin_rate < 1):
            raise ValueError(
                f"maintenance_margin_rate must be in [0, 1), got {self.maintenance_margin_rate}"
            )

        # Validate action levels
        validate_action_levels(self.action_levels)


class SequentialTradingEnv(TorchTradeOfflineEnv):
    """Sequential trading environment with automatic mode detection.

    Trading behavior is automatically determined:
    - leverage > 1: Futures mode with liquidation mechanics
    - leverage == 1: Spot mode without liquidation
    - negative action_levels: Allows short positions
    - all action_levels >= 0: Long-only

    Examples:
    ---------
    Spot (long-only):
        leverage=1, action_levels=[0, 1]  # Default: flat, long
        No liquidation, no shorts

    Leveraged long-only:
        leverage=10, action_levels=[0, 1]
        Has liquidation, but no shorts

    Futures (bidirectional):
        leverage=10, action_levels=[-1, 0, 1]  # Default: short, flat, long
        Has liquidation and shorts

    Universal Account State (6 elements for both modes):
    ----------------------------------------------------
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Element definitions:
        - exposure_pct: position_value / portfolio_value (0.0 to 1.0+ with leverage)
        - position_direction: sign(position_size) (-1=short, 0=flat, +1=long)
        - unrealized_pnl_pct: (current_price - entry_price) / entry_price * direction
        - holding_time: steps since position opened
        - leverage: 1.0 for spot, 1.0-125.0 for futures
        - distance_to_liquidation: normalized distance to liquidation (1.0 for spot/no position)

    Spot mode values:
        - position_direction: 0 or +1 (no shorts)
        - leverage: Always 1.0
        - distance_to_liquidation: Always 1.0 (no liquidation risk)

    Futures mode values:
        - position_direction: -1, 0, or +1 (full bidirectional)
        - leverage: 1.0 to 125.0
        - distance_to_liquidation: (price - liq_price) / price for longs,
                                    (liq_price - price) / price for shorts

    Action Space (Fractional Mode):
    -------------------------------
    Actions represent the fraction of portfolio to allocate.

    Spot mode (default: 3 actions):
        - 0.0: Close position (go to cash)
        - 0.5: Invest 50% of portfolio
        - 1.0: Invest 100% of portfolio (all-in)

    Futures mode (default: 5 actions):
        - -1.0: 100% short (all-in short)
        - -0.5: 50% short
        - 0.0: Market neutral (close all positions)
        - 0.5: 50% long
        - 1.0: 100% long (all-in long)

    Custom action levels are supported for both modes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SequentialTradingEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
    ):
        """Initialize the sequential trading environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration
            feature_preprocessing_fn: Optional function to preprocess features
            reward_function: Optional reward function (default: log_return_reward)
        """
        # Initialize base class (handles sampler, history, balance, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Set reward function (default to log return reward)
        self.reward_function = reward_function or log_return_reward

        # Store configuration
        self.action_levels = config.action_levels
        self.leverage = config.leverage
        self.margin_type = config.margin_type
        self.maintenance_margin_rate = config.maintenance_margin_rate

        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Warn if negative action_levels with leverage=1 (shorts impossible in spot mode)
        has_negative_actions = any(a < 0 for a in self.action_levels)
        if has_negative_actions and self.leverage == 1:
            warnings.warn(
                "Negative action_levels provided with leverage=1. "
                "Short positions are not possible in spot mode - "
                "negative actions will be clipped to 0 (flat). "
                "Either remove negative action_levels or set leverage > 1.",
                UserWarning,
                stacklevel=2,
            )

        # Build observation specs with universal 6-element account state
        account_state = [
            "exposure_pct", "position_direction", "unrealized_pnlpct",
            "holding_time", "leverage", "distance_to_liquidation"
        ]
        # Use per-timeframe feature counts to support different feature dimensions
        num_features_per_tf = self.sampler.get_num_features_per_timeframe()
        self._build_observation_specs(account_state, num_features_per_tf)

        # Initialize futures-specific tracking state
        # (These are calculation helpers, not part of PositionState)
        self.unrealized_pnl = 0.0  # Absolute unrealized PnL
        self.unrealized_pnl_pct = 0.0  # Percentage unrealized PnL
        self.liquidation_price = 0.0  # Liquidation price

    @property
    def has_liquidation(self) -> bool:
        """Check if liquidation mechanics are enabled (leverage > 1)."""
        return self.leverage > 1

    @property
    def allows_short(self) -> bool:
        """Check if short positions are allowed.

        Requires both:
        - Negative action_levels configured
        - Futures mode (leverage > 1)

        In spot mode (leverage=1), shorts are impossible regardless of action_levels.
        """
        has_negative_actions = any(a < 0 for a in self.action_levels)
        return has_negative_actions and self.leverage > 1


    def _calculate_liquidation_price(self, entry_price: float, position_size: float) -> float:
        """Calculate liquidation price for a position.

        This is the gating -- no liquidation at leverage 1, none without a position. The
        arithmetic itself lives in `isolated_liquidation_price`, shared with the live
        envs' fallback so the two cannot price the same position differently; restating
        the formula here is how that starts drifting again.

        Note it now raises, rather than returning 0.0, on an entry price of 0: a zero
        liquidation price is indistinguishable downstream from having no position.
        """
        if not self.has_liquidation:
            return 0.0

        if position_size == 0:
            return 0.0

        return isolated_liquidation_price(
            entry_price,
            is_long=position_size > 0,
            leverage=self.leverage,
            maintenance_margin_rate=self.maintenance_margin_rate,
        )

    def _check_liquidation(self, ohlcv: dict) -> bool:
        """Check if current position should be liquidated using intrabar OHLCV data.

        Checks the bar's extreme (low for longs, high for shorts) against the
        liquidation price to catch intrabar wicks.
        """
        if not self.has_liquidation:
            return False

        if self.position.position_size == 0:
            return False

        if self.position.position_size > 0:
            # Long position - liquidated if low touches/crosses below liquidation
            return ohlcv["low"] <= self.liquidation_price
        else:
            # Short position - liquidated if high touches/crosses above liquidation
            return ohlcv["high"] >= self.liquidation_price

    def _calculate_unrealized_pnl(
        self, entry_price: float, current_price: float, position_size: float
    ) -> float:
        """Calculate absolute unrealized PnL.

        For long: PnL = (current_price - entry_price) * position_size
        For short: PnL = (entry_price - current_price) * abs(position_size)
        """
        if position_size == 0 or entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) * position_size
        else:
            # Short position
            return (entry_price - current_price) * abs(position_size)

    def _calculate_unrealized_pnl_pct(
        self, entry_price: float, current_price: float, position_size: float
    ) -> float:
        """Calculate unrealized PnL as a percentage of entry value.

        Returns PnL as fraction of initial notional value.
        For both long and short, this is the return on the position.
        """
        if entry_price == 0 or position_size == 0:
            return 0.0

        # Reuse absolute PnL calculation
        pnl_absolute = self._calculate_unrealized_pnl(entry_price, current_price, position_size)

        # Normalize by entry notional value
        entry_notional = abs(position_size * entry_price)
        return pnl_absolute / entry_notional if entry_notional > 0 else 0.0

    def _calculate_distance_to_liquidation(
        self, current_price: float, liquidation_price: float, position_size: float
    ) -> float:
        """Calculate normalized distance to liquidation price.

        Returns 1.0 when liquidation is disabled or no position (no liquidation risk).
        When liquidation is enabled, returns distance as fraction of current price.

        Args:
            current_price: Current market price
            liquidation_price: Liquidation price (0.0 when liquidation disabled)
            position_size: Current position size (signed)

        Returns:
            Distance to liquidation (1.0 = no risk, 0.0 = at liquidation)
        """
        # No liquidation or no position: no liquidation risk
        if not self.has_liquidation or position_size == 0:
            return 1.0

        if current_price == 0:
            return 1.0

        if position_size > 0:
            # Long position - liquidated if price drops below liquidation_price
            distance = (current_price - liquidation_price) / current_price
        else:
            # Short position - liquidated if price rises above liquidation_price
            distance = (liquidation_price - current_price) / current_price

        # Clamp to [0, inf) - negative means already liquidated
        return max(0.0, distance)

    def _get_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """Calculate total portfolio value.

        For spot (leverage=1): PV = balance + position_value
        For futures (leverage>1): PV = free_margin + locked_margin + unrealized_pnl
        """
        if current_price is None:
            if self.current_timestamp is None:
                raise RuntimeError(
                    "current_timestamp is not set. _get_portfolio_value() must be called "
                    "after _get_observation() which sets the current timestamp."
                )
            current_price = self._cached_base_features["close"]

        if self.leverage == 1:
            # Spot mode: balance is cash after buying, add current position value
            position_value = abs(self.position.position_size) * current_price if self.position.position_size != 0 else 0.0
            return self.balance + position_value
        else:
            # Futures mode: balance is free margin, add locked margin and unrealized PnL
            # locked_margin = notional / leverage (what was deducted from balance when opening)
            locked_margin = 0.0
            if self.position.position_size != 0:
                position_notional = abs(self.position.position_size * self.position.entry_price)
                locked_margin = position_notional / self.leverage

            unrealized_pnl = self._calculate_unrealized_pnl(
                self.position.entry_price, current_price, self.position.position_size
            )
            return self.balance + locked_margin + unrealized_pnl

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data (scaffold sets current_timestamp and truncated)
        obs_dict, base_features = self._get_observation_scaffold()
        self._cached_base_features = base_features
        return self._build_observation_from_data(obs_dict, base_features)

    def _build_observation_from_data(self, obs_dict: dict, base_features: dict) -> TensorDictBase:
        """Build observation TensorDict from already-fetched market data.

        Args:
            obs_dict: Market data observations for each timeframe.
            base_features: Base OHLCV features for the current timestamp.

        Returns:
            TensorDict with account state and market data.
        """
        current_price = base_features["close"]

        # Calculate position value (absolute value of notional)
        self.position.position_value = abs(self.position.position_size * current_price)

        # Calculate unrealized PnL
        if self.position.position_size != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(
                self.position.entry_price, current_price, self.position.position_size
            )
            self.unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(
                self.position.entry_price, current_price, self.position.position_size
            )
        else:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0

        # Calculate portfolio value and exposure
        portfolio_value = self._get_portfolio_value(current_price)
        exposure_pct = self.position.position_value / portfolio_value if portfolio_value > 0 else 0.0

        # Calculate position direction (-1, 0, +1)
        if self.position.position_size > 0:
            position_direction = 1.0
        elif self.position.position_size < 0:
            position_direction = -1.0
        else:
            position_direction = 0.0

        # Calculate distance to liquidation
        distance_to_liquidation = self._calculate_distance_to_liquidation(
            current_price, self.liquidation_price, self.position.position_size
        )

        # Universal 6-element account state
        # [exposure_pct, position_direction, unrealized_pnl_pct,
        #  holding_time, leverage, distance_to_liquidation]
        account_state = torch.tensor([
            exposure_pct,                          # Element 0: exposure_pct
            position_direction,                    # Element 1: position_direction (-1, 0, +1)
            self.unrealized_pnl_pct,              # Element 2: unrealized_pnl_pct
            float(self.position.hold_counter),     # Element 3: holding_time
            float(self.leverage),                  # Element 4: leverage
            distance_to_liquidation,               # Element 5: distance_to_liquidation
        ], dtype=torch.float)

        # Combine account state and market data
        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))

        return TensorDict(obs_data, batch_size=())

    def _reset_position_state(self):
        """Reset position tracking state."""
        super()._reset_position_state()
        # Reset futures-specific state
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        self._prev_action_value = None
        # Reset cached exhaustion response (built lazily on first post-done step)
        self._exhaustion_td = None
        self._last_state_index = 0

    def _build_exhaustion_response(self) -> TensorDictBase:
        """Build terminal response when sampler is already exhausted.

        Called when self.truncated is True at the start of _step(), meaning
        the sampler ran out of data in the previous step. Returns done=True
        without advancing the sampler.

        Idempotent: the response is built once and cloned on subsequent calls
        so that repeated post-done steps don't mutate history or other state.
        """
        if self._exhaustion_td is not None:
            return self._exhaustion_td.clone()

        # Re-fetch observation from last timestamp (doesn't use sequential index)
        obs_dict = self.sampler.get_observation(self.current_timestamp)
        td = self._build_observation_from_data(obs_dict, self._cached_base_features)

        if self.random_start:
            td.set("reset_index", torch.tensor(self._reset_idx, dtype=torch.long))
            td.set("state_index", torch.tensor(self._last_state_index, dtype=torch.long))

        td.set("reward", torch.zeros(1, dtype=torch.float))
        td.set("terminated", torch.tensor([False], dtype=torch.bool))
        td.set("truncated", torch.tensor([True], dtype=torch.bool))
        td.set("done", torch.tensor([True], dtype=torch.bool))

        self._exhaustion_td = td.clone()
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step.

        Order: the agent's action fills at close(N), then bar N+1 is applied to whatever
        position that leaves. Checking the incoming bar first served a wick that had
        already breached liquidation to the policy as a healthy position (#281), and
        gating the trade on that check discarded a legitimate close or switch (#292).
        SequentialTradingEnvSLTP has kept this order since #294 and #297.
        """
        self.step_counter += 1

        # Guard: if sampler was exhausted in the previous step, terminate
        # gracefully instead of letting get_sequential_observation() raise.
        if self.truncated:
            return self._build_exhaustion_response()

        # Bar N price -- where the agent's action would execute
        cached_price = self._cached_base_features["close"]

        # Get desired action
        action_idx = tensordict["action"]
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Advance to bar N+1 before anything is decided against it (#281).
        obs_dict, base_features = self._get_observation_scaffold()
        self._cached_base_features = base_features

        # Unconditional: the fill happens before bar N+1 exists (#292).
        trade_info = self._execute_trade_if_needed(desired_action, cached_price)

        # Bar N+1, against whatever the action above left open. Must precede the portfolio
        # value and the history record below, or both read a state that ignores this exit.
        if self._check_liquidation(base_features):
            trade_info = self._execute_liquidation(base_features["open"])

        # Age the position once per step through the canonical rule, whatever happened
        # above (#275). Five hand-rolled sites used to do this -- hold, tolerance-hold,
        # open, close, liquidation -- and the resize path was simply never given one, so
        # a position that was resized every bar reported holding_time=1 forever. Deriving
        # the direction from the post-trade size means every path is covered by
        # construction, including paths added later.
        advance_hold_counter_from_size(self.position)

        # Update position flag based on actual position size
        if trade_info["executed"]:
            if self.position.position_size > 0:
                self.position.current_position = 1  # Long
            elif self.position.position_size < 0:
                self.position.current_position = -1  # Short
            else:
                self.position.current_position = 0  # Flat

        # Built from bar N+1, which the liquidation check above has already been applied to.
        next_tensordict = self._build_observation_from_data(obs_dict, base_features)
        new_price = base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            self._last_state_index = self.sampler._sequential_idx
            next_tensordict.set("reset_index", torch.tensor(self._reset_idx, dtype=torch.long))
            next_tensordict.set("state_index", torch.tensor(self._last_state_index, dtype=torch.long))

        # Determine action_type and binarize action for history
        action_type = trade_info.get("side") or "hold"
        binarized_action = binarize_action_type(action_type)

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=new_price,
            action=binarized_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=self.position.position_size,
            action_type=action_type
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination (bankruptcy) and truncation (time/data limit)
        terminated = self._check_termination(new_portfolio_value)
        truncated = self._check_truncation()

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("terminated", torch.tensor([terminated], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([truncated], dtype=torch.bool))
        next_tensordict.set("done", torch.tensor([terminated or truncated], dtype=torch.bool))

        return next_tensordict

    def _calculate_fractional_position(
        self, action_value: float, current_price: float
    ) -> Tuple[float, float, str]:
        """Calculate position size from fractional action value.

        Uses shared utility function for consistent position sizing.

        If shorts not allowed: Clips negative actions to 0.0 (close position)
        If shorts allowed: Supports full bidirectional trading

        IMPORTANT: Uses portfolio value (balance + unrealized PnL) instead of just balance
        to avoid forced trading after opening positions. With leverage or full investment,
        balance can be near zero but portfolio value stays constant.

        Args:
            action_value: Action from [-1.0, 1.0] (or [0.0, 1.0] if no shorts)
            current_price: Current market price

        Returns:
            Tuple of (position_size, notional_value, side)
        """
        # Clip negative actions if shorts not allowed
        if not self.allows_short and action_value < 0:
            action_value = 0.0

        # Use portfolio value instead of balance to avoid forced trading
        portfolio_value = self._get_portfolio_value(current_price)

        params = PositionCalculationParams(
            balance=portfolio_value,
            action_value=action_value,
            current_price=current_price,
            leverage=self.leverage,
            transaction_fee=self.transaction_fee,
        )
        position_size, notional_value, side = calculate_fractional_position(params)

        # Convert "long" to "buy" for clarity when shorts not allowed
        if not self.allows_short and side == "long":
            side = "buy"

        return position_size, notional_value, side

    def _execute_trade_if_needed(self, desired_action: float, base_price: float) -> Dict:
        """Execute trade using fractional position sizing.

        base_price is required. It used to default to reading current_timestamp, which
        was harmless while _step checked liquidation before advancing -- but _step now
        advances to bar N+1 before trading (#281), so that fallback would price the fill
        off the wrong bar. Better to fail loudly than to reintroduce it silently.
        """
        # Apply slippage
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage, generator=self._rng).item()
        execution_price = base_price * price_noise

        # Execute fractional action
        return self._execute_fractional_action(desired_action, execution_price)

    def _execute_fractional_action(self, action_value: float, execution_price: float) -> Dict:
        """Execute action using fractional position sizing.

        Handles all position lifecycle operations:
        - Opening new positions from flat
        - Closing positions to go neutral
        - Switching direction (futures only)
        - Adjusting position size
        - Implicit holding

        Args:
            action_value: Fractional action value in [-1.0, 1.0]
            execution_price: Price at which trade executes (includes slippage)

        Returns:
            trade_info: Dict with execution details
        """
        # If action hasn't changed and we already have a position, just hold
        if action_value == self._prev_action_value and self.position.position_size != 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}
        self._prev_action_value = action_value

        # Calculate target position from action value
        target_position_size, target_notional, target_side = (
            self._calculate_fractional_position(action_value, execution_price)
        )

        # A CLOSE is exempt: going flat is not a resize, and a sub-$1 position must
        # stay closable (#339).
        tolerance = (
            0.0 if target_position_size == 0.0
            else max(
                abs(target_position_size) * POSITION_TOLERANCE_PCT,
                POSITION_TOLERANCE_NOTIONAL / execution_price,
            )
        )

        # Check if already at target position (implicit hold)
        # `position_size != 0` mirrors the vectorized engine's `has_position` term. The
        # floor asks whether a RESIZE is worth the fee; an open from flat is not a resize
        # any more than a close is. Without it the two engines disagreed on every open
        # worth under $1 -- and the scalar could never re-enter, an absorbing dead state.
        if self.position.position_size != 0 and (
            abs(target_position_size - self.position.position_size) < tolerance
        ):
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Execute appropriate position change
        if target_position_size == 0.0:
            # Close to neutral
            if self.position.position_size == 0:
                return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

            trade_info = self._close_position(execution_price)
            trade_info["side"] = "sell" if not self.allows_short else "flat"
            return trade_info

        if self.position.position_size == 0.0:
            # Open new position
            return self._open_position(target_side, target_position_size, target_notional, execution_price)

        # Check for direction switch (long→short or short→long)
        has_long_position = self.position.position_size > 0
        has_short_position = self.position.position_size < 0
        wants_long = target_position_size > 0
        wants_short = target_position_size < 0

        is_direction_switch = (
            (has_long_position and wants_short) or
            (has_short_position and wants_long)
        )

        if is_direction_switch:
            # Close current position then open opposite
            self._close_position(execution_price)

            # Recalculate target position after closing (balance may have changed)
            target_position_size, target_notional, target_side = (
                self._calculate_fractional_position(action_value, execution_price)
            )

            return self._open_position(target_side, target_position_size, target_notional, execution_price)

        # Adjust position size in same direction
        return self._adjust_position_size(target_position_size, target_notional, execution_price)

    def _clamp_balance(self) -> None:
        if self.balance < -1e-10:
            logger.warning(
                f"Negative balance detected: {self.balance:.6f}. "
                f"Position: {self.position.position_size}, Entry: {self.position.entry_price}. "
                "This likely indicates an accounting bug."
            )
        self.balance = max(0.0, self.balance)

    def _open_position(
        self, side: str, position_size: float, notional_value: float, execution_price: float
    ) -> Dict:
        """Open a new position from flat."""
        # Calculate margin and fee
        margin_required = notional_value / self.leverage
        fee = abs(notional_value) * self.transaction_fee

        if margin_required + fee > self.balance * (1 + AFFORDABILITY_REL_TOL):
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Deduct fee and margin
        # For spot (leverage=1): margin_required = notional_value (full cost)
        # For futures (leverage>1): margin_required = notional_value / leverage
        self.balance -= fee + margin_required
        self._clamp_balance()

        # Set position
        self.position.position_size = position_size
        self.position.position_value = abs(notional_value)
        self.position.entry_price = execution_price

        # Set position direction
        if not self.allows_short:
            self.position.current_position = 1  # Always long
        else:
            self.position.current_position = 1 if side in ("buy", "long") else -1

        # Calculate liquidation price (futures only)
        self.liquidation_price = self._calculate_liquidation_price(execution_price, position_size)

        return {"executed": True, "side": side, "fee_paid": fee, "liquidated": False}

    def _adjust_position_size(
        self, target_position_size: float, target_notional: float, execution_price: float
    ) -> Dict:
        """Adjust existing position size (same direction)."""
        delta_position = target_position_size - self.position.position_size
        delta_notional = abs(delta_position * execution_price)

        # Determine if increasing or decreasing
        is_increasing = abs(target_position_size) > abs(self.position.position_size)

        if is_increasing:
            return self._increase_position_size(
                target_position_size, target_notional, delta_position, delta_notional, execution_price
            )

        return self._decrease_position_size(target_position_size, target_notional, execution_price)

    def _increase_position_size(
        self,
        target_position_size: float,
        target_notional: float,
        delta_position: float,
        delta_notional: float,
        execution_price: float
    ) -> Dict:
        """Increase position size by adding to existing position."""
        fee = delta_notional * self.transaction_fee
        margin_required = delta_notional / self.leverage

        if margin_required + fee > self.balance * (1 + AFFORDABILITY_REL_TOL):
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Execute trade - deduct fee and additional margin
        self.balance -= fee + margin_required
        self._clamp_balance()

        # Calculate weighted average entry price (weight by quantity, not notional)
        old_qty = abs(self.position.position_size)
        new_qty = abs(delta_position)
        total_qty = old_qty + new_qty

        if total_qty > 0:
            self.position.entry_price = (
                (self.position.entry_price * old_qty + execution_price * new_qty) / total_qty
            )

        # Update position
        self.position.position_size = target_position_size
        self.position.position_value = abs(target_notional)

        # Recalculate liquidation price
        self.liquidation_price = self._calculate_liquidation_price(
            self.position.entry_price, self.position.position_size
        )

        side = "buy" if not self.allows_short else ("long" if delta_position > 0 else "short")
        return {"executed": True, "side": side, "fee_paid": fee, "liquidated": False}

    def _decrease_position_size(
        self, target_position_size: float, target_notional: float, execution_price: float
    ) -> Dict:
        """Decrease position size by partially closing."""
        fraction_to_close = 1.0 - (abs(target_position_size) / abs(self.position.position_size))

        # Calculate PnL and fee on portion being closed
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price,
            execution_price,
            self.position.position_size * fraction_to_close
        )

        close_notional = abs(self.position.position_size * fraction_to_close * execution_price)
        fee = close_notional * self.transaction_fee

        # Calculate freed margin (based on entry price, not current price)
        freed_margin = abs(self.position.position_size * fraction_to_close * self.position.entry_price) / self.leverage

        # Update balance: add PnL, subtract fee, return freed margin
        self.balance += pnl - fee + freed_margin
        self._clamp_balance()

        # Update position
        self.position.position_size = target_position_size
        self.position.position_value = abs(target_notional)

        # Liquidation price remains the same
        self.liquidation_price = self._calculate_liquidation_price(
            self.position.entry_price, self.position.position_size
        )

        side = "sell" if not self.allows_short else "close_partial"
        return {"executed": True, "side": side, "fee_paid": fee, "liquidated": False}

    def _close_position(self, execution_price: float) -> Dict:
        """Close existing position."""
        if self.position.position_size == 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Calculate PnL and fee
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, execution_price, self.position.position_size
        )
        notional = abs(self.position.position_size * execution_price)
        fee = notional * self.transaction_fee
        # Return the margin that was locked when opening
        margin_to_return = abs(self.position.position_size * self.position.entry_price) / self.leverage

        # Update balance: add realized PnL, subtract fee, return locked margin
        self.balance += pnl - fee + margin_to_return
        self._clamp_balance()

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0

        return {"executed": True, "side": "close", "fee_paid": fee, "liquidated": False}

    def _execute_liquidation(self, open_price: float) -> Dict:
        """Execute forced liquidation of position (futures only).

        Books at the price the bar actually offered, not at the liquidation price a bar
        gapping straight through it never traded at (#314) -- the same rule #311 gave
        stop-losses, via the same helper.

        The loss is then capped at the margin posted for this position, because that is
        what isolated margin means: the venue closes at the bankruptcy price and the
        insurance fund absorbs whatever the gap costs beyond the margin. Without the cap
        the gap rule would overshoot in the other direction and lose more than the
        account ever committed. The cap binds ONLY on a gap -- at the liquidation price
        itself the loss is the margin less the maintenance buffer, by construction.
        """
        trade_info = {
            "executed": True,
            "side": "liquidation",
            "fee_paid": 0.0,
            "liquidated": True,
        }

        is_long = self.position.position_size > 0
        fill_price = stop_fill_price(self.liquidation_price, open_price, is_long)
        # Clamped to the bankruptcy price -- the price at which the position has consumed
        # exactly its margin. Beyond it the venue closes and the insurance fund absorbs
        # the rest, so nothing the account owns is priced further out. Clamping the FILL
        # rather than capping the loss keeps the fee on the same price as the PnL: a cap
        # applied to the loss alone left the fee tracking the gap, so a deeper crash was
        # CHEAPER and, at the 1% fee this repo's own fixtures use, cheaper than booking
        # at the liquidation price at all.
        # Net of the exit fee: the venue takes the liquidation fee OUT of the margin --
        # that is what the maintenance buffer is for -- so the bankruptcy price is the
        # fill where loss + fee equals the margin exactly, entry*(1 -+ 1/L)/(1 -+ f).
        # Charging the fee on top of a full-margin loss overdrew the account by the fee
        # on every gapped liquidation, which _clamp_balance() then absorbed while logging
        # "this likely indicates an accounting bug" about ordinary input (#314).
        f = self.transaction_fee
        bankruptcy_price = self.position.entry_price * (
            (1 - 1 / self.leverage) / (1 - f) if is_long
            else (1 + 1 / self.leverage) / (1 + f)
        )
        fill_price = (
            max(fill_price, bankruptcy_price) if is_long
            else min(fill_price, bankruptcy_price)
        )

        if is_long:
            loss = (fill_price - self.position.entry_price) * self.position.position_size
        else:
            loss = (self.position.entry_price - fill_price) * abs(self.position.position_size)

        # Return locked margin before applying loss and fees
        margin_to_return = abs(self.position.position_size * self.position.entry_price) / self.leverage

        # Apply loss, fees, and return margin
        liquidation_fee = abs(self.position.position_size * fill_price) * self.transaction_fee
        self.balance += loss - liquidation_fee + margin_to_return
        self._clamp_balance()
        trade_info["fee_paid"] = liquidation_fee

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate (environment dynamics only, e.g. bankruptcy)."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold

    def _check_truncation(self) -> bool:
        """Check if episode should be truncated (time limit or data exhaustion)."""
        return self.truncated or self.step_counter >= self.max_traj_length

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    import datasets
    import numpy as np

    # Load sample data
    print("Loading dataset...")
    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = df["train"].to_pandas()
    df['0'] = pd.to_datetime(df['0'])
    print(f"Dataset loaded: {len(df)} rows\n")

    # Create environment
    config = SequentialTradingEnvConfig(
        execute_on="1Hour",
        time_frames="1Hour",
        initial_cash=10000,
        transaction_fee=0.0,
        slippage=0.0,
    )
    env = SequentialTradingEnv(df, config)
    print(f"Environment created")
    print(f"Action levels: {env.action_levels}")
    print(f"Action space size: {env.action_spec.n}\n")

    # Run episode
    td = env.reset()
    print("Starting episode...\n")

    for i in range(2000):
        print(f"Step {i}")
        print("Account state:")
        acc_state = ""
        for key, value in zip(env.account_state, td["account_state"]):
            acc_state += f"  {key}: {value:.6f}\n"
        print(acc_state)

        # Random action (mostly hold, occasionally trade)
        action = np.random.choice([0, 1, 2], p=[0.80, 0.10, 0.10])  # 80% hold, 10% short, 10% long
        print(f"Action: {action} ({env.action_levels[action]})\n")

        td["action"] = action
        td = env.step(td)
        td = td["next"]

        if td["done"].item():
            print("Episode terminated!")
            break

    # Render history
    print("\nRendering history...")
    env.render_history(return_fig=False)

    env.close()
    print("\nDone!")
