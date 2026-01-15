"""
Futures One-Step Environment for GRPO-style training.

Combines futures trading mechanics (leverage, long/short positions, liquidation)
with the one-step rollout pattern from LongOnlyOneStepEnv.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import math

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchtrade.envs.offline.base import TorchTradeOfflineEnv
import torch
from torchrl.data import Bounded, Categorical
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta, compute_periods_per_year_crypto, InitialBalanceSampler, build_sltp_action_map, normalize_timeframe_config
from torchtrade.envs.reward import build_reward_context, default_log_return, validate_reward_function
from torchtrade.envs.state import FuturesHistoryTracker
from torchtrade.envs.offline.regime_features import MarketRegimeFeatures
import logging
import sys

log_level = logging.INFO

# Read log level from command line (e.g. DEBUG)
if len(sys.argv) > 1:
    log_level_name = sys.argv[1].upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(levelname)s:%(name)s:%(message)s"
)

logger = logging.getLogger(__name__)


class MarginType(Enum):
    """Margin type for futures trading."""
    ISOLATED = "isolated"
    CROSSED = "crossed"


@dataclass
class FuturesOneStepEnvConfig:
    """Configuration for Futures One-Step Environment.

    This environment supports:
    - Long and short positions with SLTP bracket orders
    - Configurable leverage (1x - 125x)
    - Liquidation mechanics
    - One-step rollout pattern for GRPO training
    """
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Min"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Min"

    # Initial capital settings
    initial_cash: Union[List[int], int] = (1000, 5000)

    # Leverage and margin settings
    leverage: int = 1  # 1x to 125x
    margin_type: MarginType = MarginType.ISOLATED
    maintenance_margin_rate: float = 0.004  # 0.4% maintenance margin

    # Trading costs
    transaction_fee: float = 0.0004  # 0.04% typical futures fee
    slippage: float = 0.001  # 0.1% slippage

    # SLTP levels
    stoploss_levels: Union[List[float], float] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], float] = (0.05, 0.1, 0.2)

    # Risk management
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    max_position_size: float = 1.0  # Max position as fraction of balance

    # Environment settings
    seed: Optional[int] = 42
    include_base_features: bool = False
    max_traj_length: Optional[int] = None
    random_start: bool = True  # Always True for one-step environments

    # Reward settings
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)
    reward_scaling: float = 1.0
    include_hold_action: bool = True  # Include HOLD action (index 0) in action space

    # Market regime feature settings
    include_regime_features: bool = False  # Include market regime features in observations
    regime_volatility_window: int = 20  # Window for volatility calculation
    regime_trend_window: int = 50  # Window for long-term trend MA
    regime_trend_short_window: int = 20  # Window for short-term trend MA
    regime_volume_window: int = 20  # Window for volume analysis
    regime_price_position_window: int = 252  # Window for price position (52-week ~= 252 daily bars)

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )


class FuturesOneStepEnv(TorchTradeOfflineEnv):
    """
    Futures One-Step Environment for GRPO-style RL training.

    Combines futures trading mechanics (leverage, long/short, liquidation)
    with the one-step rollout pattern where each step:
    1. Takes an action (hold, long+SLTP, short+SLTP)
    2. Rolls out until SLTP triggers, liquidation, or truncation
    3. Returns done=True (one-step environment)

    Action Space:
    - Action 0: Hold (no position)
    - Actions 1 to N: Long with SL/TP combinations
    - Actions N+1 to 2N: Short with SL/TP combinations

    Account State (10 elements, matching live futures env):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    """

    # Futures-specific account state (includes leverage, margin_ratio, liquidation_price)
    ACCOUNT_STATE = [
        "cash", "position_size", "position_value", "entry_price", "current_price",
        "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        config: FuturesOneStepEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        """
        Initialize the FuturesOneStepEnv.

        Args:
            df: DataFrame with OHLCV data
            config: Environment configuration
            feature_preprocessing_fn: Optional custom preprocessing function
        """
        # Initialize base class (handles sampler, history, balance, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Validate custom reward function signature if provided
        if config.reward_function is not None:
            validate_reward_function(config.reward_function)

        # Environment-specific configuration
        self.leverage = config.leverage
        self.margin_type = config.margin_type
        self.maintenance_margin_rate = config.maintenance_margin_rate

        # Validate config parameters
        if not (1 <= config.leverage <= 125):
            raise ValueError("Leverage must be between 1 and 125.")

        self.episode_idx = 0
        self.periods_per_year = compute_periods_per_year_crypto(self.execute_on_unit, self.execute_on_value)

        # SLTP levels
        self.stoploss_levels = list(config.stoploss_levels) if not isinstance(config.stoploss_levels, list) else config.stoploss_levels
        self.takeprofit_levels = list(config.takeprofit_levels) if not isinstance(config.takeprofit_levels, list) else config.takeprofit_levels

        # Create action map
        self.action_map = build_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_hold_action=config.include_hold_action,
            include_short_positions=True
        )
        self.action_spec = Categorical(len(self.action_map))

        # Build observation specs using class constant
        num_features = len(self.sampler.get_feature_keys())
        self._build_observation_specs(self.ACCOUNT_STATE, num_features)

        # Initialize market regime features if enabled
        self.include_regime_features = config.include_regime_features
        if self.include_regime_features:
            self.regime_calculator = MarketRegimeFeatures(
                volatility_window=config.regime_volatility_window,
                trend_window=config.regime_trend_window,
                trend_short_window=config.regime_trend_short_window,
                volume_window=config.regime_volume_window,
                price_position_window=config.regime_price_position_window,
            )

            # Add regime features to observation spec
            regime_spec = Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=(7,),  # 7 regime features
                dtype=torch.float32
            )
            self.observation_spec["regime_features"] = regime_spec
            logger.debug(
                f"Market regime features enabled with windows: "
                f"vol={config.regime_volatility_window}, "
                f"trend={config.regime_trend_window}/{config.regime_trend_short_window}, "
                f"volume={config.regime_volume_window}, "
                f"position={config.regime_price_position_window}"
            )

        # Reward spec
        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)

        # Initialize futures-specific state (beyond base PositionState)
        # Note: These attributes are intentionally separate from PositionState as they are
        # specific to futures/leveraged trading and not applicable to spot/long-only environments.
        self.unrealized_pnl = 0.0  # Absolute unrealized PnL (calculated from leverage)
        self.unrealized_pnl_pct = 0.0  # Percentage unrealized PnL
        self.liquidation_price = 0.0  # Price at which position would be liquidated

        # Initialize one-step specific state
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.previous_portfolio_value = 0.0
        self.rollout_returns = []

        # PERF: Pre-allocate account state tensor to avoid allocation each step
        self._account_state_buffer = torch.zeros(10, dtype=torch.float)

        # PERF: Initialize portfolio value cache
        self._cached_portfolio_value = 0.0
        self._cached_portfolio_price = 0.0

        # Force random_start to True for one-step environments (contextual bandit setting requires diverse starts)
        if not config.random_start:
            logger.warning(
                "FuturesOneStepEnv requires random_start=True for proper one-step/contextual bandit training. "
                "Ignoring config.random_start=False and forcing random_start=True."
            )
        self.random_start = True

    def _reset_history(self):
        """Reset all history tracking arrays including position history for futures."""
        self.history = FuturesHistoryTracker()

    def _reset_position_state(self):
        """Reset position tracking state including futures and one-step specific state."""
        super()._reset_position_state()
        # Futures-specific state
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        # One-step specific state
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.previous_portfolio_value = 0.0
        self.rollout_returns = []

    def _calculate_liquidation_price(self, entry_price: float, position_size: float) -> float:
        """
        Calculate liquidation price for a position.

        For ISOLATED margin:
        - Long: liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        - Short: liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        """
        if position_size == 0:
            return 0.0

        margin_fraction = 1.0 / self.leverage

        if position_size > 0:
            # Long position - liquidated if price drops
            liquidation_price = entry_price * (1 - margin_fraction + self.maintenance_margin_rate)
        else:
            # Short position - liquidated if price rises
            liquidation_price = entry_price * (1 + margin_fraction - self.maintenance_margin_rate)

        return max(0, liquidation_price)

    def _calculate_margin_required(self, position_value: float) -> float:
        """Calculate initial margin required for a position."""
        return abs(position_value) / self.leverage

    def _calculate_unrealized_pnl(self, entry_price: float, current_price: float, position_size: float) -> float:
        """
        Calculate unrealized PnL.

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

    def _calculate_unrealized_pnl_pct(self, entry_price: float, current_price: float, position_size: float) -> float:
        """Calculate unrealized PnL as a percentage."""
        if entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) / entry_price
        elif position_size < 0:
            # Short position
            return (entry_price - current_price) / entry_price
        return 0.0

    def _check_liquidation(self, current_price: float) -> bool:
        """Check if current position should be liquidated."""
        if self.position.position_size == 0:
            return False

        if self.position.position_size > 0:
            # Long position - liquidated if price below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position - liquidated if price above liquidation price
            return current_price >= self.liquidation_price

    def _get_observation(self, initial: bool = False) -> TensorDictBase:
        """Get the current observation state."""
        if initial or self.position.position_size == 0:
            # PERF: Use combined method to avoid redundant searchsorted
            obs_dict, self.current_timestamp, self.truncated, self._cached_base_features = (
                self.sampler.get_sequential_observation_with_ohlcv()
            )
            self.rollout_returns = []
        else:
            # _rollout() updates _cached_base_features internally
            trade_info, obs_dict = self._rollout()

        # Use cached base features (PERF: attribute access on namedtuple)
        current_price = self._cached_base_features.close

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

        # Calculate margin ratio
        total_balance = self.balance + self.unrealized_pnl
        margin_ratio = self.position.position_value / total_balance if total_balance > 0 else 0.0

        # PERF: Cache portfolio value and price to avoid recalculation in _get_portfolio_value()
        self._cached_portfolio_value = total_balance
        self._cached_portfolio_price = current_price

        # PERF: Update pre-allocated account state buffer in-place
        self._account_state_buffer[0] = self.balance
        self._account_state_buffer[1] = self.position.position_size  # Positive=long, Negative=short
        self._account_state_buffer[2] = self.position.position_value
        self._account_state_buffer[3] = self.position.entry_price
        self._account_state_buffer[4] = current_price
        self._account_state_buffer[5] = self.unrealized_pnl_pct
        self._account_state_buffer[6] = float(self.leverage)
        self._account_state_buffer[7] = margin_ratio
        self._account_state_buffer[8] = self.liquidation_price
        self._account_state_buffer[9] = float(self.position.hold_counter)

        # Clone to avoid issues with in-place modifications downstream
        obs_data = {self.account_state_key: self._account_state_buffer.clone()}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))

        # Add regime features if enabled
        if self.include_regime_features:
            try:
                # Get historical data for regime feature calculation
                # Use max required window to ensure sufficient data
                max_window = self.regime_calculator.min_data_required
                prices = self.sampler.get_recent_prices(window=max_window)
                volumes = self.sampler.get_recent_volumes(window=max_window)

                # Compute regime features
                regime_features = self.regime_calculator.compute_features(prices, volumes)
                obs_data["regime_features"] = regime_features
            except ValueError as e:
                # Only catch insufficient data errors at the start of episodes
                if "Insufficient data" in str(e):
                    logger.debug(f"Not enough data for regime features: {e}. Using defaults.")
                    # Use neutral/default regime features: [vol=1(med), trend=0(sideways),
                    # volume=1(normal), position=1(neutral), vol=0.01, trend=0.0, vol_ratio=1.0]
                    obs_data["regime_features"] = torch.tensor([1, 0, 1, 1, 0.01, 0.0, 1.0], dtype=torch.float32)
                else:
                    # Re-raise unexpected ValueErrors
                    raise

        return TensorDict(obs_data, batch_size=())

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action_tuple: tuple,
        trade_info: Dict,
    ) -> float:
        """
        Calculate reward using custom or default function.

        If config.reward_function is provided, uses that custom function.
        Otherwise, uses the default log return reward.

        Args:
            old_portfolio_value: Portfolio value before action
            new_portfolio_value: Portfolio value after action
            action_tuple: Action taken (side, sl, tp)
            trade_info: Dictionary with trade execution details

        Returns:
            Reward value (float)
        """
        # Use custom reward function if provided
        if self.config.reward_function is not None:
            # Calculate margin ratio
            total_balance = self.balance + self.unrealized_pnl
            margin_ratio = self.position.position_value / total_balance if total_balance > 0 else 0.0

            ctx = build_reward_context(
                self,
                old_portfolio_value,
                new_portfolio_value,
                action_tuple,
                trade_info,
                portfolio_value_history=self.history.portfolio_values,
                action_history=self.history.actions,
                reward_history=self.history.rewards,
                base_price_history=self.history.base_prices,
                position_history=self.history.positions,
                initial_portfolio_value=self.initial_portfolio_value,
                rollout_returns=self.rollout_returns,
                liquidated=trade_info.get("liquidated", False),
                leverage=float(self.leverage),
                margin_ratio=margin_ratio,
                liquidation_price=self.liquidation_price,
            )
            return float(self.config.reward_function(ctx)) * self.config.reward_scaling

        # Otherwise use default log return (no context needed)
        return default_log_return(old_portfolio_value, new_portfolio_value) * self.config.reward_scaling

    def _get_portfolio_value(self, current_price: float = None) -> float:
        """Calculate total portfolio value including unrealized PnL.

        PERF: Uses cached value when price matches the last cached price from _get_observation().
        """
        if current_price is None:
            current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # PERF: Use cached portfolio value if price matches (avoids recalculating unrealized PnL)
        if hasattr(self, '_cached_portfolio_price') and current_price == self._cached_portfolio_price:
            return self._cached_portfolio_value

        unrealized_pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, current_price, self.position.position_size
        )
        return self.balance + unrealized_pnl

    def _set_seed(self, seed: int):
        """Set the seed for the environment."""
        self.seed = seed

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Cache base features (PERF: attribute access on namedtuple)
        cached_price = self._cached_base_features.close

        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get action
        action = tensordict["action"]
        action_tuple = self.action_map[action.item()]
        logger.debug(f"Action: {action_tuple}")

        # Execute trade if needed
        trade_info = self._execute_trade_if_needed(action_tuple, cached_price)
        if trade_info["executed"]:
            logger.debug(f"Trade executed: {trade_info}")

        # Get updated state (this advances timestamp and caches new base features)
        next_tensordict = self._get_observation()

        # Use newly cached base features for new portfolio value (PERF: attribute access)
        new_price = self._cached_base_features.close
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            # Ensure state_index is within valid range for coverage tracking
            # During rollout, _sequential_idx may exceed the valid training data range
            max_valid_idx = len(self.sampler._exec_times_arr) - 1
            state_idx = min(self.sampler._sequential_idx, max_valid_idx)
            next_tensordict.set("state_index", torch.tensor(state_idx, dtype=torch.long))

        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action_tuple, trade_info)

        if self.truncated:
            logger.debug(f"Episode truncated after {self.step_counter} steps")
            reward = 0  # No reward for truncated episodes

        # Record step history
        trade_action = 0
        if trade_info["executed"]:
            if trade_info["side"] == "long":
                trade_action = 1
            elif trade_info["side"] == "short":
                trade_action = -1
        self.history.record_step(
            price=cached_price,
            action=trade_action,
            reward=reward,
            portfolio_value=old_portfolio_value,
            position=self.position.position_size
        )

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", True)  # One-step environment
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", True)

        return next_tensordict

    def compute_return(self, close_price: float) -> float:
        """Compute log return for current step.

        PERF: Returns float instead of tensor. Tensor conversion happens once
        in _calculate_reward() to eliminate tensor allocation per rollout step.
        """
        current_value = self._get_portfolio_value(close_price)
        # PERF: Use math.log instead of creating tensors for each computation
        if self.previous_portfolio_value > 0:
            log_return = math.log(current_value / self.previous_portfolio_value)
        else:
            log_return = 0.0
        self.previous_portfolio_value = current_value
        return log_return

    def _trigger_close(self, trade_info: Dict, execution_price: float) -> Dict:
        """Close position at given price."""
        logger.debug(f"Triggering close at price: {execution_price}")

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, execution_price, self.position.position_size
        )

        # Calculate fee
        notional = abs(self.position.position_size * execution_price)
        fee_paid = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee_paid

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0
        self.position.hold_counter = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

        trade_info.update({
            "executed": True,
            "side": "close",
            "success": True,
            "fee_paid": fee_paid,
            "pnl": pnl,
        })

        return trade_info

    def _trigger_liquidation(self, trade_info: Dict) -> Dict:
        """Execute forced liquidation."""
        logger.debug(f"Liquidation triggered at price: {self.liquidation_price}")

        # Realize loss at liquidation price
        if self.position.position_size > 0:
            loss = (self.liquidation_price - self.position.entry_price) * self.position.position_size
        else:
            loss = (self.position.entry_price - self.liquidation_price) * abs(self.position.position_size)

        # Apply loss and fees
        liquidation_fee = abs(self.position.position_size * self.liquidation_price) * self.transaction_fee
        self.balance += loss - liquidation_fee

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0
        self.position.hold_counter = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

        trade_info.update({
            "executed": True,
            "side": "liquidation",
            "success": True,
            "fee_paid": liquidation_fee,
            "liquidated": True,
        })

        return trade_info

    def _rollout(self) -> tuple:
        """
        Roll out position until SLTP triggers, liquidation, or truncation.

        Returns:
            (trade_info, obs_dict) tuple
        """
        trade_info = {
            "executed": False, "side": None, "success": None,
            "fee_paid": 0.0, "liquidated": False
        }
        self.rollout_returns = []
        obs_dict = None

        # PERF: Cache trigger prices and position info as locals (avoid attribute lookup per iteration)
        liq_price = self.liquidation_price
        sl_price = self.stop_loss_price
        tp_price = self.take_profit_price
        is_long = self.position.position_size > 0
        is_short = self.position.position_size < 0

        future_rollout_steps = 1
        while not self.truncated:
            # PERF: Get observation AND OHLCV in one call (avoids redundant searchsorted)
            obs_dict, self.current_timestamp, self.truncated, ohlcv = (
                self.sampler.get_sequential_observation_with_ohlcv()
            )
            self._cached_base_features = ohlcv

            # PERF: Use attribute access (namedtuple) instead of dict lookup
            open_price = ohlcv.open
            close_price = ohlcv.close
            high_price = ohlcv.high
            low_price = ohlcv.low

            self.rollout_returns.append(self.compute_return(close_price))

            # PERF: Simplified trigger logic using cached locals
            # For long: low is min (check SL), high is max (check TP)
            # For short: high is max (check SL), low is min (check TP)
            if is_long:
                # Long position - liquidated if price drops to liquidation
                if low_price <= liq_price:
                    return self._trigger_liquidation(trade_info), obs_dict
                # SL: Only need to check low (it's the min in OHLCV bar)
                if low_price < sl_price:
                    return self._trigger_close(trade_info, sl_price), obs_dict
                # TP: Only need to check high (it's the max in OHLCV bar)
                if high_price > tp_price:
                    return self._trigger_close(trade_info, tp_price), obs_dict

            elif is_short:
                # Short position - liquidated if price rises to liquidation
                if high_price >= liq_price:
                    return self._trigger_liquidation(trade_info), obs_dict
                # SL: Only need to check high (it's the max in OHLCV bar)
                if high_price > sl_price:
                    return self._trigger_close(trade_info, sl_price), obs_dict
                # TP: Only need to check low (it's the min in OHLCV bar)
                if low_price < tp_price:
                    return self._trigger_close(trade_info, tp_price), obs_dict

            future_rollout_steps += 1

        logger.debug(f"No trigger after: {future_rollout_steps} rollout steps")

        # If loop never executed (truncated from start), get an observation
        if obs_dict is None:
            # PERF: Use combined method to avoid redundant searchsorted
            obs_dict, self.current_timestamp, self.truncated, self._cached_base_features = (
                self.sampler.get_sequential_observation_with_ohlcv()
            )

        return trade_info, obs_dict

    def _execute_trade_if_needed(self, action_tuple: tuple, base_price: float = None) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {
            "executed": False, "side": None, "success": None,
            "fee_paid": 0.0, "liquidated": False
        }

        side, sl_pct, tp_pct = action_tuple

        if side is None:
            # Hold action - no trade
            logger.debug("Hold action - no trade")
            return trade_info

        # Get base price
        if base_price is None:
            base_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Apply slippage
        price_noise = 1.0  # Could add: torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = base_price * price_noise

        # Calculate position size based on available balance and leverage
        usable_balance = self.balance * self.config.max_position_size
        margin_plus_fee_rate = (1.0 / self.leverage) + self.transaction_fee
        max_notional = usable_balance / margin_plus_fee_rate * 0.999
        notional_value = max_notional
        position_qty = notional_value / execution_price

        # Calculate fee
        fee = notional_value * self.transaction_fee

        # Deduct fee
        self.balance -= fee
        trade_info["fee_paid"] = fee

        if side == "long":
            self.position.position_size = position_qty
            self.position.current_position = 1
            # Long: SL below entry, TP above entry
            self.stop_loss_price = execution_price * (1 + sl_pct)  # sl_pct is negative
            self.take_profit_price = execution_price * (1 + tp_pct)
        else:  # short
            self.position.position_size = -position_qty
            self.position.current_position = -1
            # Short: SL above entry, TP below entry
            self.stop_loss_price = execution_price * (1 - sl_pct)  # sl_pct is negative, so this raises the price
            self.take_profit_price = execution_price * (1 - tp_pct)  # tp_pct is positive, so this lowers the price

        self.position.entry_price = execution_price
        self.position.position_value = notional_value
        self.liquidation_price = self._calculate_liquidation_price(execution_price, self.position.position_size)
        self.position.hold_counter = 0
        self.previous_portfolio_value = self._get_portfolio_value(execution_price)

        logger.debug(f"Entry: {execution_price}, SL: {self.stop_loss_price}, TP: {self.take_profit_price}")
        logger.debug(f"Liquidation price: {self.liquidation_price}")

        trade_info.update({
            "executed": True,
            "side": side,
            "success": True,
        })

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold or self.step_counter >= self.max_steps

    def close(self):
        """Clean up resources."""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute episode metrics from history.

        Returns:
            Dictionary containing:
                - total_return: Total portfolio return
                - sharpe_ratio: Annualized Sharpe ratio
                - sortino_ratio: Annualized Sortino ratio
                - calmar_ratio: Calmar ratio (return / max drawdown)
                - max_drawdown: Maximum drawdown (negative value)
                - max_dd_duration: Maximum drawdown duration in periods
                - num_trades: Number of trades executed
                - win_rate (reward>0): Percentage of profitable periods
                - avg_win: Average win amount
                - avg_loss: Average loss amount
                - profit_factor: Ratio of total wins to total losses
        """
        from torchtrade.metrics import compute_all_metrics
        from torchtrade.envs.offline.utils import compute_periods_per_year_crypto

        # Convert histories to tensors
        portfolio_values = torch.tensor(self.history.portfolio_values, dtype=torch.float32)
        rewards = torch.tensor(self.history.rewards, dtype=torch.float32)

        # Compute periods per year for annualization
        periods_per_year = compute_periods_per_year_crypto(
            self.execute_on_unit,
            self.execute_on_value
        )

        # Use shared metrics computation function
        return compute_all_metrics(
            portfolio_values=portfolio_values,
            rewards=rewards,
            action_history=self.history.actions,
            periods_per_year=periods_per_year,
        )


if __name__ == "__main__":
    import pandas as pd

    time_frames = [TimeFrame(15, TimeFrameUnit.Minute)]
    window_sizes = [32]
    execute_on = TimeFrame(15, TimeFrameUnit.Minute)

    df = pd.read_csv(
        "/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv"
    )

    config = FuturesOneStepEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        leverage=5,
        transaction_fee=0.0004,
        slippage=0.001,
        stoploss_levels=[-0.02, -0.05],
        takeprofit_levels=[0.05, 0.1],
    )
    env = FuturesOneStepEnv(df, config)

    for i in range(50):
        td = env.reset()
        print(f"Episode: {i}")
        for j in range(env.max_steps):
            action = env.action_spec.sample()
            td.set("action", action)
            td = env.step(td)
            td = td["next"]
            print(f"Step: {j}, Action: {action.item()}, Reward: {td['reward'].item():.4f}")
            assert not torch.isnan(td["reward"])
            if td["done"]:
                print(f"Done: {td['done'].item()}")
                break
