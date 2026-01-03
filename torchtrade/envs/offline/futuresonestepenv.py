"""
Futures One-Step Environment for GRPO-style training.

Combines futures trading mechanics (leverage, long/short positions, liquidation)
with the one-step rollout pattern from LongOnlyOneStepEnv.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from itertools import product
from enum import Enum
import math

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Bounded, Categorical
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta, compute_periods_per_year_crypto
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


def futures_onestep_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
) -> Dict[int, tuple]:
    """
    Create action map for futures one-step environment.

    Actions:
    - 0: Hold (no action)
    - 1 to N: Long positions with SL/TP combinations
    - N+1 to 2N: Short positions with SL/TP combinations

    Args:
        stoploss_levels: List of stop-loss percentages (e.g., [-0.02, -0.05])
        takeprofit_levels: List of take-profit percentages (e.g., [0.05, 0.1])

    Returns:
        Dictionary mapping action indices to (side, sl, tp) tuples
        where side is None (hold), "long", or "short"
    """
    action_map = {}
    # Action 0 = HOLD
    action_map[0] = (None, None, None)

    idx = 1
    # Long positions
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = ("long", sl, tp)
        idx += 1

    # Short positions
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = ("short", sl, tp)
        idx += 1

    return action_map


class InitialBalanceSampler:
    """Sampler for initial balance with optional randomization."""

    def __init__(self, initial_cash: Union[List[int], int], seed: Optional[int] = None):
        self.initial_cash = initial_cash
        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> float:
        if isinstance(self.initial_cash, int):
            return float(self.initial_cash)
        else:
            return float(np.random.randint(self.initial_cash[0], self.initial_cash[1]))


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
    time_frames: Union[List[TimeFrame], TimeFrame] = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )

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


class FuturesOneStepEnv(EnvBase):
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
        self.config = config
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.leverage = config.leverage
        self.margin_type = config.margin_type
        self.maintenance_margin_rate = config.maintenance_margin_rate

        if not (0 <= config.transaction_fee <= 1):
            raise ValueError("Transaction fee must be between 0 and 1.")
        if not (0 <= config.slippage <= 1):
            raise ValueError("Slippage must be between 0 and 1.")
        if not (1 <= config.leverage <= 125):
            raise ValueError("Leverage must be between 1 and 125.")

        self.initial_cash_sampler = InitialBalanceSampler(config.initial_cash, config.seed)
        self.episode_idx = 0

        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=config.max_traj_length,
            seed=config.seed
        )
        self.random_start = True
        self.max_traj_length = config.max_traj_length

        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value
        self.periods_per_year = compute_periods_per_year_crypto(self.execute_on_unit, self.execute_on_value)

        # Reset settings
        self.initial_cash = config.initial_cash
        self.position_hold_counter = 0

        # SLTP levels
        self.stoploss_levels = list(config.stoploss_levels) if not isinstance(config.stoploss_levels, list) else config.stoploss_levels
        self.takeprofit_levels = list(config.takeprofit_levels) if not isinstance(config.takeprofit_levels, list) else config.takeprofit_levels

        # Create action map
        self.action_map = futures_onestep_action_map(self.stoploss_levels, self.takeprofit_levels)
        self.action_spec = Categorical(len(self.action_map))

        # Get market data keys
        market_data_keys = self.sampler.get_observation_keys()
        num_features = len(self.sampler.get_feature_keys())

        # Observation space
        self.observation_spec = CompositeSpec(shape=())

        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (10 elements for futures)
        account_state_spec = Bounded(
            low=-torch.inf, high=torch.inf, shape=(10,), dtype=torch.float
        )

        self.market_data_keys = []
        window_sizes_list = (
            config.window_sizes
            if isinstance(config.window_sizes, list)
            else [config.window_sizes]
        )
        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = f"market_data_{market_data_name}_{window_sizes_list[i]}"
            market_data_spec = Bounded(
                low=-torch.inf, high=torch.inf,
                shape=(window_sizes_list[i], num_features), dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)
        self.max_steps = self.sampler.get_max_steps()
        self.step_counter = 0

        # History tracking
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []

        # PERF: Pre-allocate account state tensor to avoid allocation each step
        self._account_state_buffer = torch.zeros(10, dtype=torch.float)

        # PERF: Initialize portfolio value cache
        self._cached_portfolio_value = 0.0
        self._cached_portfolio_price = 0.0

        super().__init__()

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
        if self.position_size == 0:
            return False

        if self.position_size > 0:
            # Long position - liquidated if price below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position - liquidated if price above liquidation price
            return current_price >= self.liquidation_price

    def _get_observation(self, initial: bool = False) -> TensorDictBase:
        """Get the current observation state."""
        if initial or self.position_size == 0:
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
        self.position_value = abs(self.position_size * current_price)

        # Calculate unrealized PnL
        if self.position_size != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(
                self.entry_price, current_price, self.position_size
            )
            self.unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(
                self.entry_price, current_price, self.position_size
            )
        else:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0

        # Calculate margin ratio
        total_balance = self.balance + self.unrealized_pnl
        margin_ratio = self.position_value / total_balance if total_balance > 0 else 0.0

        # PERF: Cache portfolio value and price to avoid recalculation in _get_portfolio_value()
        self._cached_portfolio_value = total_balance
        self._cached_portfolio_price = current_price

        # PERF: Update pre-allocated account state buffer in-place
        self._account_state_buffer[0] = self.balance
        self._account_state_buffer[1] = self.position_size  # Positive=long, Negative=short
        self._account_state_buffer[2] = self.position_value
        self._account_state_buffer[3] = self.entry_price
        self._account_state_buffer[4] = current_price
        self._account_state_buffer[5] = self.unrealized_pnl_pct
        self._account_state_buffer[6] = float(self.leverage)
        self._account_state_buffer[7] = margin_ratio
        self._account_state_buffer[8] = self.liquidation_price
        self._account_state_buffer[9] = float(self.position_hold_counter)

        # Clone to avoid issues with in-place modifications downstream
        obs_data = {self.account_state_key: self._account_state_buffer.clone()}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))
        return TensorDict(obs_data, batch_size=())

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action_tuple: tuple,
        trade_info: Dict,
    ) -> float:
        """
        Calculate reward using Sharpe ratio of rollout returns.

        Similar to LongOnlyOneStepEnv but accounting for futures positions.
        PERF: rollout_returns is now a list of floats, converted to tensor once here.
        """
        if len(self.rollout_returns) == 0 or action_tuple[0] is None:
            return 0.0

        # PERF: Convert list of floats to tensor once (instead of per-step tensor creation)
        returns = torch.tensor(self.rollout_returns, dtype=torch.float)

        # Need at least 2 points for a valid standard deviation
        if returns.numel() < 2:
            return float(returns.sum())

        mean_return = returns.mean()
        std_return = returns.std()

        # Compute Sharpe ratio
        sharpe = (mean_return / (std_return + 1e-9)) * np.sqrt(self.periods_per_year)

        # Add penalty for liquidation
        if trade_info.get("liquidated", False):
            sharpe = sharpe - 2.0  # Significant penalty

        # Clip to avoid extreme gradients
        return torch.clamp(sharpe, -10.0, 10.0).item()

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
            self.entry_price, current_price, self.position_size
        )
        return self.balance + unrealized_pnl

    def _set_seed(self, seed: int):
        """Set the seed for the environment."""
        self.seed = seed

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []

        max_episode_steps = self.sampler.reset(random_start=self.random_start)
        self.max_traj_length = max_episode_steps

        initial_portfolio_value = self.initial_cash_sampler.sample()
        self.balance = initial_portfolio_value
        self.initial_portfolio_value = initial_portfolio_value

        # Reset position state
        self.position_hold_counter = 0
        self.current_position = 0  # -1=short, 0=none, 1=long
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        self.step_counter = 0

        # SLTP levels
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.previous_portfolio_value = 0.0

        logger.debug(f"Reset environment with initial portfolio value: {initial_portfolio_value}")

        return self._get_observation(initial=True)

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

        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action_tuple, trade_info)

        if self.truncated:
            logger.debug(f"Episode truncated after {self.step_counter} steps")
            reward = 0  # No reward for truncated episodes

        self.reward_history.append(reward)

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
            self.entry_price, execution_price, self.position_size
        )

        # Calculate fee
        notional = abs(self.position_size * execution_price)
        fee_paid = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee_paid

        # Reset position
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.liquidation_price = 0.0
        self.current_position = 0
        self.position_hold_counter = 0
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
        if self.position_size > 0:
            loss = (self.liquidation_price - self.entry_price) * self.position_size
        else:
            loss = (self.entry_price - self.liquidation_price) * abs(self.position_size)

        # Apply loss and fees
        liquidation_fee = abs(self.position_size * self.liquidation_price) * self.transaction_fee
        self.balance += loss - liquidation_fee

        # Reset position
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.liquidation_price = 0.0
        self.current_position = 0
        self.position_hold_counter = 0
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
        is_long = self.position_size > 0
        is_short = self.position_size < 0

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
            self.position_size = position_qty
            self.current_position = 1
            # Long: SL below entry, TP above entry
            self.stop_loss_price = execution_price * (1 + sl_pct)  # sl_pct is negative
            self.take_profit_price = execution_price * (1 + tp_pct)
        else:  # short
            self.position_size = -position_qty
            self.current_position = -1
            # Short: SL above entry, TP below entry
            self.stop_loss_price = execution_price * (1 - sl_pct)  # sl_pct is negative, so this raises the price
            self.take_profit_price = execution_price * (1 - tp_pct)  # tp_pct is positive, so this lowers the price

        self.entry_price = execution_price
        self.position_value = notional_value
        self.liquidation_price = self._calculate_liquidation_price(execution_price, self.position_size)
        self.position_hold_counter = 0
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
