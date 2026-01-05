"""Sequential Futures Environment with Stop-Loss/Take-Profit support.

This environment combines:
- Long/short positions with leverage from SeqFuturesEnv
- Stop-loss/take-profit bracket orders from SeqLongOnlySLTPEnv
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from itertools import product
from enum import Enum
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Bounded
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase

from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


class MarginType(Enum):
    """Margin type for futures trading."""
    ISOLATED = "isolated"
    CROSSED = "crossed"


def futures_sltp_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
) -> Dict[int, Tuple[Optional[str], Optional[float], Optional[float]]]:
    """
    Create action map for futures SLTP environment.

    Action space:
    - 0: HOLD/Close (no new position)
    - 1 to N: Long positions with (SL, TP) combinations
    - N+1 to 2N: Short positions with (SL, TP) combinations

    Returns:
        Dict mapping action index to (side, sl_pct, tp_pct)
        where side is None, "long", or "short"
    """
    action_map = {}
    # 0 = HOLD/Close
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


@dataclass
class SeqFuturesSLTPEnvConfig:
    """Configuration for Sequential Futures SLTP Environment.

    This environment supports:
    - Long and short positions
    - Configurable leverage (1x - 125x)
    - Stop-loss and take-profit bracket orders
    - Liquidation mechanics
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
    initial_cash: Union[Tuple[int, int], int] = (1000, 5000)

    # Leverage and margin settings
    leverage: int = 1  # 1x to 125x
    margin_type: MarginType = MarginType.ISOLATED
    maintenance_margin_rate: float = 0.004  # 0.4% maintenance margin

    # Stop-loss and take-profit levels (as percentage, e.g., -0.05 = -5%)
    stoploss_levels: Union[List[float], Tuple[float, ...]] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], Tuple[float, ...]] = (0.05, 0.1, 0.2)

    # Trading costs
    transaction_fee: float = 0.0004  # 0.04% typical futures fee
    slippage: float = 0.001  # 0.1% slippage

    # Risk management
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    max_position_size: float = 1.0  # Max position as fraction of balance

    # Environment settings
    seed: Optional[int] = 42
    include_base_features: bool = False
    max_traj_length: Optional[int] = None
    random_start: bool = True

    # Reward settings
    reward_scaling: float = 1.0


class SeqFuturesSLTPEnv(EnvBase):
    """
    Sequential Futures Environment with Stop-Loss/Take-Profit support.

    Combines the leverage and shorting capabilities of SeqFuturesEnv with
    the bracket order functionality of SeqLongOnlySLTPEnv.

    Action Space:
    - Action 0: Hold / Close position
    - Actions 1 to N: Long positions with (SL, TP) combinations
    - Actions N+1 to 2N: Short positions with (SL, TP) combinations

    Where N = num_sl_levels * num_tp_levels

    Account State (10 elements, matching SeqFuturesEnv):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]

    Position size is:
    - Positive for long positions
    - Negative for short positions
    - Zero for no position
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SeqFuturesSLTPEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SeqFuturesSLTPEnv.

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

        # Convert to lists if needed
        self.stoploss_levels = (
            list(config.stoploss_levels)
            if not isinstance(config.stoploss_levels, list)
            else config.stoploss_levels
        )
        self.takeprofit_levels = (
            list(config.takeprofit_levels)
            if not isinstance(config.takeprofit_levels, list)
            else config.takeprofit_levels
        )

        # Create action map
        self.action_map = futures_sltp_action_map(
            self.stoploss_levels, self.takeprofit_levels
        )
        # PERF: Convert action_map to tuple for O(1) indexed lookup (faster than dict hashing)
        self._action_tuple = tuple(self.action_map[i] for i in range(len(self.action_map)))

        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=config.max_traj_length,
        )
        self.random_start = config.random_start
        self.max_traj_length = config.max_traj_length
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value

        # Reset settings
        self.initial_cash = config.initial_cash
        self.position_hold_counter = 0

        # Define action spec
        self.action_spec = Categorical(len(self.action_map))

        # Get the number of features from the sampler
        market_data_keys = self.sampler.get_observation_keys()
        num_features = len(self.sampler.get_feature_keys())

        # Observation space
        self.observation_spec = CompositeSpec(shape=())

        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (10 elements for futures):
        # [cash, position_size, position_value, entry_price, current_price,
        #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
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

        # Add coverage tracking indices to observation spec (only when random_start=True)
        if self.random_start:
            from torchrl.data.tensor_specs import Unbounded
            # reset_index: tracks episode start position diversity
            self.observation_spec.set(
                "reset_index",
                Unbounded(shape=(), dtype=torch.long)
            )
            # state_index: tracks all timesteps visited during episodes
            self.observation_spec.set(
                "state_index",
                Unbounded(shape=(), dtype=torch.long)
            )

        self.reward_spec = Bounded(
            low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float
        )
        self.max_steps = self.sampler.get_max_steps()
        self.step_counter = 0

        # History tracking
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []
        self.position_history = []

        # PERF: Pre-allocate account state buffer to avoid tensor creation per step
        self._account_state_buffer = torch.zeros(10, dtype=torch.float32)

        # PERF: Initialize cached OHLCV (will be set on first observation)
        self._cached_ohlcv = None

        # PERF: Cache margin fractions to avoid division per liquidation calculation
        self._inv_leverage = 1.0 / self.leverage
        self._long_liq_factor = 1 - self._inv_leverage + self.maintenance_margin_rate
        self._short_liq_factor = 1 + self._inv_leverage - self.maintenance_margin_rate

        # PERF: Cache leverage as float to avoid float() call per step
        self._leverage_float = float(self.leverage)

        # PERF: Pre-allocate trade_info template to avoid dict creation per step
        self._trade_info_template = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
            "sltp_triggered": None,
        }

        super().__init__()

    def _calculate_liquidation_price(
        self, entry_price: float, position_size: float
    ) -> float:
        """
        Calculate liquidation price for a position.

        For ISOLATED margin:
        - Long: liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        - Short: liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        """
        if position_size == 0:
            return 0.0

        # PERF: Use cached factors instead of computing division per call
        if position_size > 0:
            # Long position - liquidated if price drops
            liquidation_price = entry_price * self._long_liq_factor
        else:
            # Short position - liquidated if price rises
            liquidation_price = entry_price * self._short_liq_factor

        return max(0, liquidation_price)

    def _calculate_margin_required(self, position_value: float) -> float:
        """Calculate initial margin required for a position."""
        return abs(position_value) / self.leverage

    def _calculate_unrealized_pnl(
        self, entry_price: float, current_price: float, position_size: float
    ) -> float:
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

    def _calculate_unrealized_pnl_pct(
        self, entry_price: float, current_price: float, position_size: float
    ) -> float:
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

    def _check_sltp_trigger(self, ohlcv) -> Optional[str]:
        """
        Check if stop-loss or take-profit should trigger.

        Args:
            ohlcv: OHLCV namedtuple with open, high, low, close, volume attributes

        Returns:
            "sl" if stop-loss triggered
            "tp" if take-profit triggered
            None if neither triggered
        """
        if self.position_size == 0:
            return None
        if self.stop_loss == 0.0 and self.take_profit == 0.0:
            return None

        # PERF: Use namedtuple attribute access instead of dict lookup
        open_price = ohlcv.open
        high_price = ohlcv.high
        low_price = ohlcv.low
        close_price = ohlcv.close

        if self.position_size > 0:
            # Long position
            # SL triggers when price drops below SL level
            if self.stop_loss > 0:
                if open_price <= self.stop_loss:
                    return "sl"
                if low_price <= self.stop_loss:
                    return "sl"
            # TP triggers when price rises above TP level
            if self.take_profit > 0:
                if open_price >= self.take_profit:
                    return "tp"
                if high_price >= self.take_profit:
                    return "tp"
            # Check close for final determination
            if self.stop_loss > 0 and close_price <= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and close_price >= self.take_profit:
                return "tp"
        else:
            # Short position
            # SL triggers when price rises above SL level
            if self.stop_loss > 0:
                if open_price >= self.stop_loss:
                    return "sl"
                if high_price >= self.stop_loss:
                    return "sl"
            # TP triggers when price drops below TP level
            if self.take_profit > 0:
                if open_price <= self.take_profit:
                    return "tp"
                if low_price <= self.take_profit:
                    return "tp"
            # Check close for final determination
            if self.stop_loss > 0 and close_price >= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and close_price <= self.take_profit:
                return "tp"

        return None

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # PERF: Use combined method to get observation and OHLCV in one call
        obs_dict, self.current_timestamp, self.truncated, self._cached_ohlcv = (
            self.sampler.get_sequential_observation_with_ohlcv()
        )

        # PERF: Use namedtuple attribute access instead of dict lookup
        current_price = self._cached_ohlcv.close

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

        # PERF: Update pre-allocated account state buffer in-place
        buf = self._account_state_buffer
        buf[0] = self.balance
        buf[1] = self.position_size  # Positive=long, Negative=short
        buf[2] = self.position_value
        buf[3] = self.entry_price
        buf[4] = current_price
        buf[5] = self.unrealized_pnl_pct
        buf[6] = self._leverage_float  # PERF: Use cached float instead of float() call
        buf[7] = margin_ratio
        buf[8] = self.liquidation_price
        buf[9] = float(self.position_hold_counter)

        obs_data = {self.account_state_key: buf.clone()}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))
        out_td = TensorDict(obs_data, batch_size=())

        return out_td

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: int,
        trade_info: Dict,
    ) -> float:
        """
        Calculate the step reward.

        Uses a hybrid approach:
        - Dense rewards based on portfolio return
        - Sparse terminal reward comparing to buy-and-hold
        """
        if old_portfolio_value == 0:
            return 0.0

        # Portfolio return
        portfolio_return = (
            (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        )

        # Dense reward
        dense_reward = portfolio_return

        # Transaction cost penalty
        if trade_info.get("fee_paid", 0) > 0:
            dense_reward -= trade_info["fee_paid"] / old_portfolio_value

        # Liquidation penalty
        if trade_info.get("liquidated", False):
            dense_reward -= 0.1  # Significant penalty for liquidation

        # Clip dense reward
        dense_reward = np.clip(dense_reward, -0.1, 0.1)

        # Terminal reward
        if self.step_counter >= self.max_traj_length - 1:
            buy_and_hold_value = (
                self.initial_portfolio_value / self.base_price_history[0]
            ) * self.base_price_history[-1]

            compare_value = max(self.initial_portfolio_value, buy_and_hold_value)
            if compare_value > 0:
                # Terminal reward as percentage (1.0 = 100% better than benchmark)
                terminal_reward = (new_portfolio_value - compare_value) / compare_value
                # Clip to [-5, 5] to prevent extreme values
                terminal_reward = np.clip(terminal_reward, -5.0, 5.0)
                return terminal_reward * self.config.reward_scaling

        return dense_reward * self.config.reward_scaling

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value including unrealized PnL."""
        # PERF: Use cached OHLCV instead of calling get_base_features
        current_price = self._cached_ohlcv.close
        unrealized_pnl = self._calculate_unrealized_pnl(
            self.entry_price, current_price, self.position_size
        )
        return self.balance + unrealized_pnl

    def _set_seed(self, seed: int):
        """Set the seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Reset history
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []
        self.position_history = []

        max_episode_steps = self.sampler.reset(random_start=self.random_start)
        self.max_traj_length = max_episode_steps

        # Initialize balance
        initial_portfolio_value = (
            self.initial_cash
            if isinstance(self.initial_cash, int)
            else random.randint(self.initial_cash[0], self.initial_cash[1])
        )
        self.balance = initial_portfolio_value
        self.initial_portfolio_value = initial_portfolio_value

        # Reset position state
        self.position_hold_counter = 0
        self.current_position = 0  # -1=short, 0=none, 1=long
        self.position_size = 0.0  # Negative for short, positive for long
        self.position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        self.step_counter = 0

        # Reset SLTP
        self.stop_loss = 0.0
        self.take_profit = 0.0

        obs = self._get_observation()

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            obs.set("reset_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))
            obs.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        return obs

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value()

        # Get action
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        # PERF: Use tuple for direct indexing instead of dict lookup
        action_tuple = self._action_tuple[action_idx]
        side, sl_pct, tp_pct = action_tuple

        # PERF: Use cached OHLCV from _get_observation() instead of calling get_base_features
        ohlcv = self._cached_ohlcv
        current_price = ohlcv.close

        # PERF: Use pre-allocated template copy instead of dict literal
        trade_info = self._trade_info_template.copy()

        # Priority order: Liquidation > SL/TP > New action

        # 1. Check for liquidation first
        if self._check_liquidation(current_price):
            trade_info = self._execute_liquidation(current_price)
        # 2. Check for SL/TP trigger
        elif self.position_size != 0:
            sltp_trigger = self._check_sltp_trigger(ohlcv)
            if sltp_trigger is not None:
                trade_info = self._execute_sltp_close(ohlcv, sltp_trigger)
        # 3. Execute action if no position closed
        if not trade_info["executed"]:
            trade_info = self._execute_action(side, sl_pct, tp_pct)

        # Record history
        trade_action = 0
        if trade_info["executed"]:
            if trade_info["side"] == "long":
                trade_action = 1
            elif trade_info["side"] == "short":
                trade_action = -1
        self.action_history.append(trade_action)
        self.base_price_history.append(current_price)
        self.portfolio_value_history.append(old_portfolio_value)
        self.position_history.append(self.position_size)

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, action_idx, trade_info
        )
        self.reward_history.append(reward)

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict

    def _execute_liquidation(self, current_price: float) -> Dict:
        """Execute forced liquidation of position."""
        trade_info = {
            "executed": True,
            "side": "liquidation",
            "fee_paid": 0.0,
            "liquidated": True,
            "sltp_triggered": None,
        }

        # Realize the loss at liquidation price
        if self.position_size > 0:
            # Long position liquidated
            loss = (self.liquidation_price - self.entry_price) * self.position_size
        else:
            # Short position liquidated
            loss = (self.entry_price - self.liquidation_price) * abs(self.position_size)

        # Apply loss and fees
        liquidation_fee = (
            abs(self.position_size * self.liquidation_price) * self.transaction_fee
        )
        self.balance += loss - liquidation_fee
        trade_info["fee_paid"] = liquidation_fee

        # Reset position
        self._reset_position()

        return trade_info

    def _execute_sltp_close(self, ohlcv: dict, trigger_type: str) -> Dict:
        """Execute SL/TP triggered close."""
        trade_info = {
            "executed": True,
            "side": "close",
            "fee_paid": 0.0,
            "liquidated": False,
            "sltp_triggered": trigger_type,
        }

        # Determine execution price based on trigger
        if trigger_type == "sl":
            execution_price = self.stop_loss
        else:  # tp
            execution_price = self.take_profit

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.entry_price, execution_price, self.position_size
        )

        # Calculate fee
        notional = abs(self.position_size * execution_price)
        fee = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee
        trade_info["fee_paid"] = fee

        # Reset position
        self._reset_position()

        return trade_info

    def _execute_action(
        self,
        side: Optional[str],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> Dict:
        """Execute the action."""
        trade_info = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
            "sltp_triggered": None,
        }

        # PERF: Use cached OHLCV instead of calling get_base_features
        current_price = self._cached_ohlcv.close

        # PERF: Use Python random instead of torch.empty().uniform_() to avoid tensor allocation
        price_noise = random.uniform(1 - self.slippage, 1 + self.slippage)

        if side is None:
            # Hold or close action
            if self.position_size != 0:
                # Close existing position
                trade_info = self._close_position(current_price, price_noise)
            # Update hold counter if still holding
            if self.position_size != 0:
                self.position_hold_counter += 1
            return trade_info

        # Opening a new position
        if side == "long":
            if self.position_size < 0:
                # Close short first
                self._close_position(current_price, price_noise)

            if self.position_size <= 0:
                # Open long position
                trade_info = self._open_position(
                    "long", current_price, price_noise, sl_pct, tp_pct
                )

        elif side == "short":
            if self.position_size > 0:
                # Close long first
                self._close_position(current_price, price_noise)

            if self.position_size >= 0:
                # Open short position
                trade_info = self._open_position(
                    "short", current_price, price_noise, sl_pct, tp_pct
                )

        # Update hold counter
        if self.position_size != 0:
            self.position_hold_counter += 1

        return trade_info

    def _open_position(
        self,
        side: str,
        current_price: float,
        price_noise: float,
        sl_pct: float,
        tp_pct: float,
    ) -> Dict:
        """Open a new position with SL/TP."""
        trade_info = {
            "executed": True,
            "side": side,
            "fee_paid": 0.0,
            "liquidated": False,
            "sltp_triggered": None,
        }

        execution_price = current_price * price_noise

        # Calculate position size based on available balance and leverage
        usable_balance = self.balance * self.config.max_position_size
        margin_plus_fee_rate = (1.0 / self.leverage) + self.transaction_fee
        # Apply 0.1% safety margin to avoid floating-point precision issues
        max_notional = usable_balance / margin_plus_fee_rate * 0.999
        notional_value = max_notional
        position_qty = notional_value / execution_price

        # Calculate margin required
        margin_required = self._calculate_margin_required(notional_value)

        # Calculate fee
        fee = notional_value * self.transaction_fee

        if margin_required + fee > self.balance:
            # Not enough balance
            trade_info["executed"] = False
            return trade_info

        # Execute trade
        self.balance -= fee
        trade_info["fee_paid"] = fee

        if side == "long":
            self.position_size = position_qty
            self.current_position = 1
            # For long: SL is below entry, TP is above entry
            self.stop_loss = execution_price * (1 + sl_pct)  # sl_pct is negative
            self.take_profit = execution_price * (1 + tp_pct)  # tp_pct is positive
        else:  # short
            self.position_size = -position_qty
            self.current_position = -1
            # For short: SL is above entry, TP is below entry
            self.stop_loss = execution_price * (1 - sl_pct)  # sl_pct is negative, so this goes up
            self.take_profit = execution_price * (1 - tp_pct)  # tp_pct is positive, so this goes down

        self.entry_price = execution_price
        self.position_value = notional_value
        self.liquidation_price = self._calculate_liquidation_price(
            execution_price, self.position_size
        )
        self.position_hold_counter = 0

        return trade_info

    def _close_position(self, current_price: float, price_noise: float) -> Dict:
        """Close existing position."""
        trade_info = {
            "executed": True,
            "side": "close",
            "fee_paid": 0.0,
            "liquidated": False,
            "sltp_triggered": None,
        }

        if self.position_size == 0:
            trade_info["executed"] = False
            return trade_info

        execution_price = current_price * price_noise

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.entry_price, execution_price, self.position_size
        )

        # Calculate fee
        notional = abs(self.position_size * execution_price)
        fee = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee
        trade_info["fee_paid"] = fee

        # Reset position
        self._reset_position()

        return trade_info

    def _reset_position(self):
        """Reset position state."""
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.liquidation_price = 0.0
        self.current_position = 0
        self.position_hold_counter = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = (
            self.config.bankrupt_threshold * self.initial_portfolio_value
        )
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
        portfolio_values = torch.tensor(self.portfolio_value_history, dtype=torch.float32)
        rewards = torch.tensor(self.reward_history, dtype=torch.float32)

        # Compute periods per year for annualization
        periods_per_year = compute_periods_per_year_crypto(
            self.execute_on_unit,
            self.execute_on_value
        )

        # Use shared metrics computation function
        return compute_all_metrics(
            portfolio_values=portfolio_values,
            rewards=rewards,
            action_history=self.action_history,
            periods_per_year=periods_per_year,
        )

    def render_history(self, return_fig=False):
        """Render the history of the environment."""
        price_history = self.base_price_history
        time_indices = list(range(len(price_history)))
        action_history = self.action_history
        portfolio_value_history = self.portfolio_value_history
        position_history = self.position_history

        # Calculate buy-and-hold balance
        initial_balance = portfolio_value_history[0]
        initial_price = price_history[0]
        units_held = initial_balance / initial_price
        buy_and_hold_balance = [units_held * price for price in price_history]

        # Create subplots
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 2, 1]}
        )
        ax1, ax2, ax3 = axes

        # Plot price history
        ax1.plot(
            time_indices, price_history,
            label='Price History', color='blue', linewidth=1.5
        )

        # Plot buy/sell actions
        long_indices = [i for i, action in enumerate(action_history) if action == 1]
        long_prices = [price_history[i] for i in long_indices]
        short_indices = [i for i, action in enumerate(action_history) if action == -1]
        short_prices = [price_history[i] for i in short_indices]

        ax1.scatter(
            long_indices, long_prices,
            marker='^', color='green', s=80, label='Long', zorder=5
        )
        ax1.scatter(
            short_indices, short_prices,
            marker='v', color='red', s=80, label='Short', zorder=5
        )

        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price History with Long/Short Actions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot portfolio value
        ax2.plot(
            time_indices, portfolio_value_history,
            label='Portfolio Value', color='green', linewidth=1.5
        )
        ax2.plot(
            time_indices, buy_and_hold_balance,
            label='Buy and Hold', color='purple', linestyle='--', linewidth=1.5
        )
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.set_title('Portfolio Value vs Buy and Hold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot position
        ax3.fill_between(
            time_indices, position_history, 0,
            where=[p > 0 for p in position_history], color='green', alpha=0.3, label='Long'
        )
        ax3.fill_between(
            time_indices, position_history, 0,
            where=[p < 0 for p in position_history], color='red', alpha=0.3, label='Short'
        )
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Time (Index)')
        ax3.set_ylabel('Position Size')
        ax3.set_title('Position History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    time_frames = [
        TimeFrame(1, TimeFrameUnit.Minute),
    ]
    window_sizes = [12]
    execute_on = TimeFrame(1, TimeFrameUnit.Minute)

    # Load sample data
    df = pd.read_csv(
        "/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/"
        "binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv"
    )

    config = SeqFuturesSLTPEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        leverage=10,
        stoploss_levels=[-0.02, -0.05],
        takeprofit_levels=[0.05, 0.1],
        transaction_fee=0.0004,
        slippage=0.001,
        bankrupt_threshold=0.1,
    )
    env = SeqFuturesSLTPEnv(df, config)

    print(f"Action space size: {env.action_spec.n}")
    print(f"Action map: {env.action_map}")

    td = env.reset()
    for i in range(env.max_steps):
        action = env.action_spec.sample()
        td.set("action", action)
        td = env.step(td)
        td = td["next"]
        portfolio_value = td["account_state"][0].item() + td["account_state"][2].item()
        print(f"Step {i}: Action={action.item()}, Portfolio=${portfolio_value:.2f}")
        if td["done"]:
            print(f"Episode done at step {i}")
            break

    env.render_history()
