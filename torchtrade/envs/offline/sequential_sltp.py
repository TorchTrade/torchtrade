"""Sequential Trading Environment with Stop-Loss/Take-Profit support.

A TorchRL-compatible environment for algorithmic trading with bracket orders.
Supports both spot and futures trading with stop-loss and take-profit levels.

Key Features:
    - Inherits from SequentialTradingEnv (6-element account state)
    - Stop-loss and take-profit bracket order support
    - Mode-aware action space:
        * Spot: HOLD + N long positions with SL/TP combinations
        * Futures: HOLD + N long + N short positions with SL/TP combinations
    - Intrabar SL/TP trigger detection using OHLC data
    - Automatic position exit on trigger
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable

import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.offline.sequential import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
)
from torchtrade.envs.core.state import binarize_action_type
from torchtrade.envs.utils.sltp_helpers import (
    calculate_long_bracket_prices,
    calculate_short_bracket_prices,
)
from torchtrade.envs.utils.action_maps import create_sltp_action_map


@dataclass
class SequentialTradingEnvSLTPConfig(SequentialTradingEnvConfig):
    """Configuration for sequential trading environment with SLTP support.

    Extends SequentialTradingEnvConfig with bracket order parameters.
    """
    # Stop-loss and take-profit levels (as percentages)
    stoploss_levels: Union[List[float], Tuple[float, ...]] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], Tuple[float, ...]] = (0.05, 0.1, 0.2)

    # Action space options
    include_hold_action: bool = True  # Include HOLD action (index 0)
    include_close_action: bool = False  # Include CLOSE action (default: False for SLTP)

    def __post_init__(self):
        """Validate configuration after dataclass initialization."""
        # Call parent post_init first
        super().__post_init__()

        # Convert to lists if needed
        if not isinstance(self.stoploss_levels, list):
            self.stoploss_levels = list(self.stoploss_levels)
        if not isinstance(self.takeprofit_levels, list):
            self.takeprofit_levels = list(self.takeprofit_levels)

        # Validate SL/TP levels
        for sl in self.stoploss_levels:
            if sl >= 0:
                raise ValueError(
                    f"Stop-loss levels must be negative (e.g., -0.05 for 5% loss), got {sl}"
                )
        for tp in self.takeprofit_levels:
            if tp <= 0:
                raise ValueError(
                    f"Take-profit levels must be positive (e.g., 0.1 for 10% profit), got {tp}"
                )


class SequentialTradingEnvSLTP(SequentialTradingEnv):
    """Sequential trading environment with stop-loss/take-profit support.

    Combines the base SequentialTradingEnv with bracket order functionality.

    Action Space Structure:
    -----------------------
    Spot Mode (3 SL levels × 3 TP levels = 10 actions):
        - Action 0: HOLD (optional, controlled by include_hold_action)
        - Actions 1-9: Long positions with (SL, TP) combinations
            * (-0.025, 0.05), (-0.025, 0.1), (-0.025, 0.2)  # SL=-2.5%
            * (-0.05, 0.05), (-0.05, 0.1), (-0.05, 0.2)     # SL=-5%
            * (-0.1, 0.05), (-0.1, 0.1), (-0.1, 0.2)        # SL=-10%

    Futures Mode (3 SL × 3 TP × 2 directions = 19 actions):
        - Action 0: HOLD (optional)
        - Actions 1-9: Long positions with (SL, TP) combinations
        - Actions 10-18: Short positions with (SL, TP) combinations

    SL/TP Trigger Behavior:
    -----------------------
    Long positions:
        - Stop-loss triggers when price <= SL level (loss protection)
        - Take-profit triggers when price >= TP level (profit taking)

    Short positions:
        - Stop-loss triggers when price >= SL level (loss protection)
        - Take-profit triggers when price <= TP level (profit taking)

    Universal Account State (inherited from SequentialTradingEnv):
    ---------------------------------------------------------------
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    See SequentialTradingEnv docstring for detailed element descriptions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SequentialTradingEnvSLTPConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
    ):
        """Initialize the sequential SLTP environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration with SLTP parameters
            feature_preprocessing_fn: Optional function to preprocess features
            reward_function: Optional reward function (default: log_return_reward)
        """
        # Store SLTP configuration before initializing parent
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels
        self.include_hold_action = config.include_hold_action
        self.include_close_action = config.include_close_action

        # Build action map for SLTP
        self.action_map = self._build_action_map(
            config.stoploss_levels,
            config.takeprofit_levels,
            config.include_hold_action,
            config.include_close_action,
            allow_short=(config.leverage > 1),  # Futures mode: leverage > 1
        )

        # Temporarily set action_levels to a dummy list for parent initialization
        # This prevents the parent from failing when creating action_spec
        original_action_levels = config.action_levels
        config.action_levels = [0.0]  # Dummy value, will be overridden

        # Initialize parent class (sets up base SequentialTradingEnv)
        super().__init__(df, config, feature_preprocessing_fn, reward_function)

        # Restore original action_levels
        config.action_levels = original_action_levels

        # Override action spec with SLTP action space
        self.action_spec = Categorical(len(self.action_map))

        # Initialize SLTP-specific state
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Convert action_map to tuple for O(1) indexed lookup (performance optimization)
        self._action_tuple = tuple(self.action_map[i] for i in range(len(self.action_map)))

    def _build_action_map(
        self,
        stoploss_levels: List[float],
        takeprofit_levels: List[float],
        include_hold_action: bool,
        include_close_action: bool,
        allow_short: bool,
    ) -> Dict[int, Tuple]:
        """Build action map for SLTP environment.

        Delegates to the shared create_sltp_action_map to ensure consistent
        SL/TP swapping for short positions across live and offline environments.
        """
        return create_sltp_action_map(
            stoploss_levels=stoploss_levels,
            takeprofit_levels=takeprofit_levels,
            include_short_positions=allow_short,
            include_hold_action=include_hold_action,
            include_close_action=include_close_action,
        )

    def _reset_position_state(self):
        """Reset position tracking state including SLTP-specific state."""
        super()._reset_position_state()
        # Reset SLTP state
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def _check_sltp_trigger(self, ohlcv: dict) -> Optional[str]:
        """Check if stop-loss or take-profit should trigger.

        Uses intrabar OHLC data to detect SL/TP triggers that may occur
        within the candle, not just at the close.

        Args:
            ohlcv: Dictionary with keys "open", "high", "low", "close", "volume"

        Returns:
            "sl" if stop-loss triggered
            "tp" if take-profit triggered
            None if neither triggered
        """
        if self.position.position_size == 0:
            return None
        if self.stop_loss == 0.0 and self.take_profit == 0.0:
            return None

        open_price = ohlcv["open"]
        high_price = ohlcv["high"]
        low_price = ohlcv["low"]
        close_price = ohlcv["close"]

        if self.position.position_size > 0:
            # Long position
            # SL triggers when price drops below SL level
            if self.stop_loss > 0:
                # Check open first (immediate trigger at bar open)
                if open_price <= self.stop_loss:
                    return "sl"
                # Check if low touched SL (triggered intrabar)
                if low_price <= self.stop_loss:
                    return "sl"

            # TP triggers when price rises above TP level
            if self.take_profit > 0:
                # Check open first
                if open_price >= self.take_profit:
                    return "tp"
                # Check if high touched TP (triggered intrabar)
                if high_price >= self.take_profit:
                    return "tp"

            # Final check at close price
            if self.stop_loss > 0 and close_price <= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and close_price >= self.take_profit:
                return "tp"

        else:
            # Short position
            # SL triggers when price rises above SL level
            if self.stop_loss > 0:
                # Check open first
                if open_price >= self.stop_loss:
                    return "sl"
                # Check if high touched SL (triggered intrabar)
                if high_price >= self.stop_loss:
                    return "sl"

            # TP triggers when price drops below TP level
            if self.take_profit > 0:
                # Check open first
                if open_price <= self.take_profit:
                    return "tp"
                # Check if low touched TP (triggered intrabar)
                if low_price <= self.take_profit:
                    return "tp"

            # Final check at close price
            if self.stop_loss > 0 and close_price >= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and close_price <= self.take_profit:
                return "tp"

        return None

    def _execute_sltp_close(self, execution_price: float, trigger_type: str) -> Dict:
        """Execute SL/TP triggered close.

        Args:
            execution_price: Price at which SL/TP triggered
            trigger_type: Either "sl" or "tp"

        Returns:
            Trade info dictionary with execution details
        """
        if self.position.position_size == 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, execution_price, self.position.position_size
        )

        # Calculate fee and margin to return
        notional = abs(self.position.position_size * execution_price)
        fee = notional * self.transaction_fee
        # Return the margin that was locked when opening
        margin_to_return = abs(self.position.position_size * self.position.entry_price) / self.leverage

        # Update balance: add realized PnL, subtract fee, return locked margin
        self.balance += pnl - fee + margin_to_return
        self.balance = max(0.0, self.balance)

        # Reset position and SLTP state
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.position.current_position = 0
        self.position.hold_counter = 0
        self.liquidation_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        return {
            "executed": True,
            "side": f"sltp_{trigger_type}",  # "sltp_sl" or "sltp_tp"
            "fee_paid": fee,
            "liquidated": False
        }

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step with SLTP logic.

        Priority order:
        1. Liquidation check (futures only, highest priority)
        2. SL/TP trigger check (bracket orders)
        3. New action execution

        This ordering ensures:
        - Liquidation takes precedence over everything
        - Bracket orders execute before new actions
        - New actions only execute if no position closed
        """
        self.step_counter += 1

        # Cache base features and get current price
        cached_base = self._cached_base_features
        cached_price = cached_base["close"]

        # Get desired action
        action_idx = tensordict["action"]
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self._action_tuple[action_idx]
        side, sl_pct, tp_pct = action_tuple

        # Priority 1: Check for liquidation (futures only)
        if self._check_liquidation(cached_price):
            trade_info = self._execute_liquidation()

        # Priority 2: Check for SL/TP trigger (if not liquidated)
        elif self.position.position_size != 0:
            sltp_trigger = self._check_sltp_trigger(cached_base)
            if sltp_trigger is not None:
                # Determine execution price based on trigger type
                if sltp_trigger == "sl":
                    execution_price = self.stop_loss
                else:  # "tp"
                    execution_price = self.take_profit
                trade_info = self._execute_sltp_close(execution_price, sltp_trigger)
            else:
                # No trigger, execute new action
                trade_info = self._execute_sltp_action(side, sl_pct, tp_pct, cached_price)

        # Priority 3: Execute new action (if no liquidation or trigger)
        else:
            trade_info = self._execute_sltp_action(side, sl_pct, tp_pct, cached_price)

        # Update position flag based on actual position size
        if trade_info["executed"]:
            if self.position.position_size > 0:
                self.position.current_position = 1  # Long
            elif self.position.position_size < 0:
                self.position.current_position = -1  # Short
            else:
                self.position.current_position = 0  # Flat

        # Get updated state (advances timestamp and caches new base features)
        next_tensordict = self._get_observation()
        new_price = self._cached_base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            next_tensordict.set("reset_index", torch.tensor(self._reset_idx, dtype=torch.long))
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

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

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict

    def _execute_sltp_action(
        self, side: Optional[str], sl_pct: Optional[float], tp_pct: Optional[float], base_price: float
    ) -> Dict:
        """Execute action with SLTP bracket order setup.

        Args:
            side: Position side ("long", "short", "close", or None for hold)
            sl_pct: Stop-loss percentage (negative)
            tp_pct: Take-profit percentage (positive)
            base_price: Base price for execution

        Returns:
            Trade info dictionary
        """
        # HOLD action
        if side is None:
            if self.position.position_size != 0:
                self.position.hold_counter += 1
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # CLOSE action
        if side == "close":
            if self.position.position_size != 0:
                # Apply slippage
                price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
                execution_price = base_price * price_noise
                return self._close_position(execution_price)
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Opening new position (long or short)
        # Apply slippage
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = base_price * price_noise

        # Check if already in same direction - if so, hold (ignore duplicate action)
        if side == "long" and self.position.position_size > 0:
            self.position.hold_counter += 1
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}
        if side == "short" and self.position.position_size < 0:
            self.position.hold_counter += 1
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # If switching direction, close existing position first
        if self.position.position_size != 0:
            self._close_position(execution_price)

        # Open new position with SLTP brackets
        return self._open_position_with_sltp(side, execution_price, sl_pct, tp_pct)

    def _open_position_with_sltp(
        self, side: str, execution_price: float, sl_pct: float, tp_pct: float
    ) -> Dict:
        """Open a new position with SL/TP bracket orders.

        Args:
            side: Position side ("long" or "short")
            execution_price: Price at which to execute (includes slippage)
            sl_pct: Stop-loss percentage
            tp_pct: Take-profit percentage

        Returns:
            Trade info dictionary
        """
        # Use fractional sizing from parent class
        # For spot mode, action_value represents fraction of portfolio to invest
        # For futures mode, action_value represents leverage-adjusted position
        if self.leverage == 1:
            # Spot: invest 100% of portfolio (all-in strategy for SLTP)
            action_value = 1.0
        else:
            # Futures: use leverage for position sizing
            action_value = 1.0 if side == "long" else -1.0

        # Calculate position size using parent's fractional logic
        from torchtrade.envs.utils.fractional_sizing import (
            calculate_fractional_position,
            PositionCalculationParams,
        )

        portfolio_value = self._get_portfolio_value(execution_price)
        params = PositionCalculationParams(
            balance=portfolio_value,
            action_value=action_value,
            current_price=execution_price,
            leverage=self.leverage,
            transaction_fee=self.transaction_fee,
            allow_short=(side == "short")
        )
        position_size, notional_value, calc_side = calculate_fractional_position(params)

        # Calculate margin and fee
        margin_required = notional_value / self.leverage
        fee = abs(notional_value) * self.transaction_fee

        # Check if sufficient balance
        if margin_required + fee > self.balance:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Deduct fee and margin
        # For spot (leverage=1): margin_required = notional_value (full cost)
        # For futures (leverage>1): margin_required = notional_value / leverage
        self.balance -= fee + margin_required
        self.balance = max(0.0, self.balance)

        # Set position
        self.position.position_size = position_size if side == "long" else -abs(position_size)
        self.position.position_value = abs(notional_value)
        self.position.entry_price = execution_price
        self.position.hold_counter = 0

        # Set position direction
        if self.leverage == 1:
            self.position.current_position = 1  # Always long
        else:
            self.position.current_position = 1 if side == "long" else -1

        # Calculate liquidation price (futures only)
        self.liquidation_price = self._calculate_liquidation_price(execution_price, self.position.position_size)

        # Set SLTP brackets using helper functions
        if side == "long":
            self.stop_loss, self.take_profit = calculate_long_bracket_prices(
                execution_price, sl_pct, tp_pct
            )
        else:  # short
            self.stop_loss, self.take_profit = calculate_short_bracket_prices(
                execution_price, sl_pct, tp_pct
            )

        return {"executed": True, "side": side, "fee_paid": fee, "liquidated": False}

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
    config = SequentialTradingEnvSLTPConfig(
        execute_on="1Hour",
        time_frames="1Hour",
        initial_cash=10000,
        transaction_fee=0.0,
        slippage=0.0,
        stoploss_levels=(-0.025, -0.05, -0.1),
        takeprofit_levels=(0.05, 0.1, 0.2),
        include_hold_action=True,
    )
    env = SequentialTradingEnvSLTP(df, config)
    print(f"Environment created")
    print(f"Action map size: {len(env.action_map)}")
    print(f"Action space size: {env.action_spec.n}")
    print(f"Sample actions:")
    for i in range(min(5, len(env.action_map))):
        print(f"  {i}: {env.action_map[i]}")
    print()

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

        # Random action (mostly hold, occasionally take position)
        action = np.random.choice(
            range(len(env.action_map)),
            p=[0.95] + [0.05 / (len(env.action_map) - 1)] * (len(env.action_map) - 1)
        )
        print(f"Action: {action} {env.action_map[action]}\n")

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
