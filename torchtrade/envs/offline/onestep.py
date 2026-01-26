"""One-Step Trading Environment for GRPO/Contextual Bandit Training.

A TorchRL-compatible environment for algorithmic trading with one-step rollouts.
Designed for GRPO and contextual bandit approaches where the agent makes a single
decision per episode, then the environment simulates until a terminal condition.

Key Features:
    - Inherits from SequentialTradingEnvSLTP (SLTP bracket orders)
    - One decision per episode (contextual bandit pattern)
    - Internal rollout simulation until SL/TP trigger or truncation
    - Returns terminal reward only (sum of step-wise returns)
    - Mode-aware action space and rollout logic
    - Supports both spot and futures trading
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable, Literal
import math
import warnings

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase

from torchtrade.envs.offline.sequential_sltp import (
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
)


@dataclass
class OneStepTradingEnvConfig(SequentialTradingEnvSLTPConfig):
    """Configuration for one-step trading environment.

    Extends SequentialTradingEnvSLTPConfig with one-step specific parameters.

    The one-step environment is designed for GRPO training where:
    - Agent makes one decision per episode
    - Environment internally rolls out until SL/TP trigger or truncation
    - Only terminal reward matters (accumulated over the rollout)
    """
    # Override: Force random_start to True (required for contextual bandit setting)
    random_start: bool = True

    def __post_init__(self):
        """Validate configuration after dataclass initialization."""
        # Call parent post_init first
        super().__post_init__()

        # Force random_start for one-step environments
        if not self.random_start:
            warnings.warn(
                "OneStepTradingEnv requires random_start=True for proper contextual bandit training. "
                "Forcing random_start=True."
            )
            self.random_start = True


class OneStepTradingEnv(SequentialTradingEnvSLTP):
    """One-step trading environment for GRPO/contextual bandit training.

    This environment supports both spot and futures trading with a one-step rollout pattern:

    Episode Flow:
    -------------
    1. Agent observes initial market state
    2. Agent selects action (HOLD, or open position with SL/TP levels)
    3. Environment internally simulates position until terminal condition:
       - Stop-loss trigger (exit with loss)
       - Take-profit trigger (exit with profit)
       - Liquidation trigger (futures only, forced exit)
       - Episode truncation (max steps reached)
    4. Environment returns done=True with terminal reward

    Rollout Logic (Mode-Aware):
    ---------------------------
    Spot Mode:
        - Only simulates long positions (no shorts)
        - No liquidation checks
        - SL trigger: price <= stop_loss_price
        - TP trigger: price >= take_profit_price

    Futures Mode:
        - Simulates both long and short positions
        - Checks liquidation first (highest priority)
        - Long positions:
            * Liquidation: price <= liquidation_price
            * SL trigger: price <= stop_loss_price
            * TP trigger: price >= take_profit_price
        - Short positions:
            * Liquidation: price >= liquidation_price
            * SL trigger: price >= stop_loss_price
            * TP trigger: price <= take_profit_price

    Terminal Reward:
    ----------------
    The environment accumulates step-wise returns during the rollout:
        terminal_reward = sum(log(portfolio_value[t] / portfolio_value[t-1]))

    This sum is stored in `self.rollout_returns` and can be accessed by
    custom reward functions via the reward context.

    Action Space (inherited from SequentialTradingEnvSLTP):
    -------------------------------------------------------
    Spot Mode (3 SL × 3 TP = 10 actions):
        - Action 0: HOLD (optional, controlled by include_hold_action)
        - Actions 1-9: Long positions with (SL, TP) combinations

    Futures Mode (3 SL × 3 TP × 2 directions = 19 actions):
        - Action 0: HOLD (optional)
        - Actions 1-9: Long positions with (SL, TP) combinations
        - Actions 10-18: Short positions with (SL, TP) combinations

    Universal Account State (inherited from SequentialTradingEnv):
    --------------------------------------------------------------
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    See SequentialTradingEnv docstring for detailed element descriptions.

    Usage Context:
    --------------
    This environment is designed for GRPO (Group Relative Policy Optimization)
    and other contextual bandit approaches where:
    - Each episode is a single decision
    - The environment handles trade execution and exit logic
    - Only the terminal outcome matters for policy gradient estimation
    - Random starting points ensure diverse state coverage
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: OneStepTradingEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        """Initialize the one-step trading environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration with one-step parameters
            feature_preprocessing_fn: Optional function to preprocess features
        """
        # Initialize parent class (SequentialTradingEnvSLTP)
        super().__init__(df, config, feature_preprocessing_fn)

        # Initialize one-step specific state
        self.previous_portfolio_value = 0.0
        self.rollout_returns = []
        self.episode_idx = 0

        # Force random_start for contextual bandit setting
        if not config.random_start:
            warnings.warn(
                "OneStepTradingEnv requires random_start=True. Forcing random_start=True."
            )
        self.random_start = True

    def _reset_position_state(self):
        """Reset position tracking state including one-step specific state."""
        super()._reset_position_state()
        # Reset one-step specific state
        self.previous_portfolio_value = 0.0
        self.rollout_returns = []

    def _get_observation(self, initial: bool = False) -> TensorDictBase:
        """Get the current observation state.

        This method is overridden to handle the one-step rollout pattern.

        Args:
            initial: Whether this is the initial observation (reset).
                    If True or position is 0, gets new observation.
                    Otherwise, performs rollout to SL/TP trigger.

        Returns:
            TensorDict with observation data
        """
        # Get market data
        if initial or self.position.position_size == 0:
            # Initial observation or no position - get new observation
            obs_dict = self._get_observation_scaffold()
            self.rollout_returns = []
        else:
            # Position exists - rollout until SL/TP trigger or truncation
            # _rollout() updates _cached_base_features internally
            trade_info, obs_dict = self._rollout()

        # Use cached base features (avoids redundant get_base_features calls)
        current_price = self._cached_base_features["close"]

        # Calculate position value
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
        exposure_pct = self._calculate_exposure_pct(self.position.position_value, portfolio_value)

        # Calculate position direction (-1, 0, +1)
        position_direction = float(
            1 if self.position.position_size > 0
            else -1 if self.position.position_size < 0
            else 0
        )

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

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step (always returns done=True for one-step).

        The one-step environment workflow:
        1. Execute action (if not HOLD)
        2. Get updated observation (which triggers rollout if position opened)
        3. Calculate terminal reward (accumulated over rollout)
        4. Return done=True (single decision per episode)

        Args:
            tensordict: Input TensorDict containing "action" key

        Returns:
            TensorDict with "reward", "done", "truncated", "terminated" keys
        """
        self.step_counter += 1

        # Cache base features and get current price
        cached_base = self._cached_base_features
        cached_price = cached_base["close"]

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get desired action
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self._action_tuple[action_idx]
        side, sl_pct, tp_pct = action_tuple

        # Initialize trade info
        trade_info = {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Execute action (only HOLD or open position - no closing in one-step)
        # Priority 1: Check for liquidation (futures only)
        if self._check_liquidation(cached_price):
            trade_info = self._execute_liquidation(cached_price)
        # Priority 2: Execute new action
        else:
            trade_info = self._execute_sltp_action(side, sl_pct, tp_pct, cached_price)

        # Update position flag
        if trade_info["executed"]:
            if self.trading_mode == "spot":
                # Spot: only in position if we opened long
                self.position.current_position = 1 if side == "long" else 0
            else:
                # Futures: track direction (1=long, -1=short, 0=flat)
                if side == "long":
                    self.position.current_position = 1
                elif side == "short":
                    self.position.current_position = -1
                elif side in ("close", "sltp_sl", "sltp_tp", "liquidation"):
                    self.position.current_position = 0

        # Get updated state (advances timestamp and triggers rollout if position opened)
        # This is where the magic happens - _get_observation() will call _rollout()
        # if a position was opened, simulating until SL/TP trigger or truncation
        next_tensordict = self._get_observation()
        new_price = self._cached_base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Calculate terminal reward (accumulated over rollout)
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, action_tuple, trade_info
        )

        # If truncated without trigger, set reward to 0 (no reward for incomplete episodes)
        if self.truncated and not trade_info.get("executed"):
            reward = 0.0

        # Determine action_type for history tracking
        action_type = trade_info.get("side") or "hold"
        from torchtrade.envs.core.state import binarize_action_type

        if self.trading_mode == "spot":
            # Spot mode: convert side to binarized action
            if action_type == "long":
                binarized_action = 1
            elif action_type in ("close", "sltp_sl", "sltp_tp"):
                binarized_action = -1
            else:
                binarized_action = 0
        else:
            # Futures mode: use binarize_action_type utility
            binarized_action = binarize_action_type(action_type)

        # Record step history (minimal for one-step - just initial state)
        self.history.record_step(
            price=cached_price,
            action=binarized_action,
            reward=reward,
            portfolio_value=old_portfolio_value,
            position=self.position.position_size,
            action_type=action_type
        )

        # Check termination (always done for one-step environments)
        done = True  # One-step environment - always done after single action

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", done)

        return next_tensordict

    def compute_return(self, close_price: float) -> float:
        """Compute log return for current rollout step.

        This is called during internal rollout to accumulate step-wise returns.
        The sum of these returns forms the terminal reward.

        Args:
            close_price: Current market price

        Returns:
            Log return for this step
        """
        current_value = self._get_portfolio_value(close_price)

        # Initialize previous_portfolio_value if this is the first rollout step
        if self.previous_portfolio_value == 0.0:
            self.previous_portfolio_value = current_value
            return 0.0

        # Calculate log return
        if current_value > 0 and self.previous_portfolio_value > 0:
            log_return = math.log(current_value / self.previous_portfolio_value)
        else:
            log_return = 0.0

        self.previous_portfolio_value = current_value
        return log_return

    def _rollout(self) -> Tuple[Dict, dict]:
        """Simulate position until SL/TP trigger, liquidation, or truncation.

        This is the core of the one-step environment. After opening a position,
        this method simulates time forward until a terminal condition occurs.

        Mode-Aware Rollout Logic:
        -------------------------
        Spot Mode:
            1. Check SL trigger (price <= stop_loss)
            2. Check TP trigger (price >= take_profit)
            3. Accumulate step-wise returns

        Futures Mode:
            1. Check liquidation first (highest priority)
            2. Check SL trigger (direction-aware)
            3. Check TP trigger (direction-aware)
            4. Accumulate step-wise returns

        Returns:
            Tuple of (trade_info, obs_dict) where:
                - trade_info: Dict with execution details
                - obs_dict: Market observation at terminal state
        """
        trade_info = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False
        }
        self.rollout_returns = []
        obs_dict = None  # Initialize to prevent UnboundLocalError

        future_rollout_steps = 1
        while not self.truncated:
            # Get next time step
            obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()

            # Cache base features once per iteration
            ohlcv_base_values = self.sampler.get_base_features(self.current_timestamp)
            self._cached_base_features = ohlcv_base_values

            # Extract OHLCV for intrabar SL/TP detection
            open_price = ohlcv_base_values["open"]
            close_price = ohlcv_base_values["close"]
            high_price = ohlcv_base_values["high"]
            low_price = ohlcv_base_values["low"]

            # Accumulate step-wise return
            self.rollout_returns.append(self.compute_return(close_price))

            # Mode-aware trigger detection
            if self.trading_mode == "futures":
                # Futures: check liquidation first (highest priority)
                if trigger_result := self._check_liquidation_in_rollout(ohlcv_base_values):
                    return trigger_result, obs_dict

            # Check SL/TP triggers (mode-aware)
            if trigger_result := self._check_sltp_triggers(ohlcv_base_values):
                return trigger_result, obs_dict

            future_rollout_steps += 1

        # If loop never executed (truncated from start), get an observation
        if obs_dict is None:
            obs_dict = self._get_observation_scaffold()

        return trade_info, obs_dict

    def _check_liquidation_in_rollout(self, ohlcv: dict) -> Optional[Dict]:
        """Check if liquidation should trigger during rollout (futures only).

        This is separate from the parent _check_liquidation() to return
        trade_info directly for the rollout flow.

        Args:
            ohlcv: Dictionary with keys "open", "high", "low", "close", "volume"

        Returns:
            trade_info dict if liquidation triggered, None otherwise
        """
        if self.trading_mode == "spot":
            return None

        if self.position.position_size == 0:
            return None

        open_price = ohlcv["open"]
        high_price = ohlcv["high"]
        low_price = ohlcv["low"]
        close_price = ohlcv["close"]

        if self.position.position_size > 0:
            # Long position - liquidated if price drops to liquidation price
            if (open_price <= self.liquidation_price or
                low_price <= self.liquidation_price or
                close_price <= self.liquidation_price):
                return self._execute_liquidation(self.liquidation_price)
        else:
            # Short position - liquidated if price rises to liquidation price
            if (open_price >= self.liquidation_price or
                high_price >= self.liquidation_price or
                close_price >= self.liquidation_price):
                return self._execute_liquidation(self.liquidation_price)

        return None

    def _check_sltp_triggers(self, ohlcv: dict) -> Optional[Dict]:
        """Check if stop-loss or take-profit should trigger during rollout.

        Uses intrabar OHLC data to detect SL/TP triggers that may occur
        within the candle, not just at the close.

        Args:
            ohlcv: Dictionary with keys "open", "high", "low", "close", "volume"

        Returns:
            trade_info dict if SL/TP triggered, None otherwise
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
                if (open_price <= self.stop_loss or
                    low_price <= self.stop_loss or
                    close_price <= self.stop_loss):
                    return self._execute_sltp_close(self.stop_loss, "sl")

            # TP triggers when price rises above TP level
            if self.take_profit > 0:
                if (open_price >= self.take_profit or
                    high_price >= self.take_profit or
                    close_price >= self.take_profit):
                    return self._execute_sltp_close(self.take_profit, "tp")
        else:
            # Short position
            # SL triggers when price rises above SL level
            if self.stop_loss > 0:
                if (open_price >= self.stop_loss or
                    high_price >= self.stop_loss or
                    close_price >= self.stop_loss):
                    return self._execute_sltp_close(self.stop_loss, "sl")

            # TP triggers when price drops below TP level
            if self.take_profit > 0:
                if (open_price <= self.take_profit or
                    low_price <= self.take_profit or
                    close_price <= self.take_profit):
                    return self._execute_sltp_close(self.take_profit, "tp")

        return None

    def _execute_sltp_action(
        self, side: Optional[str], sl_pct: Optional[float], tp_pct: Optional[float], base_price: float
    ) -> Dict:
        """Execute action with SLTP bracket order setup.

        This overrides the parent method to ensure positions are opened with
        the one-step rollout pattern (no mid-position adjustments).

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
            return self._create_trade_info()

        # CLOSE action (shouldn't happen in one-step, but handle for safety)
        if side == "close":
            if self.position.position_size != 0:
                # Apply slippage
                price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
                execution_price = base_price * price_noise
                return self._close_position(execution_price)
            return self._create_trade_info()

        # Opening new position (long or short)
        # Check if already in same direction - if so, hold (ignore duplicate action)
        if side == "long" and self.position.position_size > 0:
            return self._create_trade_info()
        if side == "short" and self.position.position_size < 0:
            return self._create_trade_info()

        # If switching direction, close existing position first
        if self.position.position_size != 0:
            price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
            execution_price = base_price * price_noise
            self._close_position(execution_price)
            # Recalculate base_price after closing (balance may have changed)
            base_price = self._cached_base_features["close"]

        # Apply slippage for opening
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = base_price * price_noise

        # Initialize previous portfolio value for rollout returns
        self.previous_portfolio_value = self._get_portfolio_value(execution_price)

        # Open new position with SLTP brackets
        return self._open_position_with_sltp(side, execution_price, sl_pct, tp_pct)

    def close(self):
        """Clean up resources."""
        pass
