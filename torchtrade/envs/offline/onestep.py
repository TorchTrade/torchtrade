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
from torchtrade.envs.core.state import binarize_action_type


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
        reward_function: Optional[Callable] = None,
    ):
        """Initialize the one-step trading environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration with one-step parameters
            feature_preprocessing_fn: Optional function to preprocess features
            reward_function: Optional reward function (default: log_return_reward)
        """
        # Initialize parent class (SequentialTradingEnvSLTP)
        super().__init__(df, config, feature_preprocessing_fn, reward_function)

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
        exposure_pct = self.position.position_value / portfolio_value if portfolio_value > 0 else 0.0

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

        # Get desired action
        action_idx = tensordict["action"]
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self._action_tuple[action_idx]
        side, sl_pct, tp_pct = action_tuple

        # Execute action (only HOLD or open position - no closing in one-step)
        if self._check_liquidation(cached_price):
            trade_info = self._execute_liquidation()
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

        # Get updated state (advances timestamp and triggers rollout if position opened)
        # This is where the magic happens - _get_observation() will call _rollout()
        # if a position was opened, simulating until SL/TP trigger or truncation
        next_tensordict = self._get_observation()
        new_price = self._cached_base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Determine action_type and binarize action for history
        action_type = trade_info.get("side") or "hold"
        binarized_action = binarize_action_type(action_type)

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=cached_price,
            action=binarized_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=self.position.position_size,
            action_type=action_type
        )

        # Calculate terminal reward (accumulated over rollout) using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # If truncated without trigger, set reward to 0 (no reward for incomplete episodes)
        if self.truncated and not trade_info.get("executed"):
            reward = 0.0

        # Update the reward in history
        self.history.rewards[-1] = reward

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
            if self.leverage > 1:
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
        if self.leverage == 1:
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
                return self._execute_liquidation()
        else:
            # Short position - liquidated if price rises to liquidation price
            if (open_price >= self.liquidation_price or
                high_price >= self.liquidation_price or
                close_price >= self.liquidation_price):
                return self._execute_liquidation()

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
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # CLOSE action (shouldn't happen in one-step, but handle for safety)
        if side == "close":
            if self.position.position_size != 0:
                # Apply slippage
                price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
                execution_price = base_price * price_noise
                return self._close_position(execution_price)
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Opening new position (long or short)
        # Check if already in same direction - if so, hold (ignore duplicate action)
        if side == "long" and self.position.position_size > 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}
        if side == "short" and self.position.position_size < 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

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
    config = OneStepTradingEnvConfig(
        execute_on="1Hour",
        time_frames="1Hour",
        initial_cash=10000,
        transaction_fee=0.0,
        slippage=0.0,
        stoploss_levels=(-0.025, -0.05, -0.1),
        takeprofit_levels=(0.05, 0.1, 0.2),
        include_hold_action=True,
        random_start=True,
    )
    env = OneStepTradingEnv(df, config)
    print(f"Environment created (One-Step mode)")
    print(f"Action map size: {len(env.action_map)}")
    print(f"Action space size: {env.action_spec.n}")
    print(f"Sample actions:")
    for i in range(min(5, len(env.action_map))):
        print(f"  {i}: {env.action_map[i]}")
    print()

    # Run multiple episodes (one-step per episode)
    print("Running episodes (one decision per episode)...\n")
    episode_rewards = []

    for episode in range(50):
        td = env.reset()
        print(f"Episode {episode}")
        print("Initial account state:")
        acc_state = ""
        for key, value in zip(env.account_state, td["account_state"]):
            acc_state += f"  {key}: {value:.6f}\n"
        print(acc_state)

        # Random action (mostly hold, occasionally take position)
        action = np.random.choice(
            range(len(env.action_map)),
            p=[0.50] + [0.50 / (len(env.action_map) - 1)] * (len(env.action_map) - 1)
        )
        print(f"Action: {action} {env.action_map[action]}")

        td["action"] = action
        td = env.step(td)

        # Extract terminal reward
        reward = td["next"]["reward"].item()
        episode_rewards.append(reward)
        print(f"Terminal reward: {reward:.6f}")
        print(f"Done: {td['next']['done'].item()}\n")

    # Statistics
    print("\nEpisode Statistics:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Mean reward: {np.mean(episode_rewards):.6f}")
    print(f"  Std reward: {np.std(episode_rewards):.6f}")
    print(f"  Min reward: {np.min(episode_rewards):.6f}")
    print(f"  Max reward: {np.max(episode_rewards):.6f}")

    # Render history (last episode)
    print("\nRendering last episode history...")
    env.render_history(return_fig=False)

    env.close()
    print("\nDone!")
