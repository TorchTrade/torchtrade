from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

import torch

logger = logging.getLogger(__name__)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.alpaca.utils import normalize_alpaca_timeframe_config
from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass, TradeMode
from tensordict import TensorDictBase
from torchrl.data import Categorical
from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_alpaca_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin


@dataclass
class AlpacaSLTPTradingEnvConfig:
    """Configuration for AlpacaSLTPTorchTradingEnv.

    This environment uses a combinatorial action space where each action
    represents a (stop_loss_pct, take_profit_pct) pair for bracket orders.
    """
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"
    # Stop loss levels as percentages (negative values, e.g., -0.025 = -2.5%)
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    # Take profit levels as percentages (positive values, e.g., 0.05 = 5%)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    include_hold_action: bool = True  # Include HOLD action (index 0) in action space
    include_close_action: bool = False  # Include CLOSE action for manual position exit (default: False for SLTP)
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    paper: bool = True
    trade_mode: TradeMode = "notional"
    seed: Optional[int] = 42
    include_base_features: bool = False

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = normalize_alpaca_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )


class AlpacaSLTPTorchTradingEnv(SLTPMixin, AlpacaBaseTorchTradingEnv):
    """Alpaca Live Trading Environment with Stop Loss and Take Profit action spec.

    This environment uses bracket orders to implement stop-loss and take-profit
    functionality. The action space is a categorical distribution over all
    combinations of (stop_loss, take_profit) levels plus a HOLD action.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: BUY with specific (stop_loss_pct, take_profit_pct) combination

    The environment automatically sells when either the stop-loss or take-profit
    is triggered by Alpaca's bracket order system.
    """

    def __init__(
        self,
        config: AlpacaSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """Initialize the AlpacaSLTPTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        # Set reward function
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Create action map from SL/TP combinations
        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_alpaca_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action
        )

        # Categorical action spec: 0=HOLD (if included), 1..N = SL/TP combinations
        self.action_spec = Categorical(len(self.action_map))

        # Track active SL/TP levels for current position
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment, including SLTP-specific state."""
        # Call base reset
        result = super()._reset(tensordict, **kwargs)

        # Reset SLTP-specific state using mixin
        self._reset_sltp_state()

        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

        # Get current price from trader status (avoids redundant observation call)
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        current_price = position_status.current_price if position_status else 0.0

        # Get action and map to SL/TP tuple
        action_idx = tensordict.get("action", 0)
        if hasattr(action_idx, "item"):
            action_idx = action_idx.item()
        action_tuple = self.action_map[action_idx]

        # Check if position was closed by SL/TP
        position_closed = self._check_position_closed()

        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        if trade_info["executed"] and trade_info.get("success") is not False:
            self.position.current_position = 1 if trade_info["side"] == "buy" else 0

        if position_closed:
            self.position.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Update position hold counter
        if self.position.current_position != 0:
            self.position.hold_counter += 1
        else:
            self.position.hold_counter = 0

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Convert action_tuple to numeric action for history
        action_value = 1.0 if action_tuple != (None, None) else 0.0

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=current_price,
            action=action_value,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([False], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict

    def _execute_trade_if_needed(self, action_tuple: Tuple[Optional[float], Optional[float]]) -> Dict:
        """Execute trade if position change is needed.

        Args:
            action_tuple: (stop_loss_pct, take_profit_pct) or (None, None) for HOLD

        Returns:
            Dict with trade execution info
        """
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None}

        stop_loss_pct, take_profit_pct = action_tuple

        # HOLD action or already in position
        if action_tuple == (None, None) or self.position.current_position == 1:
            return trade_info

        # BUY with SL/TP bracket order
        if self.position.current_position == 0 and stop_loss_pct is not None and take_profit_pct is not None:
            amount = self._calculate_trade_amount("buy")

            # Get current price to calculate absolute SL/TP levels
            status = self.trader.get_status()
            # Use market data to get current price
            obs = self.observer.get_observations(return_base_ohlc=True)
            current_price = obs["base_features"][-1, 3]  # Close price

            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            try:
                success = self.trader.trade(
                    side="buy",
                    amount=amount,
                    order_type="market",
                    time_in_force="gtc",
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                )

                if success:
                    self.active_stop_loss = stop_loss_price
                    self.active_take_profit = take_profit_price

                trade_info.update({
                    "executed": True,
                    "amount": amount,
                    "side": "buy",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                logger.error(
                    f"SLTP trade execution failed: buy ${amount:.2f} with SL={stop_loss_price:.2f}, TP={take_profit_price:.2f} - {str(e)}",
                    exc_info=True
                )
                trade_info["success"] = False

        return trade_info

    def _calculate_trade_amount(self, side: str) -> float:
        """Calculate the dollar amount to trade."""
        if self.config.trade_mode == "quantity":
            raise NotImplementedError("quantity trade mode not implemented for SLTP env")

        if side == "buy":
            return self.balance
        else:
            return -1

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    config = AlpacaSLTPTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
        ],
        window_sizes=[15],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        stoploss_levels=(-0.02, -0.05),
        takeprofit_levels=(0.03, 0.06, 0.10),
    )

    env = AlpacaSLTPTorchTradingEnv(
        config,
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY")
    )

    print(f"Action space size: {env.action_spec.n}")
    print(f"Action map: {env.action_map}")

    td = env.reset()
    print(td)
