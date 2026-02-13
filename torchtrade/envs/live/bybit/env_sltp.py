"""Bybit Futures TorchRL trading environment with Stop Loss and Take Profit."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import (
    BybitFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bybit.base import BybitBaseTorchTradingEnv
from torchtrade.envs.core.common import TradeMode
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

logger = logging.getLogger(__name__)


@dataclass
class BybitFuturesSLTPTradingEnvConfig:
    """Configuration for Bybit Futures SLTP Trading Environment.

    Uses a combinatorial action space where each action represents a
    (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    """
    symbol: str = "BTCUSDT"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"

    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY
    quantity_per_trade: float = 0.001
    trade_mode: TradeMode = "quantity"

    # Stop loss levels as percentages (negative values, e.g., -0.025 = -2.5%)
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    # Take profit levels as percentages (positive values, e.g., 0.05 = 5%)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    # Include short positions in action space
    include_short_positions: bool = True
    # Include HOLD action (index 0)
    include_hold_action: bool = True
    # Include CLOSE action for manual position exit
    include_close_action: bool = False

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Environment settings
    demo: bool = True
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_reset: bool = False

    def __post_init__(self):
        from torchtrade.envs.live.bybit.utils import normalize_bybit_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_bybit_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )


class BybitFuturesSLTPTorchTradingEnv(SLTPMixin, BybitBaseTorchTradingEnv):
    """
    Bybit Futures trading environment with Stop Loss and Take Profit action spec.

    Uses bracket orders via pybit's native takeProfit/stopLoss parameters.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific SL/TP combination (if enabled)
    """

    def __init__(
        self,
        config: BybitFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BybitObservationClass] = None,
        trader: Optional[BybitFuturesOrderClass] = None,
    ):
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_short_positions=config.include_short_positions,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action
        )

        self.action_spec = Categorical(len(self.action_map))

        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment, including SLTP-specific state."""
        result = super()._reset(tensordict, **kwargs)
        self._reset_sltp_state()
        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        if position_status:
            current_price = position_status.mark_price
            position_size = position_status.qty
        else:
            current_price = self.trader.get_mark_price()
            position_size = 0.0

        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self.action_map[action_idx]

        position_closed = self._check_position_closed()

        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        if trade_info["executed"] and trade_info.get("success") is not False:
            if trade_info["side"] == "buy":
                self.position.current_position = 1
            elif trade_info["side"] == "sell":
                self.position.current_position = -1
            elif trade_info.get("closed_position"):
                self.position.current_position = 0

        if position_closed:
            self.position.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        self._wait_for_next_timestamp()

        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        side, _, _ = action_tuple
        if side == "long":
            action_value = 1.0
        elif side == "short":
            action_value = -1.0
        else:
            action_value = 0.0

        self.history.record_step(
            price=current_price,
            action=action_value,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=position_size
        )

        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([False], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict

    def _execute_trade_if_needed(
        self, action_tuple: Tuple[Optional[str], Optional[float], Optional[float]]
    ) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {
            "executed": False,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }

        side, stop_loss_pct, take_profit_pct = action_tuple

        # HOLD action
        if side is None:
            return trade_info

        # Check if already in same position
        position_map = {"long": 1, "short": -1}
        if side in position_map and self.position.current_position == position_map[side]:
            return trade_info

        # Get current price
        obs = self.observer.get_observations(return_base_ohlc=True)
        current_price = obs["base_features"][-1, 3]  # Close price

        # Close opposite position if switching directions
        # (we already returned above if same direction, so any existing position is opposite)
        if self.position.current_position != 0:
            try:
                close_success = self.trader.close_position()
            except Exception as e:
                logger.error(f"Close position failed for {self.config.symbol}: {e}")
                return trade_info
            if not close_success:
                return trade_info
            self.position.current_position = 0

        # Map position side to trade side
        trade_side = "buy" if side == "long" else "sell"

        stop_loss_price, take_profit_price = calculate_bracket_prices(
            side, current_price, stop_loss_pct, take_profit_pct
        )

        try:
            success = self.trader.trade(
                side=trade_side,
                quantity=self.config.quantity_per_trade,
                order_type="market",
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
            )

            if success:
                self.active_stop_loss = stop_loss_price
                self.active_take_profit = take_profit_price

            trade_info.update({
                "executed": True,
                "quantity": self.config.quantity_per_trade,
                "side": trade_side,
                "success": success,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
            })
        except Exception as e:
            logger.error(
                f"{side.capitalize()} trade failed for {self.config.symbol}: "
                f"quantity={self.config.quantity_per_trade}, "
                f"SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, error={e}"
            )
            trade_info["success"] = False
            return trade_info

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold
