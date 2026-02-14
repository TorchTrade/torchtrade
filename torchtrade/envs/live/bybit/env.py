"""Bybit Futures TorchRL trading environment with fractional position sizing."""
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict
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
from torchtrade.envs.utils.fractional_sizing import (
    calculate_fractional_position,
    PositionCalculationParams,
)


logger = logging.getLogger(__name__)


@dataclass
class BybitFuturesTradingEnvConfig:
    """Configuration for Bybit Futures Trading Environment."""

    symbol: str = "BTCUSDT"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"

    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY

    # Action space configuration
    action_levels: List[float] = None

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

        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]


class BybitFuturesTorchTradingEnv(BybitBaseTorchTradingEnv):
    """
    TorchRL environment for Bybit Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-100x)
    - Multiple timeframe observations
    - Demo (testnet) trading
    - Fractional position sizing

    Action Space (Fractional Mode - Default):
    - action = -1.0: 100% short (all-in short)
    - action = -0.5: 50% short
    - action = 0.0: Market neutral (close all positions)
    - action = 0.5: 50% long
    - action = 1.0: 100% long (all-in long)

    Default action_levels: [-1.0, -0.5, 0.0, 0.5, 1.0]
    """

    def __init__(
        self,
        config: BybitFuturesTradingEnvConfig,
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

        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))

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
        if action_idx < 0 or action_idx >= len(self.action_levels):
            logger.warning(f"Action index {action_idx} out of range [0, {len(self.action_levels) - 1}], clamping")
            action_idx = max(0, min(action_idx, len(self.action_levels) - 1))
        desired_action = self.action_levels[action_idx]

        trade_info = self._execute_trade_if_needed(desired_action)

        if trade_info["executed"] and trade_info.get("success") is not False:
            if trade_info["side"] == "buy":
                self.position.current_position = 1
            elif trade_info["side"] == "sell" and trade_info.get("closed_position"):
                self.position.current_position = 0
            elif trade_info["side"] == "sell":
                self.position.current_position = -1
            self.position.current_action_level = desired_action

        self._wait_for_next_timestamp()

        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        self.history.record_step(
            price=current_price,
            action=desired_action,
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

    def _get_current_position_quantity(self) -> float:
        """Get current position quantity from trader status."""
        status = self.trader.get_status()
        position = status.get("position_status")
        return position.qty if position is not None else 0.0

    def _create_trade_info(self, executed=False, **kwargs) -> Dict:
        """Create trade info dictionary with defaults."""
        info = {
            "executed": executed,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }
        info.update(kwargs)
        return info

    def _execute_market_order(self, side: str, quantity: float) -> Dict:
        """Execute a market order with error handling."""
        try:
            success = self.trader.trade(
                side=side,
                quantity=quantity,
                order_type="market",
            )
            return self._create_trade_info(
                executed=True,
                quantity=quantity,
                side=side,
                success=success,
            )
        except Exception as e:
            logger.error(f"{side.capitalize()} trade failed for {self.config.symbol}: quantity={quantity}, error={e}")
            return self._create_trade_info(executed=False, success=False)

    def _handle_close_action(self, current_qty: float) -> Dict:
        """Handle close position action."""
        if current_qty == 0:
            return self._create_trade_info(executed=False)

        try:
            success = self.trader.close_position()
        except Exception as e:
            logger.error(f"Close position failed for {self.config.symbol}: {e}")
            return self._create_trade_info(executed=False, success=False)

        side = "sell" if current_qty > 0 else "buy"

        if success:
            self.position.current_position = 0

        return self._create_trade_info(
            executed=True,
            quantity=abs(current_qty),
            side=side,
            success=success,
            closed_position=True,
        )

    def _calculate_fractional_position(self, action_value: float, current_price: float) -> tuple[float, float, str]:
        """Calculate target position size from fractional action."""
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        balance_info = self.trader.get_account_balance()
        total_balance = balance_info.get('total_margin_balance', 0.0)

        if total_balance <= 0:
            logger.warning("No balance for fractional position sizing")
            return 0.0, 0.0, "flat"

        effective_balance = total_balance * 0.98
        fee_rate = 0.00055  # Bybit futures taker fee
        params = PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=fee_rate,
        )
        return calculate_fractional_position(params)

    def _execute_fractional_action(self, action_value: float) -> Dict:
        """Execute action using fractional position sizing."""
        current_qty = self._get_current_position_quantity()
        current_price = self.trader.get_mark_price()

        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            else:
                return self._create_trade_info(executed=False)

        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)
        delta_qty = target_qty - current_qty

        min_qty = 0.001
        if abs(delta_qty) < min_qty:
            return self._create_trade_info(executed=False)

        side = "buy" if delta_qty > 0 else "sell"
        amount = int(abs(delta_qty) / min_qty) * min_qty

        if amount < min_qty:
            return self._create_trade_info(executed=False)

        return self._execute_market_order(side, amount)

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade based on desired action value."""
        if desired_action == self.position.current_action_level:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(desired_action)

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold
