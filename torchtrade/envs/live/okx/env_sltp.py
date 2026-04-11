"""OKX Futures TorchRL trading environment with Stop Loss and Take Profit."""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices
from torchtrade.envs.core.common import TradeMode

logger = logging.getLogger(__name__)


@dataclass
class OKXFuturesSLTPTradingEnvConfig:
    """Configuration for OKX Futures SLTP Trading Environment.

    Uses a combinatorial action space where each action represents a
    (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    """
    symbol: str = "BTC-USDT-SWAP"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"

    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.NET
    quantity_per_trade: float = 0.001
    trade_mode: TradeMode = "quantity"
    position_fraction: float = 1.0  # Used when trade_mode="fractional"
    lock_position_until_sltp: bool = False  # If True, ignore actions while in position

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
    close_position_on_init: bool = True
    close_position_on_reset: bool = False

    def __post_init__(self):
        from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config
        from torchtrade.envs.core.common import validate_trade_mode

        self.trade_mode = validate_trade_mode(self.trade_mode)
        if self.trade_mode == "fractional":
            if not (0 < self.position_fraction <= 1.0):
                raise ValueError(f"position_fraction must be in (0, 1.0], got {self.position_fraction}")
        elif self.trade_mode in ("notional", "quantity"):
            if self.quantity_per_trade <= 0:
                raise ValueError(f"quantity_per_trade must be positive, got {self.quantity_per_trade}")
        self.execute_on, self.time_frames, self.window_sizes = normalize_okx_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )


class OKXFuturesSLTPTorchTradingEnv(SLTPMixin, OKXBaseTorchTradingEnv):
    """
    OKX Futures trading environment with Stop Loss and Take Profit action spec.

    Uses bracket orders via OKX's attachAlgoOrds parameter.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific SL/TP combination (if enabled)
    """

    def __init__(
        self,
        config: OKXFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[OKXObservationClass] = None,
        trader: Optional[OKXFuturesOrderClass] = None,
    ):
        super().__init__(config, api_key, api_secret, passphrase, feature_preprocessing_fn, observer, trader)

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

        # Sync position state from exchange — this is the source of truth.
        position_closed = self._sync_position_from_exchange(position_status)

        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        if not isinstance(action_idx, int):
            if isinstance(action_idx, float) and math.isfinite(action_idx):
                action_idx = int(action_idx)
            else:
                logger.warning(f"Invalid action index {action_idx}, defaulting to 0")
                action_idx = 0
        if action_idx < 0 or action_idx >= len(self.action_map):
            logger.warning(f"Action index {action_idx} out of range [0, {len(self.action_map) - 1}], clamping")
            action_idx = max(0, min(action_idx, len(self.action_map) - 1))
        action_tuple = self.action_map[action_idx]

        # Execute trade if needed (duplicate guard uses synced state)
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        # Eagerly update position from trade result
        if trade_info["executed"] and trade_info.get("success") is not False:
            if trade_info.get("closed_position"):
                self.position.current_position = 0
            elif trade_info["side"] == "buy":
                self.position.current_position = 1
            elif trade_info["side"] == "sell":
                self.position.current_position = -1

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

        # Position locking: ignore all actions while in position
        if self.config.lock_position_until_sltp and self.position.current_position != 0:
            return trade_info

        # CLOSE action - close any open position
        if side == "close":
            if self.position.current_position == 0:
                return trade_info
            try:
                success = self.trader.close_position()
            except Exception as e:
                logger.error(f"Close position failed for {self.config.symbol}: {e}")
                return trade_info
            if success:
                close_side = "sell" if self.position.current_position > 0 else "buy"
                self.position.current_position = 0
                self.active_stop_loss = 0.0
                self.active_take_profit = 0.0
                trade_info.update({
                    "executed": True, "side": close_side,
                    "success": True, "closed_position": True,
                })
            return trade_info

        # Check if already in same position
        position_map = {"long": 1, "short": -1}
        if side in position_map and self.position.current_position == position_map[side]:
            return trade_info

        # Get current mark price (more accurate than candle close for bracket orders)
        current_price = float(self.trader.get_mark_price())

        # Resolve quantity based on trade_mode
        if self.config.trade_mode == "fractional":
            balance = float(self.trader.get_account_balance()["total_wallet_balance"])
            if current_price <= 0 or balance <= 0:
                logger.error(f"Invalid price={current_price} or balance={balance} for {self.config.symbol}")
                trade_info["success"] = False
                return trade_info
            quantity = balance * self.config.position_fraction * self.config.leverage / current_price
        elif self.config.trade_mode == "notional":
            if current_price <= 0:
                logger.error(f"Invalid current_price={current_price} for {self.config.symbol}")
                trade_info["success"] = False
                return trade_info
            quantity = float(self.config.quantity_per_trade) / current_price
        elif self.config.trade_mode == "quantity":
            quantity = float(self.config.quantity_per_trade)
        else:
            raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")

        # Close opposite position if switching directions
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
                quantity=quantity,
                order_type="market",
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
            )

            if success:
                self.active_stop_loss = stop_loss_price
                self.active_take_profit = take_profit_price

            trade_info.update({
                "executed": True,
                "quantity": quantity,
                "side": trade_side,
                "success": success,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
            })
        except Exception as e:
            logger.error(
                f"{side.capitalize()} trade failed for {self.config.symbol}: "
                f"quantity={quantity}, "
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
