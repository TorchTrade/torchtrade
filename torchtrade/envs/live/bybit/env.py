"""Bybit Futures TorchRL trading environment with fractional position sizing."""
import math

from torchtrade.envs.utils.precision import decimals_for_step
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict
import logging

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import (
    TAKER_FEE,
    BybitFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bybit.base import BybitBaseTorchTradingEnv
from torchtrade.envs.core.live import (
    ObservationFailurePolicy,
)
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
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
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: ObservationFailurePolicy | str = ObservationFailurePolicy.HALT

    def __post_init__(self):
        self.observation_failure_policy = ObservationFailurePolicy(self.observation_failure_policy)
        from torchtrade.envs.live.bybit.utils import normalize_bybit_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_bybit_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

        validate_action_levels(self.action_levels)


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
        status, position_status, current_price, position_size = self._acquire_pre_trade_state()

        # No-op today (this env's _execute_trade_if_needed recomputes qty live and never reads
        # current_action_level), but keeps the field consistent so adding a duplicate-action
        # guard here can't reintroduce the silent no-op that bit alpaca/binance/bitget.
        self._sync_position_from_exchange(position_status)

        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        if not isinstance(action_idx, int):
            if isinstance(action_idx, float) and math.isfinite(action_idx):
                action_idx = int(action_idx)
            else:
                logger.warning(f"Invalid action index {action_idx}, defaulting to 0")
                action_idx = 0
        if action_idx < 0 or action_idx >= len(self.action_levels):
            logger.warning(f"Action index {action_idx} out of range [0, {len(self.action_levels) - 1}], clamping")
            action_idx = max(0, min(action_idx, len(self.action_levels) - 1))
        desired_action = self.action_levels[action_idx]

        trade_info = self._execute_trade_if_needed(desired_action)

        self._record_position_after_trade(desired_action, trade_info)

        self._wait_for_next_timestamp()

        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: there is no position mark to read, and
        # fetching one would add a round-trip that can halt the episode. The
        # pre-trade price is the honest fallback -- flat rows carry no PnL anyway.
        new_price = new_price if new_price is not None else current_price

        self.history.record_step(
            price=new_price,
            action=desired_action,
            reward=0.0,
            portfolio_value=new_portfolio_value,
            position=new_qty
        )

        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict

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


    def _calculate_fractional_position(self, action_value: float, current_price: float) -> tuple[float, float, str]:
        """Calculate target position size from fractional action."""
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        balance_info = self.trader.get_account_balance()
        # Indexed, and `not (x > 0)`: defaulting to 0.0 turned a broken adapter into a
        # permanent silent refusal to trade, and `<= 0` lets a NaN balance through to
        # size a NaN position (#277).
        total_balance = balance_info['total_margin_balance']

        # isfinite, not `not (x > 0)`: that catches NaN but passes +inf, and an inf
        # balance sizes an inf target -- bitget's amount rounding then yields NaN and
        # hands it to create_order. Same defect this PR fixed on the baselines (#277).
        if not math.isfinite(total_balance) or total_balance <= 0:
            raise ValueError(
                f"cannot size a trade against a portfolio value of {total_balance}"
            )

        effective_balance = total_balance * 0.98
        params = PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=TAKER_FEE,
        )
        return calculate_fractional_position(params)

    def _execute_fractional_action(self, action_value: float) -> Dict:
        """Execute action using fractional position sizing."""
        current_qty = self._get_current_position_quantity()
        current_price = self._current_mark_price()

        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            return self._create_trade_info(executed=False)

        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)
        delta_qty = target_qty - current_qty

        lot_size = self.trader.get_lot_size()
        min_qty = lot_size["min_qty"]
        qty_step = lot_size["qty_step"]

        if abs(delta_qty) < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        side = "buy" if delta_qty > 0 else "sell"
        # round() to avoid float artifacts (e.g. 0.003000000000003). The decimals come
        # from Decimal, not from str(): a 1e-06 step has no "." in its repr, so the old
        # string check answered 0 and rounded 0.977 up to a whole unit (#278).
        amount = round(int(abs(delta_qty) / qty_step) * qty_step, decimals_for_step(qty_step))

        if amount < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        info["target_tol"] = lot_size["min_qty"]
        return info

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade based on desired action value."""
        return self._execute_fractional_action(desired_action)
