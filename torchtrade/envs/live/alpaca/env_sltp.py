import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

import torch

logger = logging.getLogger(__name__)
from torchtrade.envs.core.state import position_qty_from_status
from torchtrade.envs.utils.timeframe import TimeFrame
from torchtrade.envs.core.common import (
    validate_position_sizing,
    validate_sltp_levels,
    validate_trade_mode,
)
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
    # Parity with the four futures exchanges (#289): alpaca hardcoded the init close and
    # offered no reset close at all, so a restart had no escape hatch for a stale
    # position. Defaults reproduce the previous behaviour exactly.
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    trade_mode: TradeMode = "fractional"
    position_fraction: float = 1.0        # Used when trade_mode="fractional" (1.0 = all-in, backward compat)
    quantity_per_trade: float = 100.0      # Used when trade_mode="notional"
    lock_position_until_sltp: bool = False  # If True, ignore actions while in position
    seed: Optional[int] = 42
    include_base_features: bool = False

    def __post_init__(self):
        self.trade_mode = validate_trade_mode(self.trade_mode)
        validate_sltp_levels(self.stoploss_levels, self.takeprofit_levels)
        if self.trade_mode == "quantity":
            raise ValueError(
                "trade_mode='quantity' is not supported for Alpaca SLTP -- its bracket "
                "order API takes dollar amounts. Use 'notional' or 'fractional'. "
                "(_calculate_trade_amount raised this one trade later, mid-episode.)"
            )
        validate_position_sizing(
            self.trade_mode, self.position_fraction, self.quantity_per_trade
        )
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

        # Fractional mode computes USD amount, so executor uses notional API
        if self.config.trade_mode == "fractional":
            self.trader.trade_mode = "notional"

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
        # Shared fallback chain, not an inline read: a flat bar has no position_status,
        # and the inline form recorded a price of 0 for it (#290).
        current_price = self._get_current_price(position_status)
        # From the exchange, as every other live env does: the alpaca envs never populate
        # self.position.position_size, so reading it recorded a flat history forever
        # (#290). This is the size held ENTERING the bar, matching bybit.
        position_size = position_qty_from_status(position_status)

        # Sync position state from exchange — this is the source of truth.
        # Detects SL/TP closures AND fixes state drift from failed bracket orders.
        position_closed = self._sync_position_from_exchange(position_status)

        # Get action and map to SL/TP tuple
        action_tuple = self._resolve_action_tuple(tensordict)

        # Calculate and execute trade if needed (duplicate guard uses synced state)
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        # Eagerly update position from trade result so the rest of this step
        # sees the new state without waiting for the next sync cycle.
        if trade_info["executed"] and trade_info.get("success") is not False:
            # Alpaca's action map drops the side (long-only spot), so a bracket tuple
            # targets a long and the close action's (None, None) targets flat.
            self._record_sltp_position("long" if action_tuple[0] is not None else None)

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Observation FIRST, then the portfolio value (#278). Under a ReplayObserver the
        # clock advances only inside get_observations(), so reading PV first recorded the
        # PREVIOUS bar's equity against this bar's action.
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()
        # Post-bar price too: recording the pre-trade one here put two different bars in
        # a single history row (#278). Non-fatal, unlike the observation and equity reads
        # above -- this value only labels a history row, and letting it raise here would
        # add a failure point that can end a live episode for a price nothing trades on.
        try:
            new_price = self._get_current_price()
        except Exception:
            new_price = 0.0
        # Post-trade size too: recording the size held ENTERING the bar against a
        # post-bar price and PV labels a return with the exposure that did not produce
        # it -- opening rows read flat, closing rows read still-open. Offline records the
        # post-trade size. Non-fatal for the same reason as the price.
        try:
            new_qty = position_qty_from_status(
                self.trader.get_status().get("position_status", None)
            )
        except Exception:
            logger.warning(
                "post-bar position unavailable for %s; the history row will carry the "
                "pre-trade size", self.config.symbol,
            )
            new_qty = position_size

        if not new_price or not math.isfinite(new_price) or new_price <= 0:
            # _get_current_price RETURNS 0.0 when all three sources fail rather than
            # raising, so an except-only guard never fired on the path that actually
            # degrades (#290). Check the value, not just the exception.
            logger.warning(
                "post-bar price unavailable for %s; the history row will carry the "
                "pre-trade price", self.config.symbol,
            )
            new_price = current_price

        # Convert action_tuple to numeric action for history
        action_value = 1.0 if action_tuple != (None, None) else 0.0

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=new_price,
            action=action_value,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=new_qty,
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
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

        # HOLD action or already in position (Alpaca is long-only, so position=1 means locked)
        if action_tuple == (None, None) or self.position.current_position == 1:
            return trade_info

        # Position locking: ignore all actions while in position
        if self.config.lock_position_until_sltp and self.position.current_position != 0:
            return trade_info

        # BUY with SL/TP bracket order
        if self.position.current_position == 0 and stop_loss_pct is not None and take_profit_pct is not None:
            amount = self._calculate_trade_amount("buy")

            # Get current price to calculate absolute SL/TP levels
            # Use market data to get current price
            obs = self.observer.get_observations(return_base_ohlc=True)
            current_price = float(obs["base_features"][-1, 3])  # Close price
            # This prices BOTH brackets: a NaN close sends NaN legs, and a negative one
            # puts the stop above the take-profit. The entry is a full-balance market buy
            # either way, and if the venue takes it while rejecting the legs the position
            # sits unprotected in an env whose only exit is SL/TP (#347).
            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError(
                    f"unusable close price ({current_price}) for {self.config.symbol}"
                )

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
                    # Only record SL/TP levels that actually placed on-exchange
                    bs = getattr(self.trader, 'bracket_status', {"tp_placed": True, "sl_placed": True})
                    self.active_stop_loss = stop_loss_price if bs["sl_placed"] else 0.0
                    self.active_take_profit = take_profit_price if bs["tp_placed"] else 0.0

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
        """Calculate the trade amount based on trade_mode.

        Returns:
            Amount in USD (for notional/fractional) or units (for quantity).
        """
        if side != "buy":
            return -1  # Close full position

        if self.config.trade_mode == "fractional":
            # Not fee-reserved, unlike the futures SLTP envs (#278). Alpaca's rate is
            # asset-class dependent -- commission-free on stocks, a taker fee on crypto --
            # so there is no single constant to reserve, and spot at leverage 1 makes the
            # shortfall the fee itself rather than a multiple of it. Left deliberately.
            return self.balance * self.config.position_fraction
        elif self.config.trade_mode == "notional":
            return float(self.config.quantity_per_trade)
        elif self.config.trade_mode == "quantity":
            raise NotImplementedError(
                "quantity trade_mode is not supported for Alpaca SLTP. "
                "Alpaca's bracket order API requires dollar amounts. "
                "Use trade_mode='notional' or 'fractional' instead."
            )
        else:
            raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")
