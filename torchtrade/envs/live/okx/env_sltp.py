"""OKX Futures TorchRL trading environment with Stop Loss and Take Profit."""
import math
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import logging

from torchrl.data import Categorical

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

logger = logging.getLogger(__name__)


@dataclass
class OKXFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for OKX Futures SLTP Trading Environment.

    Uses a combinatorial action space where each action represents a
    (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    """
    symbol: str = "BTC-USDT-SWAP"

    # Trading parameters
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.NET

    _normalize_timeframes = staticmethod(normalize_okx_timeframe_config)


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

    def _execute_trade_if_needed(
        self, action_tuple: Tuple[Optional[str], Optional[float], Optional[float]],
        *, current_price: float,
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

        if side == "close":
            return self._close_action(trade_info)

        if side in self.SIDE_DIRECTION and self.position.current_position == self.SIDE_DIRECTION[side]:
            return trade_info

        # Get current mark price (more accurate than candle close for bracket orders)
        # Threaded from `_step`'s halted read; required, never defaulted (#295).
        current_price = float(current_price)
        # Re-validated at the seam that USES it, not just where it was read. Threading
        # moved the read out of this method, and with it `_current_mark_price`'s
        # `isfinite`/`<= 0` guard -- a 0.0 divides in the sizing path below.
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(
                f"venue reported an unusable mark price ({current_price})"
            )

        quantity = self._resolve_bracket_quantity(current_price)
        if quantity is None:
            trade_info["success"] = False
            return trade_info

        # Short-circuit if quantity is below exchange minimum
        lot = self.trader.get_lot_size()
        if quantity < lot["min_qty"]:
            trade_info["success"] = False
            return trade_info

        # Close opposite position if switching directions
        if self.position.current_position != 0:
            try:
                close_success = self.trader.close_position()
            except Exception as e:
                logger.error(f"Close position failed for {self.config.symbol}: {e}")
                return trade_info
            if not close_success:
                return trade_info
            # A realised close moves equity; the cached balance is now wrong by the
            # trade's P&L. Reached only on success (#295).
            self._last_confirmed_read.pop("balance", None)
            self.position.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

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
                # OKX attachAlgoOrds is atomic — SL/TP succeed or fail with the main order
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

    def _dispatch_sltp_trade(self, action_tuple, current_price: float):
        # Threaded, not re-read: re-reading the mark inside the trade path bypassed the
        # halt policy, so a grace bar that priced a bracket died instead of truncating
        # (#295). binance and bitget price off a candle close and take the default.
        return self._execute_trade_if_needed(action_tuple, current_price=current_price)
