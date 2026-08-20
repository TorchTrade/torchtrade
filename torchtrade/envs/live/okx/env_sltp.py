"""OKX Futures TorchRL trading environment with Stop Loss and Take Profit."""
from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
)
from torchtrade.envs.live.okx.order_executor import TAKER_FEE
import math
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
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

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment, including SLTP-specific state."""
        result = super()._reset(tensordict, **kwargs)
        self._reset_sltp_state()
        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        status, position_status, current_price, position_size = self._acquire_pre_trade_state()

        # Sync position state from exchange — this is the source of truth.
        position_closed = self._sync_position_from_exchange(position_status)

        action_tuple = self._resolve_action_tuple(tensordict)

        # Execute trade if needed (duplicate guard uses synced state)
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        # Eagerly update position from trade result
        if trade_info["executed"] and trade_info.get("success") is not False:
            self._record_sltp_position(action_tuple[0])

        self._wait_for_next_timestamp()

        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: there is no position mark to read, and
        # fetching one would add a round-trip that can halt the episode. The
        # pre-trade price is the honest fallback -- flat rows carry no PnL anyway.
        new_price = new_price if new_price is not None else current_price

        side, _, _ = action_tuple
        if side == "long":
            action_value = 1.0
        elif side == "short":
            action_value = -1.0
        else:
            action_value = 0.0

        self.history.record_step(
            price=new_price,
            action=action_value,
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
        if side in self.SIDE_DIRECTION and self.position.current_position == self.SIDE_DIRECTION[side]:
            return trade_info

        # Get current mark price (more accurate than candle close for bracket orders)
        current_price = float(self._current_mark_price())

        # Resolve quantity based on trade_mode
        if self.config.trade_mode == "fractional":
            # Size on total_margin_balance (equity), matching offline sizing and the non-SLTP
            # live path. Binance's total_wallet_balance excludes unrealized PnL and would under-size;
            # bitget/bybit/okx map both keys to equity, so the switch is a no-op there.
            balance = float(self.trader.get_account_balance()["total_margin_balance"])
            # current_price already raised in _current_mark_price(); balance has
            # no such accessor, and `nan <= 0` is False (#347).
            if not math.isfinite(balance) or balance <= 0:
                logger.error(f"Invalid price={current_price} or balance={balance} for {self.config.symbol}")
                trade_info["success"] = False
                return trade_info
            # Reserve what will actually be CHARGED: ReplayOrderExecutor carries its own
            # rate, so reserving a constant left a higher-fee caller with every open
            # refused -- #278 reproduced. 0.98 as the non-SLTP path uses, so a
            # full-fraction open leaves some maintenance buffer instead of zero.
            # Reserve what the trader will CHARGE, or say so. float() alone is unsafe --
            # MagicMock implements __float__ and returns 1.0 -- but an isinstance chain
            # is worse: it rejects np.float32 and Decimal, which reproduces #278 with no
            # diagnostic at all (the only log is the executor's "Insufficient balance",
            # which reads like a small account rather than an under-reserving sizer).
            # Coerce, range-check, and WARN on the fallback.
            raw = getattr(self.trader, "transaction_fee", None)
            fee = TAKER_FEE
            if raw is not None:
                try:
                    candidate = float(raw)
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None and math.isfinite(candidate) and 0 <= candidate < 1:
                    fee = candidate
                else:
                    logger.warning(
                        f"{self.config.symbol}: trader.transaction_fee={raw!r} is not a "
                        f"usable rate; reserving the venue constant {TAKER_FEE}. If the "
                        f"trader charges more, opens will be refused."
                    )
            quantity = abs(calculate_fractional_position(PositionCalculationParams(
                balance=balance * 0.98,
                action_value=self.config.position_fraction,
                current_price=current_price,
                leverage=self.config.leverage,
                transaction_fee=fee,
            ))[0])
        elif self.config.trade_mode == "notional":
            quantity = float(self.config.quantity_per_trade) / current_price
        elif self.config.trade_mode == "quantity":
            quantity = float(self.config.quantity_per_trade)
        else:
            raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")

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
