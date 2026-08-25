from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
)
from torchtrade.envs.live.binance.order_executor import TAKER_FEE
import math
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import logging


logger = logging.getLogger(__name__)
from torchrl.data import Categorical

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import (
    BinanceFuturesOrderClass,
    MarginMode,
)
from torchtrade.envs.live.binance.utils import normalize_binance_timeframe_config
from torchtrade.envs.live.binance.base import BinanceBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin


@dataclass
class BinanceFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for Binance Futures SLTP Trading Environment.

    This environment uses a combinatorial action space where each action
    represents a (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    Supports both long and short positions with stop-loss/take-profit.
    """

    # Trading parameters
    margin_mode: MarginMode = MarginMode.ISOLATED

    _normalize_timeframes = staticmethod(normalize_binance_timeframe_config)


class BinanceFuturesSLTPTorchTradingEnv(SLTPMixin, BinanceBaseTorchTradingEnv):
    """
    Binance Futures trading environment with Stop Loss and Take Profit action spec.

    This environment uses bracket orders to implement stop-loss and take-profit
    functionality for futures trading. The action space is a categorical distribution
    over all combinations of (side, stop_loss, take_profit) levels plus a HOLD action.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific (stop_loss_pct, take_profit_pct) combination (if enabled)

    The environment automatically closes the position when either the stop-loss or
    take-profit is triggered by Binance's bracket order system.

    Key differences from standard BinanceFuturesTorchTradingEnv:
    - Combinatorial action space with SL/TP levels
    - Bracket orders instead of simple market orders
    - Tracks active SL/TP levels
    - Can optionally disable short positions for long-only strategies

    Account State (6 elements; the list is ACCOUNT_STATE on the exchange base class):
    [exposure_pct, position_direction, unrealized_pnlpct, holding_time,
     leverage, distance_to_liquidation]
    """

    def __init__(
        self,
        config: BinanceFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        trader: Optional[BinanceFuturesOrderClass] = None,
    ):
        """
        Initialize the BinanceFuturesSLTPTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Binance API key
            api_secret: Binance API secret
            feature_preprocessing_fn: Optional custom preprocessing function
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured BinanceObservationClass
            trader: Optional pre-configured BinanceFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        # Set reward function
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Create action map from SL/TP combinations
        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_short_positions=config.include_short_positions,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action
        )

        # Categorical action spec: 0=HOLD, 1..N = (side, SL, TP) combinations
        self.action_spec = Categorical(len(self.action_map))

        # Track active SL/TP levels for current position
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0



    def _execute_trade_if_needed(
        self, action_tuple: Tuple[Optional[str], Optional[float], Optional[float]]
    ) -> Dict:
        """Execute trade if position change is needed.

        Args:
            action_tuple: (side, stop_loss_pct, take_profit_pct) or (None, None, None) for HOLD
                         side is "long", "short", or None

        Returns:
            Dict with trade execution info
        """
        trade_info = {
            "executed": False,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }

        side, stop_loss_pct, take_profit_pct = action_tuple

        # HOLD action - do nothing
        if side is None:
            return trade_info

        # Position locking: ignore all actions while in position
        if self.config.lock_position_until_sltp and self.position.current_position != 0:
            return trade_info

        # Check if already in same position (ignore duplicate actions)
        if side in self.SIDE_DIRECTION and self.position.current_position == self.SIDE_DIRECTION[side]:
            return trade_info  # Already in this position, ignore duplicate action

        # Under `_halting`, read AND verdict (#295). `get_observations` raises ValueError
        # on a short window or a stale last bar, and this sat outside the policy -- so a
        # degraded feed escaped as a bare ValueError in EVERY trade_mode, since this runs
        # before the mode branch. bybit/okx take a threaded mark instead; these two price
        # brackets off the candle close deliberately, so the read stays rather than being
        # replaced by the mark.
        def read_close():
            obs = self.observer.get_observations(return_base_ohlc=True)
            current_price = float(obs["base_features"][-1, 3])
            # This price divides the notional sizing AND prices both brackets in every
            # mode, including the "quantity" default which checked nothing. dropna() does
            # not clear a candle close of inf (#347). The name is load-bearing:
            # test_sltp_sizing_rejects_a_non_finite_price_or_balance greps for it.
            if not math.isfinite(current_price) or current_price <= 0:
                raise ValueError(
                    f"unusable close price ({current_price}) for {self.config.symbol}"
                )
            return current_price

        # cache_key is load-bearing, not decoration: without it `cached` is None, grace
        # cannot apply, and this still raises -- it just raises a nicer type. The claimed
        # behaviour is "serve the last CONFIRMED close and flag the bar", which needs a
        # slot to serve from. Its own slot, because it is a candle close, not the mark.
        current_price = self._halting(read_close, cache_key="candle_close")

        # Resolve quantity based on trade_mode
        if self.config.trade_mode == "fractional":
            # Size on total_margin_balance (equity), matching offline sizing and the non-SLTP
            # live path. Binance's total_wallet_balance excludes unrealized PnL and would under-size;
            # bitget/bybit/okx map both keys to equity, so the switch is a no-op there.
            # Under `_halting` -- the read that SIZES a bracket (#295).
            balance = float(
                self._halting(self.trader.get_account_balance, cache_key="balance")[
                    "total_margin_balance"
                ]
            )
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

        # Close opposite position if switching directions
        if self.position.current_position != 0:
            # We have an existing position that needs to be closed before opening new one
            if (side == "long" and self.position.current_position == -1) or \
               (side == "short" and self.position.current_position == 1):
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

        if side == "long":
            # Open LONG with SL/TP bracket order
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            try:
                success = self.trader.trade(
                    side="buy",
                    quantity=quantity,
                    order_type="market",
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
                    "quantity": quantity,
                    "side": "buy",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                logger.error(f"Long trade failed for {self.config.symbol}: quantity={quantity}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, error={e}")
                trade_info["success"] = False
                return trade_info

        elif side == "short":
            # Open SHORT with SL/TP bracket order
            # For shorts: SL is above entry, TP is below entry
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            try:
                success = self.trader.trade(
                    side="sell",
                    quantity=quantity,
                    order_type="market",
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                )

                if success:
                    bs = getattr(self.trader, 'bracket_status', {"tp_placed": True, "sl_placed": True})
                    self.active_stop_loss = stop_loss_price if bs["sl_placed"] else 0.0
                    self.active_take_profit = take_profit_price if bs["tp_placed"] else 0.0

                trade_info.update({
                    "executed": True,
                    "quantity": quantity,
                    "side": "sell",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                logger.error(f"Short trade failed for {self.config.symbol}: quantity={quantity}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, error={e}")
                trade_info["success"] = False
                return trade_info

        return trade_info
