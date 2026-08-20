from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
)
from torchtrade.envs.live.bitget.order_executor import TAKER_FEE
import math
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import logging

import torch

logger = logging.getLogger(__name__)
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import (
    BitgetFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bitget.utils import normalize_bitget_timeframe_config
from torchtrade.envs.live.bitget.base import BitgetBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices


@dataclass
class BitgetFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for Bitget Futures SLTP Trading Environment.

    This environment uses a combinatorial action space where each action
    represents a (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    Supports both long and short positions with stop-loss/take-profit.
    """
    symbol: str = "BTC/USDT:USDT"  # CCXT perpetual swap format

    # Trading parameters
    product_type: str = "USDT-FUTURES"  # V2 API: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY  # ONE_WAY or HEDGE

    _normalize_timeframes = staticmethod(normalize_bitget_timeframe_config)


class BitgetFuturesSLTPTorchTradingEnv(SLTPMixin, BitgetBaseTorchTradingEnv):
    """
    Bitget Futures trading environment with Stop Loss and Take Profit action spec.

    This environment uses bracket orders to implement stop-loss and take-profit
    functionality for futures trading. The action space is a categorical distribution
    over all combinations of (side, stop_loss, take_profit) levels plus a HOLD action.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific (stop_loss_pct, take_profit_pct) combination (if enabled)

    The environment automatically closes the position when either the stop-loss or
    take-profit is triggered by Bitget's order system.

    Key differences from standard BitgetFuturesTorchTradingEnv:
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
        config: BitgetFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BitgetObservationClass] = None,
        trader: Optional[BitgetFuturesOrderClass] = None,
    ):
        """
        Initialize the BitgetFuturesSLTPTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Bitget API key
            api_secret: Bitget API secret
            api_passphrase: Bitget API passphrase (required!)
            feature_preprocessing_fn: Optional custom preprocessing function
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured BitgetObservationClass
            trader: Optional pre-configured BitgetFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, api_passphrase, feature_preprocessing_fn, observer, trader)

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

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment, including SLTP-specific state."""
        # Call base reset
        result = super()._reset(tensordict, **kwargs)

        # Reset SLTP-specific state using mixin
        self._reset_sltp_state()

        return result

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

        # Get current price and position from trader status (avoids redundant observation call)
        status, position_status, current_price, position_size = self._acquire_pre_trade_state()

        # Sync position state from exchange — this is the source of truth.
        # Detects SL/TP closures AND fixes state drift from failed bracket orders.
        position_closed = self._sync_position_from_exchange(position_status)

        # Get action and map to (side, SL, TP) tuple
        action_tuple = self.action_map[
            self._resolve_action_index(tensordict, len(self.action_map))
        ]

        # Execute trade if needed (duplicate guard uses synced state)
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        # Eagerly update position from trade result so the rest of this step
        # sees the new state without waiting for the next sync cycle.
        if trade_info["executed"] and trade_info.get("success") is not False:
            self._record_sltp_position(action_tuple[0])

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: there is no position mark to read, and
        # fetching one would add a round-trip that can halt the episode. The
        # pre-trade price is the honest fallback -- flat rows carry no PnL anyway.
        new_price = new_price if new_price is not None else current_price

        # Convert action_tuple to numeric action for history
        # action_tuple is (side, sl, tp) where side can be "long", "short", or None
        side, _, _ = action_tuple
        if side == "long":
            action_value = 1.0
        elif side == "short":
            action_value = -1.0
        else:
            action_value = 0.0

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=new_price,
            action=action_value,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=new_qty
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

        # Get current price for calculating absolute SL/TP levels
        obs = self.observer.get_observations(return_base_ohlc=True)
        current_price = float(obs["base_features"][-1, 3])  # Close price
        # Validated here, not per trade_mode: this price divides the notional sizing
        # AND prices both brackets in every mode, including the "quantity" default
        # which checked nothing. bybit/okx get this from _current_mark_price(); these
        # two read a candle close, which dropna() does not clear of inf (#347).
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(
                f"unusable close price ({current_price}) for {self.config.symbol}"
            )

        # Resolve quantity based on trade_mode
        if self.config.trade_mode == "fractional":
            # Size on total_margin_balance (equity), matching offline sizing and the non-SLTP
            # live path. Binance's total_wallet_balance excludes unrealized PnL and would under-size;
            # bitget/bybit/okx map both keys to equity, so the switch is a no-op there.
            balance = float(self.trader.get_account_balance()["total_margin_balance"])
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
                self.position.current_position = 0

        if side == "long":
            # Open LONG with SL/TP bracket order
            # Use helper to calculate correct SL/TP for longs
            stop_loss_price, take_profit_price = calculate_bracket_prices(
                "long", current_price, stop_loss_pct, take_profit_pct
            )

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
            # Use helper to calculate correct SL/TP for shorts
            # The action_map already negates SL/TP for shorts, helper handles this correctly
            stop_loss_price, take_profit_price = calculate_bracket_prices(
                "short", current_price, stop_loss_pct, take_profit_pct
            )

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
