from dataclasses import dataclass
from typing import Optional, Callable, Dict

from torchrl.data import Categorical

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import (
    BitgetFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bitget.base import BitgetBaseTorchTradingEnv
from torchtrade.envs.live.shared.futures_config import BaseFuturesTradingConfig
from torchtrade.envs.live.bitget.utils import normalize_bitget_timeframe_config


@dataclass
class BitgetFuturesTradingEnvConfig(BaseFuturesTradingConfig):
    """Configuration for Bitget Futures Trading Environment."""

    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY
    # V2 API: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES. A bare str with no
    # validator, so this comment is where the legal values live.
    product_type: str = "USDT-FUTURES"

    _normalize_timeframes = staticmethod(normalize_bitget_timeframe_config)


class BitgetFuturesTorchTradingEnv(BitgetBaseTorchTradingEnv):
    """
    TorchRL environment for Bitget Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Multiple timeframe observations
    - Demo (testnet) trading
    - Query-first pattern for reliable position tracking

    Action Space (Fractional Mode - Default):
    --------------------------------------
    Actions represent the fraction of available balance to allocate to a position.
    Action values in range [-1.0, 1.0]:

    - action = -1.0: 100% short (all-in short)
    - action = 0.0: Market neutral (close all positions, stay in cash)
    - action = 1.0: 100% long (all-in long)

    Position sizing formula:
        position_size = (balance × |action| × leverage) / price

    Default action_levels: [-1, 0, 1]; see BaseFuturesTradingConfig.action_levels.
    Custom levels supported: e.g., [-1, -0.3, -0.1, 0, 0.1, 0.3, 1]

    Leverage Design:
    ----------------
    Leverage is a **fixed global parameter** (not part of action space).
    See SeqFuturesEnv documentation for rationale on fixed vs dynamic leverage.

    **Dynamic Leverage** (not currently implemented):
    Could be implemented as multi-dimensional actions if needed, but fixed
    leverage is recommended for most use cases.

    Account State (6 elements; the list is ACCOUNT_STATE on the exchange base class):
    ---------------------------
    [exposure_pct, position_direction, unrealized_pnlpct, holding_time,
     leverage, distance_to_liquidation]
    """

    def __init__(
        self,
        config: BitgetFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BitgetObservationClass] = None,
        trader: Optional[BitgetFuturesOrderClass] = None,
    ):
        """
        Initialize the BitgetFuturesTorchTradingEnv.

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

        # Set reward function (default to log return reward)
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))


    def _execute_fractional_action(
        self, action_value: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """Execute action using fractional position sizing.

        Args:
            action_value: Fractional action value in [-1.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        # Get current position and price from exchange
        # Threaded from `_step`'s halted read; required, never defaulted (#295).
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            else:
                return self._create_trade_info(executed=False)

        # Calculate target position
        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)

        # Calculate delta (what we need to trade)
        delta_qty = target_qty - current_qty

        # Query real min-order-size from Bitget market info (not hardcoded)
        lot_size = self.trader.get_lot_size()
        min_qty = lot_size["min_qty"]

        if abs(delta_qty) < min_qty:
            # Already at target (delta below the minimum tradeable size)
            return self._create_trade_info(executed=False, at_target=True)

        side = "buy" if delta_qty > 0 else "sell"
        # Floor to the exchange lot step (CCXT truncates -> never exceeds margin)
        amount = self.trader._round_amount(abs(delta_qty))

        if amount < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        # The venue's notional floor, on the floored quantity actually sent (#414).
        raw_floor = lot_size["min_notional"]
        if raw_floor is None:
            # Unknown floor: refuse rather than assume there is none (#414).
            return self._create_trade_info(executed=False, at_target=True)
        min_notional = float(raw_floor)
        if min_notional > 0 and amount * current_price < min_notional:
            return self._create_trade_info(executed=False, at_target=True)

        # Execute market order
        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        info["target_tol"] = min_qty
        return info

    def _execute_trade_if_needed(
        self, desired_action: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """Execute trade based on desired action value.

        Skips execution if already in the requested position direction.

        Args:
            desired_action: Fractional action value in [-1.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        if desired_action == self.position.current_action_level:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(
            desired_action, current_qty=current_qty, current_price=current_price,
        )
