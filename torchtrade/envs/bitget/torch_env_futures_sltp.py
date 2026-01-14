from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.bitget.obs_class import BitgetObservationClass
from torchtrade.envs.bitget.futures_order_executor import (
    BitgetFuturesOrderClass,
    TradeMode,
    MarginMode,
)
from torchtrade.envs.bitget.base import BitgetBaseTorchTradingEnv
from torchtrade.envs.action_maps import create_sltp_action_map
from torchtrade.envs.sltp_mixin import SLTPMixin
from torchtrade.envs.sltp_helpers import calculate_bracket_prices


@dataclass
class BitgetFuturesSLTPTradingEnvConfig:
    """Configuration for Bitget Futures SLTP Trading Environment.

    This environment uses a combinatorial action space where each action
    represents a (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    Supports both long and short positions with stop-loss/take-profit.
    """
    symbol: str = "BTCUSDT"

    # Timeframes and windows
    intervals: Union[List[str], str] = "1m"
    window_sizes: Union[List[int], int] = 10
    execute_on: str = "1m"  # Interval for trade execution timing

    # Trading parameters
    product_type: str = "SUMCBL"  # SUMCBL=testnet, UMCBL=production
    leverage: int = 1  # Leverage (1-125)
    margin_mode: MarginMode = MarginMode.ISOLATED
    quantity_per_trade: float = 0.001  # Base quantity per trade
    trade_mode: TradeMode = TradeMode.QUANTITY

    # Stop loss levels as percentages (negative values, e.g., -0.025 = -2.5%)
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    # Take profit levels as percentages (positive values, e.g., 0.05 = 5%)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    # Include short positions in action space
    include_short_positions: bool = True

    # Reward settings
    reward_scaling: float = 1.0

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use testnet
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_reset: bool = False  # Whether to close positions on env.reset()
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)

    def __post_init__(self):
        # Normalize to lists
        if isinstance(self.intervals, str):
            self.intervals = [self.intervals]
        if isinstance(self.window_sizes, int):
            self.window_sizes = [self.window_sizes]


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

    Account State (10 elements):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    """

    def __init__(
        self,
        config: BitgetFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
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
            observer: Optional pre-configured BitgetObservationClass
            trader: Optional pre-configured BitgetFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, api_passphrase, feature_preprocessing_fn, observer, trader)

        # Create action map from SL/TP combinations
        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_short_positions=config.include_short_positions
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
        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value()

        # Get current price and position from trader status (avoids redundant observation call)
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        if position_status:
            current_price = position_status.mark_price
            position_size = position_status.qty
        else:
            current_price = self.trader.get_mark_price()
            position_size = 0.0

        # Get action and map to (side, SL, TP) tuple
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self.action_map[action_idx]

        # Check if position was closed by SL/TP
        position_closed = self._check_position_closed()

        # Execute trade if needed
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        if trade_info["executed"]:
            if trade_info["side"] == "buy":
                self.position.current_position = 1  # Long
            elif trade_info["side"] == "sell":
                self.position.current_position = -1  # Short
            elif trade_info.get("closed_position"):
                self.position.current_position = 0  # Closed

        if position_closed:
            self.position.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Calculate reward and check termination
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, action_tuple, trade_info
        )
        done = self._check_termination(new_portfolio_value)

        # Record step history (convert action_tuple to numeric action)
        # action_tuple is (side, sl, tp) where side can be "long", "short", or None
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
            reward=reward,
            portfolio_value=old_portfolio_value,
            position=position_size
        )

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([False], dtype=torch.bool))
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

        # HOLD action or already in position
        if action_tuple == (None, None, None) or self.position.current_position != 0:
            return trade_info

        # Get current price for calculating absolute SL/TP levels
        obs = self.observer.get_observations(return_base_ohlc=True)
        current_price = obs["base_features"][-1, 3]  # Close price

        if side == "long":
            # Open LONG with SL/TP bracket order
            # Use helper to calculate correct SL/TP for longs
            stop_loss_price, take_profit_price = calculate_bracket_prices(
                "long", current_price, stop_loss_pct, take_profit_pct
            )

            try:
                success = self.trader.trade(
                    side="buy",
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
                    "side": "buy",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                print(f"Long trade failed: ${self.config.quantity_per_trade} with SL={stop_loss_price:.2f}, TP={take_profit_price:.2f} - {str(e)}")
                trade_info["success"] = False

        elif side == "short":
            # Open SHORT with SL/TP bracket order
            # Use helper to calculate correct SL/TP for shorts
            # The action_map already swaps SL/TP for shorts, helper handles this correctly
            stop_loss_price, take_profit_price = calculate_bracket_prices(
                "short", current_price, stop_loss_pct, take_profit_pct
            )

            try:
                success = self.trader.trade(
                    side="sell",
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
                    "side": "sell",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                print(f"Short trade failed: ${self.config.quantity_per_trade} with SL={stop_loss_price:.2f}, TP={take_profit_price:.2f} - {str(e)}")
                trade_info["success"] = False

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold


if __name__ == "__main__":
    import os

    print("Testing BitgetFuturesSLTPTorchTradingEnv...")

    # Create environment configuration
    config = BitgetFuturesSLTPTradingEnvConfig(
        symbol="BTCUSDT",
        demo=True,
        intervals=["1m"],
        window_sizes=[10],
        execute_on="1m",
        leverage=5,
        quantity_per_trade=0.002,
        stoploss_levels=(-0.02, -0.05),
        takeprofit_levels=(0.03, 0.06, 0.10),
        include_short_positions=True,  # Enable both long and short
        include_base_features=False,
    )

    try:
        # Create environment
        env = BitgetFuturesSLTPTorchTradingEnv(
            config,
            api_key=os.getenv("BITGET_API_KEY", ""),
            api_secret=os.getenv("BITGET_SECRET", ""),
            api_passphrase=os.getenv("BITGET_PASSPHRASE", ""),
        )

        print(f"✓ Environment created")
        print(f"  Action space size: {env.action_spec.n}")
        print(f"  Number of SL levels: {len(config.stoploss_levels)}")
        print(f"  Number of TP levels: {len(config.takeprofit_levels)}")
        print(f"  Long actions: {len(config.stoploss_levels) * len(config.takeprofit_levels)}")
        if config.include_short_positions:
            print(f"  Short actions: {len(config.stoploss_levels) * len(config.takeprofit_levels)}")

        print(f"\n  Action map (first 5 and last 5):")
        action_items = list(env.action_map.items())
        for idx, action in action_items[:5]:
            print(f"    Action {idx}: {action}")
        print("    ...")
        for idx, action in action_items[-5:]:
            print(f"    Action {idx}: {action}")

        # Test reset
        print("\n✓ Testing reset...")
        td = env.reset()
        print(f"  Observation keys: {list(td.keys())}")
        print(f"  Account state shape: {td['account_state'].shape}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
