import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
from itertools import product

import numpy as np
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.alpaca.obs_class import AlpacaObservationClass
from torchtrade.envs.alpaca.order_executor import AlpacaOrderClass, TradeMode
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Categorical, Bounded


def combinatory_action_map(stoploss_levels: List[float], takeprofit_levels: List[float]) -> Dict:
    """Create a mapping from action indices to (stop_loss, take_profit) tuples.

    Action 0 = HOLD (no trade)
    Actions 1..N = combinations of (stoploss_level, takeprofit_level)
    """
    action_map = {}
    # 0 = HOLD
    action_map[0] = (None, None)
    idx = 1
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = (sl, tp)
        idx += 1
    return action_map


@dataclass
class AlpacaSLTPTradingEnvConfig:
    """Configuration for AlpacaSLTPTorchTradingEnv.

    This environment uses a combinatorial action space where each action
    represents a (stop_loss_pct, take_profit_pct) pair for bracket orders.
    """
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )
    # Stop loss levels as percentages (negative values, e.g., -0.025 = -2.5%)
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    # Take profit levels as percentages (positive values, e.g., 0.05 = 5%)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    reward_scaling: float = 1.0
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    paper: bool = True
    trade_mode: TradeMode = TradeMode.NOTIONAL
    seed: Optional[int] = 42
    include_base_features: bool = False


class AlpacaSLTPTorchTradingEnv(EnvBase):
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
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """Initialize the AlpacaSLTPTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        self.config = config

        # Initialize Alpaca clients - use injected instances or create new ones
        self.observer = observer if observer is not None else AlpacaObservationClass(
            symbol=config.symbol,
            timeframes=config.time_frames,
            window_sizes=config.window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
        )

        self.trader = trader if trader is not None else AlpacaOrderClass(
            symbol=config.symbol.replace('/', ''),
            trade_mode=config.trade_mode,
            api_key=api_key,
            api_secret=api_secret,
            paper=config.paper,
        )

        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.amount
        self.execute_on_unit = str(config.execute_on.unit)

        # Reset settings
        self.trader.close_all_positions()
        self.trader.cancel_open_orders()

        account = self.trader.client.get_account()
        cash = float(account.cash)
        self.initial_portfolio_value = cash
        self.position_hold_counter = 0

        # Create action map from SL/TP combinations
        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = combinatory_action_map(self.stoploss_levels, self.takeprofit_levels)

        # Categorical action spec: 0=HOLD, 1..N = SL/TP combinations
        self.action_spec = Categorical(len(self.action_map))

        # Get the number of features from the observer
        num_features = self.observer.get_observations()[
            self.observer.get_keys()[0]
        ].shape[1]

        # Get market data obs names
        market_data_names = self.observer.get_keys()

        # Observation space includes market data features and current position info
        self.observation_spec = CompositeSpec(shape=())

        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec: [cash, position_size, position_value, entry_price,
        #                      current_price, unrealized_pnlpct, holding_time]
        account_state_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(7,), dtype=torch.float)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            window_size = config.window_sizes[i] if isinstance(config.window_sizes, list) else config.window_sizes
            market_data_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(window_size, num_features), dtype=torch.float)
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)

        # Track active SL/TP levels for current position
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0

        self._reset(TensorDict({}))
        super().__init__()

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(return_base_ohlc=True if self.config.include_base_features else False)

        if self.config.include_base_features:
            base_features = obs_dict["base_features"][-1]

        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state
        status = self.trader.get_status()
        account = self.trader.client.get_account()
        cash = float(account.cash)
        position_status = status.get("position_status", None)

        if position_status is None:
            self.position_hold_counter = 0
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            current_price = 0.0
            unrealized_pnlpc = 0.0
            holding_time = self.position_hold_counter
        else:
            self.position_hold_counter += 1
            position_size = position_status.qty
            position_value = position_status.market_value
            entry_price = position_status.avg_entry_price
            current_price = position_status.current_price
            unrealized_pnlpc = position_status.unrealized_plpc
            holding_time = self.position_hold_counter

        # Account state: [cash, position_size, position_value, entry_price,
        #                 current_price, unrealized_pnlpct, holding_time]
        account_state = torch.tensor(
            [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time],
            dtype=torch.float
        )

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action_tuple: Tuple[Optional[float], Optional[float]],
        trade_info: Dict,
    ) -> float:
        """Calculate the step reward.

        Reward is based on realized profit when position is closed (by SL/TP trigger
        or manual sell). A small penalty is applied for invalid actions.

        Args:
            old_portfolio_value: Portfolio value before the action.
            new_portfolio_value: Portfolio value after the action.
            action_tuple: The (stop_loss, take_profit) tuple for this action.
            trade_info: Trade information dict.

        Returns:
            float: The reward for this step.
        """
        # Check if position was closed (SL/TP triggered or sold)
        if trade_info.get("position_closed", False):
            portfolio_return = (
                new_portfolio_value - old_portfolio_value
            ) / old_portfolio_value
        elif not trade_info["executed"] and action_tuple != (None, None):
            # Penalty for trying to open position when already holding
            portfolio_return = -0.001
        else:
            portfolio_return = 0.0

        return portfolio_return * self.config.reward_scaling

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        account = self.trader.client.get_account()
        self.balance = float(account.cash)

        if position_status is None:
            return self.balance
        return self.balance + position_status.market_value

    def _wait_for_next_timestamp(self) -> None:
        """Wait until the next time step based on the configured execute_on_value and execute_on_unit."""
        unit_to_timedelta = {
            "TimeFrameUnit.Minute": "minutes",
            "TimeFrameUnit.Hour": "hours",
            "TimeFrameUnit.Day": "days",
        }

        if self.execute_on_unit not in unit_to_timedelta:
            raise ValueError(f"Unsupported time unit: {self.execute_on_unit}")

        wait_duration = timedelta(
            **{unit_to_timedelta[self.execute_on_unit]: self.execute_on_value}
        )

        current_time = datetime.now(ZoneInfo("America/New_York"))
        next_step = (current_time + wait_duration).replace(second=0, microsecond=0)

        while datetime.now(ZoneInfo("America/New_York")) < next_step:
            time.sleep(1)

    def _set_seed(self, seed: int):
        """Set the seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Cancel all orders and close all positions
        self.trader.cancel_open_orders()
        account = self.trader.client.get_account()
        self.balance = float(account.cash)
        self.last_portfolio_value = self.balance
        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position_hold_counter = 0
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0

        if position_status is None:
            self.current_position = 0.0
        else:
            self.current_position = 1 if position_status.qty > 0 else 0

        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value()

        # Get action and map to SL/TP tuple
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self.action_map[action_idx]

        # Check if position was closed by SL/TP
        position_closed = self._check_position_closed()

        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_info["position_closed"] = position_closed

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        if position_closed:
            self.current_position = 0
            self.active_stop_loss = 0.0
            self.active_take_profit = 0.0

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action_tuple, trade_info)
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", done)
        next_tensordict.set("truncated", False)
        next_tensordict.set("terminated", False)

        return next_tensordict

    def _check_position_closed(self) -> bool:
        """Check if position was closed by stop-loss or take-profit trigger."""
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        # If we had a position but now we don't, it was closed
        if self.current_position == 1 and position_status is None:
            return True
        return False

    def _execute_trade_if_needed(self, action_tuple: Tuple[Optional[float], Optional[float]]) -> Dict:
        """Execute trade if position change is needed.

        Args:
            action_tuple: (stop_loss_pct, take_profit_pct) or (None, None) for HOLD

        Returns:
            Dict with trade execution info
        """
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None}

        stop_loss_pct, take_profit_pct = action_tuple

        # HOLD action or already in position
        if action_tuple == (None, None) or self.current_position == 1:
            return trade_info

        # BUY with SL/TP bracket order
        if self.current_position == 0 and stop_loss_pct is not None and take_profit_pct is not None:
            amount = self._calculate_trade_amount("buy")

            # Get current price to calculate absolute SL/TP levels
            status = self.trader.get_status()
            # Use market data to get current price
            obs = self.observer.get_observations(return_base_ohlc=True)
            current_price = obs["base_features"][-1, 3]  # Close price

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
                    self.active_stop_loss = stop_loss_price
                    self.active_take_profit = take_profit_price

                trade_info.update({
                    "executed": True,
                    "amount": amount,
                    "side": "buy",
                    "success": success,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                })
            except Exception as e:
                print(f"Trade failed: buy ${amount:.2f} with SL={stop_loss_price:.2f}, TP={take_profit_price:.2f} - {str(e)}")
                trade_info["success"] = False

        return trade_info

    def _calculate_trade_amount(self, side: str) -> float:
        """Calculate the dollar amount to trade."""
        if self.config.trade_mode == TradeMode.QUANTITY:
            raise NotImplementedError("QUANTITY trade mode not implemented for SLTP env")

        if side == "buy":
            return self.balance
        else:
            return -1

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold

    def close(self):
        """Clean up resources."""
        self.trader.cancel_open_orders()
        self.trader.close_all_positions()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    config = AlpacaSLTPTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
        ],
        window_sizes=[15],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        stoploss_levels=(-0.02, -0.05),
        takeprofit_levels=(0.03, 0.06, 0.10),
    )

    env = AlpacaSLTPTorchTradingEnv(
        config,
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY")
    )

    print(f"Action space size: {env.action_spec.n}")
    print(f"Action map: {env.action_map}")

    td = env.reset()
    print(td)
