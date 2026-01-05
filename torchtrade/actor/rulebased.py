"""
Rule-based trading actors for imitation learning pre-training.

These actors implement simple, profitable heuristics that can be used as "experts"
to generate demonstrations for behavioral cloning. Each actor follows a specific
trading strategy based on technical indicators.

References:
    Issue #54: https://github.com/TorchTrade/torchtrade_envs/issues/54
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
from tensordict import TensorDict


class RuleBasedActor(ABC):
    """
    Abstract base class for rule-based trading actors.

    Rule-based actors implement deterministic trading strategies using
    technical indicators computed from market data. They are designed to:

    1. Generate expert demonstrations for imitation learning
    2. Provide baselines for RL policy evaluation
    3. Bootstrap RL training with prior knowledge

    All rule-based actors must implement the `select_action` method which
    takes an observation TensorDict and returns an action index.

    Args:
        market_data_keys: List of market data keys to use from observation
                         (e.g., ["market_data_1Minute_12", "market_data_5Minute_8"])
        features_order: Order of features in market data tensor
                       (e.g., ["close", "open", "high", "low", "volume"])
        action_space_size: Number of discrete actions (default: 3 for sell/hold/buy)
        debug: Whether to print debug information (default: False)
    """

    def __init__(
        self,
        market_data_keys: Optional[List[str]] = None,
        features_order: Optional[List[str]] = None,
        action_space_size: int = 3,
        debug: bool = False,
    ):
        self.market_data_keys = market_data_keys or ["market_data_1Minute_12"]
        self.features_order = features_order or ["close", "open", "high", "low", "volume"]
        self.action_space_size = action_space_size
        self.debug = debug

        # Build feature index mapping
        self.feature_idx = {feat: i for i, feat in enumerate(self.features_order)}

    def extract_market_data(self, observation: TensorDict) -> Dict[str, torch.Tensor]:
        """
        Extract and organize market data from observation TensorDict.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            Dictionary mapping timeframe names to market data tensors

        Example:
            >>> data = actor.extract_market_data(obs)
            >>> prices = data["1Minute"][:, actor.feature_idx["close"]]
        """
        market_data = {}
        for key in self.market_data_keys:
            if key in observation.keys():
                # Extract timeframe name (e.g., "market_data_1Minute_12" -> "1Minute")
                timeframe_name = key.replace("market_data_", "").rsplit("_", 1)[0]
                data = observation[key]

                # Handle batch dimension if present
                if len(data.shape) == 3:  # (batch, window, features)
                    data = data.squeeze(0)

                market_data[timeframe_name] = data

        return market_data

    def get_feature(self, data: torch.Tensor, feature_name: str) -> torch.Tensor:
        """
        Extract a specific feature from market data tensor.

        Args:
            data: Market data tensor of shape (window_size, num_features)
            feature_name: Name of feature to extract (e.g., "close", "high")

        Returns:
            1D tensor of feature values across the window

        Example:
            >>> closes = actor.get_feature(data, "close")  # Shape: (window_size,)
        """
        if feature_name not in self.feature_idx:
            raise ValueError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {list(self.feature_idx.keys())}"
            )
        return data[:, self.feature_idx[feature_name]]

    @abstractmethod
    def select_action(self, observation: TensorDict) -> int:
        """
        Select an action based on the observation using rule-based strategy.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            Action index (0-based integer)

        Example:
            For 3-action space (SeqLongOnlyEnv):
                0 = Sell all
                1 = Hold (do nothing)
                2 = Buy all
        """
        pass

    def __call__(self, observation: TensorDict) -> TensorDict:
        """Make actor callable like a TorchRL policy."""
        return self.forward(observation)

    def forward(self, observation: TensorDict) -> TensorDict:
        """
        Process observation and set action in TensorDict.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            TensorDict with "action" key set
        """
        action = self.select_action(observation)

        if self.debug:
            print(f"{self.__class__.__name__} selected action: {action}")

        # Set action in observation TensorDict
        observation.set("action", torch.tensor([action], dtype=torch.long))

        return observation


class MomentumActor(RuleBasedActor):
    """
    Simple momentum trading strategy.

    **Strategy Logic:**
    - Calculate short-term momentum (price change over last N bars)
    - Calculate volatility (standard deviation of returns)
    - Go long when momentum is positive and volatility is moderate
    - Go short when momentum is negative and volatility is moderate
    - Hold when momentum is weak or volatility is too high

    **Expected Performance:**
    - Sharpe Ratio: 0.5 to 1.0
    - Action Distribution: 40% long, 40% short, 20% hold
    - Works best in: Trending markets
    - Fails in: Ranging/choppy markets

    Args:
        momentum_window: Number of bars to calculate momentum (default: 10)
        volatility_window: Number of bars to calculate volatility (default: 20)
        momentum_threshold: Minimum momentum to trigger trade (default: 0.01 = 1%)
        volatility_threshold: Maximum volatility to allow trade (default: 0.02 = 2%)
        **kwargs: Additional arguments passed to RuleBasedActor

    Example:
        >>> actor = MomentumActor(
        ...     market_data_keys=["market_data_5Minute_24"],
        ...     momentum_window=10,
        ...     volatility_threshold=0.015
        ... )
        >>> action = actor.select_action(observation)
    """

    def __init__(
        self,
        momentum_window: int = 10,
        volatility_window: int = 20,
        momentum_threshold: float = 0.01,
        volatility_threshold: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.momentum_threshold = momentum_threshold
        self.volatility_threshold = volatility_threshold

    def select_action(self, observation: TensorDict) -> int:
        """Select action based on momentum and volatility."""
        # Extract market data (use first timeframe if multiple)
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]  # Get first timeframe

        # Get close prices
        prices = self.get_feature(data, "close")

        # Calculate returns
        returns = torch.diff(prices) / prices[:-1]

        # Calculate momentum (mean return over momentum_window)
        momentum_window = min(self.momentum_window, len(returns))
        momentum = returns[-momentum_window:].mean().item()

        # Calculate volatility (std of returns over volatility_window)
        volatility_window = min(self.volatility_window, len(returns))
        volatility = returns[-volatility_window:].std().item()

        if self.debug:
            print(f"Momentum: {momentum:.4f}, Volatility: {volatility:.4f}")

        # Decision logic
        if momentum > self.momentum_threshold and volatility < self.volatility_threshold:
            return 2  # Buy (long)
        elif momentum < -self.momentum_threshold and volatility < self.volatility_threshold:
            return 0  # Sell (short)
        else:
            return 1  # Hold


class MeanReversionActor(RuleBasedActor):
    """
    Mean reversion trading strategy.

    **Strategy Logic:**
    - Calculate moving average of prices
    - Calculate deviation from moving average
    - Buy when price is significantly below MA (oversold)
    - Sell when price is significantly above MA (overbought)
    - Hold when price is near MA

    **Expected Performance:**
    - Sharpe Ratio: 0.3 to 0.8
    - Action Distribution: 30% long, 30% short, 40% hold
    - Works best in: Ranging/sideways markets
    - Fails in: Strong trending markets

    Args:
        ma_window: Number of bars for moving average (default: 20)
        deviation_threshold: Minimum deviation to trigger trade (default: 0.02 = 2%)
        **kwargs: Additional arguments passed to RuleBasedActor

    Example:
        >>> actor = MeanReversionActor(
        ...     market_data_keys=["market_data_15Minute_24"],
        ...     ma_window=30,
        ...     deviation_threshold=0.025
        ... )
        >>> action = actor.select_action(observation)
    """

    def __init__(
        self,
        ma_window: int = 20,
        deviation_threshold: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold

    def select_action(self, observation: TensorDict) -> int:
        """Select action based on mean reversion."""
        # Extract market data
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]

        # Get close prices
        prices = self.get_feature(data, "close")

        # Calculate moving average
        ma_window = min(self.ma_window, len(prices))
        ma = prices[-ma_window:].mean().item()

        # Current price
        current_price = prices[-1].item()

        # Calculate deviation from MA
        deviation = (current_price - ma) / ma

        if self.debug:
            print(f"Price: {current_price:.2f}, MA: {ma:.2f}, Deviation: {deviation:.4f}")

        # Decision logic
        if deviation < -self.deviation_threshold:
            return 2  # Oversold -> Buy
        elif deviation > self.deviation_threshold:
            return 0  # Overbought -> Sell
        else:
            return 1  # Hold


class BreakoutActor(RuleBasedActor):
    """
    Breakout/volatility expansion strategy using Bollinger Bands.

    **Strategy Logic:**
    - Calculate Bollinger Bands (MA ± 2*std)
    - Buy aggressively when price breaks above upper band
    - Sell aggressively when price breaks below lower band
    - Hold when price is within bands

    **Expected Performance:**
    - Sharpe Ratio: 0.2 to 1.5 (high variance)
    - Action Distribution: 25% long, 25% short, 50% hold
    - Works best in: Volatile markets with strong breakouts
    - Fails in: Many false breakouts

    Args:
        bb_window: Number of bars for Bollinger Bands (default: 20)
        bb_std: Number of standard deviations for bands (default: 2.0)
        **kwargs: Additional arguments passed to RuleBasedActor

    Example:
        >>> actor = BreakoutActor(
        ...     market_data_keys=["market_data_1Hour_24"],
        ...     bb_window=20,
        ...     bb_std=2.5
        ... )
        >>> action = actor.select_action(observation)
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bb_window = bb_window
        self.bb_std = bb_std

    def select_action(self, observation: TensorDict) -> int:
        """Select action based on Bollinger Band breakouts."""
        # Extract market data
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]

        # Get close prices
        prices = self.get_feature(data, "close")

        # Calculate Bollinger Bands
        bb_window = min(self.bb_window, len(prices))
        ma = prices[-bb_window:].mean().item()
        std = prices[-bb_window:].std().item()

        upper_band = ma + self.bb_std * std
        lower_band = ma - self.bb_std * std

        # Current price
        current_price = prices[-1].item()

        if self.debug:
            print(f"Price: {current_price:.2f}, Upper: {upper_band:.2f}, "
                  f"MA: {ma:.2f}, Lower: {lower_band:.2f}")

        # Decision logic
        if current_price > upper_band:
            return 2  # Breakout upward -> Buy
        elif current_price < lower_band:
            return 0  # Breakout downward -> Sell
        else:
            return 1  # Hold


# ============================================================================
# Stop-Loss/Take-Profit (SLTP) Environment Actors
# ============================================================================


class SLTPRuleBasedActor(RuleBasedActor):
    """
    Base class for rule-based actors in SLTP environments.

    SLTP environments have a combinatorial action space:
    - Action 0: HOLD (do nothing)
    - Actions 1...N: BUY with specific (stop_loss%, take_profit%) combinations

    This base class adds functionality to select appropriate SL/TP levels
    based on market conditions (e.g., volatility, confidence).

    Args:
        stoploss_levels: List of stop-loss levels (negative values, e.g., [-0.02, -0.05])
        takeprofit_levels: List of take-profit levels (positive values, e.g., [0.05, 0.10])
        **kwargs: Additional arguments passed to RuleBasedActor
    """

    def __init__(
        self,
        stoploss_levels: List[float] = None,
        takeprofit_levels: List[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stoploss_levels = stoploss_levels or [-0.025, -0.05, -0.1]
        self.takeprofit_levels = takeprofit_levels or [0.05, 0.1, 0.2]

        # Build action map (same as in SeqLongOnlySLTPEnv)
        from itertools import product
        self.action_map = {0: (None, None)}
        idx = 1
        for sl, tp in product(self.stoploss_levels, self.takeprofit_levels):
            self.action_map[idx] = (sl, tp)
            idx += 1

        self.action_space_size = len(self.action_map)

    def select_sltp_action(
        self, should_buy: bool, volatility: float, confidence: float
    ) -> int:
        """
        Select appropriate SL/TP action based on market conditions.

        Args:
            should_buy: Whether to enter a long position
            volatility: Current market volatility (higher = wider stops needed)
            confidence: Strategy confidence level (0.0 to 1.0, higher = tighter stops)

        Returns:
            Action index (0 for HOLD, 1...N for BUY with SL/TP)
        """
        if not should_buy:
            return 0  # HOLD

        # Select SL/TP based on volatility and confidence
        # Higher volatility -> wider stops
        # Higher confidence -> tighter stops (more aggressive)

        # Select stop loss (wider for high volatility, tighter for high confidence)
        if volatility > 0.03:  # High volatility
            sl_idx = len(self.stoploss_levels) - 1  # Widest stop
        elif confidence > 0.7:  # High confidence
            sl_idx = 0  # Tightest stop
        else:
            sl_idx = len(self.stoploss_levels) // 2  # Medium stop

        # Select take profit (higher for high confidence)
        if confidence > 0.7:
            tp_idx = len(self.takeprofit_levels) - 1  # Highest TP
        elif confidence < 0.3:
            tp_idx = 0  # Lowest TP (take profit quickly)
        else:
            tp_idx = len(self.takeprofit_levels) // 2  # Medium TP

        # Convert to action index
        sl = self.stoploss_levels[sl_idx]
        tp = self.takeprofit_levels[tp_idx]

        for action_idx, (sl_level, tp_level) in self.action_map.items():
            if sl_level == sl and tp_level == tp:
                return action_idx

        return 0  # Fallback to HOLD


class MomentumSLTPActor(SLTPRuleBasedActor, MomentumActor):
    """Momentum strategy adapted for SLTP environments."""

    def select_action(self, observation: TensorDict) -> int:
        """Select momentum-based action with appropriate SL/TP levels."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")
        returns = torch.diff(prices) / prices[:-1]

        momentum_window = min(self.momentum_window, len(returns))
        momentum = returns[-momentum_window:].mean().item()

        volatility_window = min(self.volatility_window, len(returns))
        volatility = returns[-volatility_window:].std().item()

        # Determine if we should buy
        should_buy = (
            momentum > self.momentum_threshold
            and volatility < self.volatility_threshold
        )

        # Calculate confidence based on strength of momentum
        confidence = min(abs(momentum) / self.momentum_threshold, 1.0)

        return self.select_sltp_action(should_buy, volatility, confidence)


class MeanReversionSLTPActor(SLTPRuleBasedActor, MeanReversionActor):
    """Mean reversion strategy adapted for SLTP environments."""

    def select_action(self, observation: TensorDict) -> int:
        """Select mean reversion action with appropriate SL/TP levels."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")

        ma_window = min(self.ma_window, len(prices))
        ma = prices[-ma_window:].mean().item()
        current_price = prices[-1].item()
        deviation = (current_price - ma) / ma

        # Determine if we should buy (oversold)
        should_buy = deviation < -self.deviation_threshold

        # Calculate volatility for SL/TP selection
        returns = torch.diff(prices) / prices[:-1]
        volatility = returns[-20:].std().item() if len(returns) >= 20 else 0.02

        # Confidence based on deviation magnitude
        confidence = min(abs(deviation) / self.deviation_threshold, 1.0)

        return self.select_sltp_action(should_buy, volatility, confidence)


class BreakoutSLTPActor(SLTPRuleBasedActor, BreakoutActor):
    """Breakout strategy adapted for SLTP environments."""

    def select_action(self, observation: TensorDict) -> int:
        """Select breakout action with appropriate SL/TP levels."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")

        bb_window = min(self.bb_window, len(prices))
        ma = prices[-bb_window:].mean().item()
        std = prices[-bb_window:].std().item()
        current_price = prices[-1].item()

        upper_band = ma + self.bb_std * std
        lower_band = ma - self.bb_std * std

        # Determine if we should buy (breakout upward)
        should_buy = current_price > upper_band

        # Use std as volatility measure
        volatility = std / ma  # Normalized volatility

        # Confidence based on how far above upper band
        if should_buy:
            confidence = min((current_price - upper_band) / (upper_band - ma), 1.0)
        else:
            confidence = 0.0

        return self.select_sltp_action(should_buy, volatility, confidence)


# ============================================================================
# Futures Environment Actors
# ============================================================================


class FuturesRuleBasedActor(RuleBasedActor):
    """
    Base class for rule-based actors in futures environments.

    Futures environments support both long and short positions:
    - Action 0: Go SHORT (or close long and open short)
    - Action 1: HOLD / Close position
    - Action 2: Go LONG (or close short and open long)

    Account state has 10 elements (vs 7 for spot):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]

    Position size is positive for longs, negative for shorts, zero for no position.
    """

    def __init__(self, **kwargs):
        kwargs["action_space_size"] = 3  # SHORT, HOLD, LONG
        super().__init__(**kwargs)

    def get_account_state(self, observation: TensorDict) -> torch.Tensor:
        """
        Extract account state from observation.

        Returns:
            Tensor of shape (10,) with account information
        """
        return observation.get("account_state", torch.zeros(10))

    def get_position_size(self, observation: TensorDict) -> float:
        """Get current position size (positive=long, negative=short, 0=flat)."""
        account_state = self.get_account_state(observation)
        return account_state[1].item()  # position_size is index 1


class MomentumFuturesActor(FuturesRuleBasedActor):
    """
    Momentum strategy adapted for futures environments.

    Goes long in uptrends, short in downtrends, flat otherwise.
    """

    def __init__(
        self,
        momentum_window: int = 10,
        volatility_window: int = 20,
        momentum_threshold: float = 0.01,
        volatility_threshold: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.momentum_threshold = momentum_threshold
        self.volatility_threshold = volatility_threshold

    def select_action(self, observation: TensorDict) -> int:
        """Select action: 0=short, 1=hold, 2=long."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")
        returns = torch.diff(prices) / prices[:-1]

        momentum_window = min(self.momentum_window, len(returns))
        momentum = returns[-momentum_window:].mean().item()

        volatility_window = min(self.volatility_window, len(returns))
        volatility = returns[-volatility_window:].std().item()

        if self.debug:
            print(f"Momentum: {momentum:.4f}, Volatility: {volatility:.4f}")

        # Futures: Can go long OR short
        if momentum > self.momentum_threshold and volatility < self.volatility_threshold:
            return 2  # Go LONG
        elif momentum < -self.momentum_threshold and volatility < self.volatility_threshold:
            return 0  # Go SHORT
        else:
            return 1  # HOLD / Close position


class MeanReversionFuturesActor(FuturesRuleBasedActor):
    """Mean reversion strategy for futures (fade extremes)."""

    def __init__(
        self,
        ma_window: int = 20,
        deviation_threshold: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold

    def select_action(self, observation: TensorDict) -> int:
        """Select action based on mean reversion."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")

        ma_window = min(self.ma_window, len(prices))
        ma = prices[-ma_window:].mean().item()
        current_price = prices[-1].item()
        deviation = (current_price - ma) / ma

        if self.debug:
            print(f"Price: {current_price:.2f}, MA: {ma:.2f}, Deviation: {deviation:.4f}")

        # Futures mean reversion: fade extremes
        if deviation < -self.deviation_threshold:
            return 2  # Oversold -> Go LONG
        elif deviation > self.deviation_threshold:
            return 0  # Overbought -> Go SHORT
        else:
            return 1  # HOLD


class BreakoutFuturesActor(FuturesRuleBasedActor):
    """Breakout strategy for futures with long/short capability."""

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bb_window = bb_window
        self.bb_std = bb_std

    def select_action(self, observation: TensorDict) -> int:
        """Select action based on Bollinger Band breakouts."""
        market_data = self.extract_market_data(observation)
        data = list(market_data.values())[0]
        prices = self.get_feature(data, "close")

        bb_window = min(self.bb_window, len(prices))
        ma = prices[-bb_window:].mean().item()
        std = prices[-bb_window:].std().item()

        upper_band = ma + self.bb_std * std
        lower_band = ma - self.bb_std * std
        current_price = prices[-1].item()

        if self.debug:
            print(f"Price: {current_price:.2f}, Upper: {upper_band:.2f}, "
                  f"MA: {ma:.2f}, Lower: {lower_band:.2f}")

        # Futures breakout: long above, short below
        if current_price > upper_band:
            return 2  # Breakout upward -> LONG
        elif current_price < lower_band:
            return 0  # Breakout downward -> SHORT
        else:
            return 1  # HOLD


# ============================================================================
# Convenience Functions
# ============================================================================


def create_expert_ensemble(
    market_data_keys: Optional[List[str]] = None,
    features_order: Optional[List[str]] = None,
    action_space_size: int = 3,
    env_type: str = "spot",
    stoploss_levels: Optional[List[float]] = None,
    takeprofit_levels: Optional[List[float]] = None,
) -> List[RuleBasedActor]:
    """
    Create an ensemble of all rule-based expert actors.

    This is useful for collecting diverse demonstrations from multiple
    expert strategies for imitation learning.

    Args:
        market_data_keys: List of market data keys to use
        features_order: Order of features in market data
        action_space_size: Number of discrete actions
        env_type: Type of environment ("spot", "sltp", or "futures")
        stoploss_levels: Stop-loss levels for SLTP envs (e.g., [-0.02, -0.05])
        takeprofit_levels: Take-profit levels for SLTP envs (e.g., [0.05, 0.10])

    Returns:
        List of rule-based actors appropriate for the environment type

    Example:
        >>> # For spot trading
        >>> experts = create_expert_ensemble(
        ...     market_data_keys=["market_data_5Minute_24"],
        ...     env_type="spot"
        ... )
        >>> # For SLTP trading
        >>> experts = create_expert_ensemble(
        ...     market_data_keys=["market_data_5Minute_24"],
        ...     env_type="sltp",
        ...     stoploss_levels=[-0.02, -0.05],
        ...     takeprofit_levels=[0.05, 0.10]
        ... )
        >>> # For futures trading
        >>> experts = create_expert_ensemble(
        ...     market_data_keys=["market_data_15Minute_24"],
        ...     env_type="futures"
        ... )
    """
    base_kwargs = {
        "market_data_keys": market_data_keys,
        "features_order": features_order,
        "action_space_size": action_space_size,
    }

    if env_type == "spot":
        return [
            MomentumActor(**base_kwargs),
            MeanReversionActor(**base_kwargs),
            BreakoutActor(**base_kwargs),
        ]
    elif env_type == "sltp":
        sltp_kwargs = {
            **base_kwargs,
            "stoploss_levels": stoploss_levels,
            "takeprofit_levels": takeprofit_levels,
        }
        return [
            MomentumSLTPActor(**sltp_kwargs),
            MeanReversionSLTPActor(**sltp_kwargs),
            BreakoutSLTPActor(**sltp_kwargs),
        ]
    elif env_type == "futures":
        return [
            MomentumFuturesActor(**base_kwargs),
            MeanReversionFuturesActor(**base_kwargs),
            BreakoutFuturesActor(**base_kwargs),
        ]
    else:
        raise ValueError(
            f"Unknown env_type: {env_type}. Must be 'spot', 'sltp', or 'futures'"
        )
