"""
Mean Reversion Actor using Bollinger Bands and Stochastic RSI.

This actor implements a mean reversion trading strategy that works best in ranging/sideways
markets. It uses Bollinger Bands to identify overbought/oversold conditions and confirms
entry signals with Stochastic RSI.
"""

from typing import Callable

import pandas as pd
import ta
from tensordict import TensorDict

from torchtrade.actor.rulebased.base import RuleBasedActor
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


class MeanReversionActor(RuleBasedActor):
    """
    Mean reversion strategy using Bollinger Bands and Stochastic RSI.

    **Strategy Logic:**
    - Uses Bollinger Bands to identify overbought/oversold conditions
    - Confirms with Stochastic RSI for entry signals
    - Buy when price near lower BB AND Stoch RSI oversold
    - Sell when price near upper BB AND Stoch RSI overbought
    - Hold otherwise

    **Expected Performance:**
    - Sharpe Ratio: 0.3 to 0.8
    - Action Distribution: 30% long, 30% short, 40% hold
    - Works best in: Ranging/sideways markets
    - Fails in: Strong trending markets

    **Preprocessing:**
    This actor requires preprocessing to compute features on the full dataset.
    The preprocessing function computes:
    - Bollinger Bands (middle, upper, lower, std, position)
    - Stochastic RSI (K and D lines)

    Args:
        bb_window: Bollinger Bands period (default: 20)
        bb_std: Bollinger Bands standard deviations (default: 2.0)
        stoch_rsi_window: Stochastic RSI period (default: 14)
        stoch_k_window: Stochastic %K smoothing (default: 3)
        stoch_d_window: Stochastic %D smoothing (default: 3)
        oversold_threshold: Stoch RSI oversold threshold (default: 20.0)
        overbought_threshold: Stoch RSI overbought threshold (default: 80.0)
        execute_timeframe: Timeframe to extract features from (default: "1Minute")
        **kwargs: Additional arguments passed to RuleBasedActor

    Example:
        >>> actor = MeanReversionActor(
        ...     market_data_keys=["market_data_1Minute_1"],
        ...     features_order=["open", "high", "low", "close", "volume",
        ...                     "features_bb_middle", "features_bb_std", "features_bb_upper",
        ...                     "features_bb_lower", "features_bb_position",
        ...                     "features_stoch_rsi_k", "features_stoch_rsi_d"],
        ...     bb_window=20,
        ...     oversold_threshold=20
        ... )
        >>> preprocessing_fn = actor.get_preprocessing_fn()
        >>> # Pass preprocessing_fn to environment
        >>> action = actor.select_action(observation)
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        stoch_rsi_window: int = 14,
        stoch_k_window: int = 3,
        stoch_d_window: int = 3,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
        volume_confirmation: float = 1.5,
        execute_timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.stoch_rsi_window = stoch_rsi_window
        self.stoch_k_window = stoch_k_window
        self.stoch_d_window = stoch_d_window
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.execute_timeframe = execute_timeframe

    def get_preprocessing_fn(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        Return preprocessing function that computes BB, Stoch RSI, and avg volume on full dataset.

        Added: Rolling average volume for entry confirmation.
        """

        def preprocess(df: pd.DataFrame) -> pd.DataFrame:
            # Bollinger Bands (computed on FULL dataset)
            df = df.copy().reset_index(drop=False)

            df["features_bb_middle"] = df["close"].rolling(self.bb_window).mean()
            df["features_bb_std"] = df["close"].rolling(self.bb_window).std()
            df["features_bb_upper"] = df["features_bb_middle"] + (self.bb_std * df["features_bb_std"])
            df["features_bb_lower"] = df["features_bb_middle"] - (self.bb_std * df["features_bb_std"])
            df["features_bb_position"] = (df["close"] - df["features_bb_lower"]) / (
                df["features_bb_upper"] - df["features_bb_lower"]
            )

            # Stochastic RSI (computed on FULL dataset)
            df["features_stoch_rsi_k"] = ta.momentum.stochrsi_k(
                df["close"],
                window=self.stoch_rsi_window,
                smooth1=self.stoch_k_window,
                smooth2=self.stoch_d_window
            )
            df["features_stoch_rsi_d"] = ta.momentum.stochrsi_d(
                df["close"],
                window=self.stoch_rsi_window,
                smooth1=self.stoch_k_window,
                smooth2=self.stoch_d_window
            )

            # Added: Rolling average volume (using bb_window for consistency)
            df["features_volume"] = df["volume"]
            df["features_avg_volume"] = df["volume"].rolling(self.bb_window).mean()

            # Fill NaN values from indicator warmup
            df.fillna(0, inplace=True)
            return df

        return preprocess

    def select_action(self, observation: TensorDict) -> int:
        """
        Select action based on mean reversion signals with crossover and volume confirmation.

        Adapted decision logic:
        1. BUY (action=2): Price below lower BB AND Stoch RSI bullish crossover from oversold AND volume confirmation AND (position <=0)
        - Opens long from flat or closes short.
        2. SELL (action=0): Price above upper BB AND Stoch RSI bearish crossover from overbought AND volume confirmation AND (position >=0)
        - Opens short from flat or closes long.
        3. HOLD (action=1): Otherwise

        Uses history from observation for crossover checks.
        """

        # Extract market data for execution timeframe
        market_data = self.extract_market_data(observation)
        key = f"{self.execute_timeframe.value}{self.execute_timeframe.unit.value}"
        data = market_data[key]  # Shape: [window_size, num_features]

        # Extract most recent feature values using get_feature() by name
        bb_position = self.get_feature(data, "features_bb_position")[-1].item()
        stoch_k_now = self.get_feature(data, "features_stoch_rsi_k")[-1].item()
        stoch_d_now = self.get_feature(data, "features_stoch_rsi_d")[-1].item()
        stoch_k_prev = self.get_feature(data, "features_stoch_rsi_k")[-2].item()
        stoch_d_prev = self.get_feature(data, "features_stoch_rsi_d")[-2].item()
        volume_now = self.get_feature(data, "features_volume")[-1].item()
        avg_volume = self.get_feature(data, "features_avg_volume")[-1].item()

        # Get account state
        account = observation["account_state"]
        position_size = account[1].item()

        if self.debug:
            print(f"BB Position: {bb_position:.4f}, Stoch RSI K: {stoch_k_now:.2f} (prev: {stoch_k_prev:.2f}), "
                f"D: {stoch_d_now:.2f} (prev: {stoch_d_prev:.2f}), Volume: {volume_now:.2f} (avg: {avg_volume:.2f}), "
                f"Position: {position_size:.2f}")

        # Volume confirmation (at least 50% above average)
        volume_confirmed = volume_now >= 1.5 * avg_volume

        # BUY/Long signal: Price below lower BB AND bullish Stoch RSI crossover from oversold AND volume
        is_long_signal = (
            bb_position < 0 and
            (stoch_k_now > stoch_d_now) and (stoch_k_prev <= stoch_d_prev) and (stoch_k_prev < self.oversold_threshold) and
            volume_confirmed
        )
        if is_long_signal and position_size <= 0:
            return 2  # BUY (open long or close short)

        # SELL/Short signal: Price above upper BB AND bearish Stoch RSI crossover from overbought AND volume
        is_short_signal = (
            bb_position > 1 and
            (stoch_k_now < stoch_d_now) and (stoch_k_prev >= stoch_d_prev) and (stoch_k_prev > self.overbought_threshold) and
            volume_confirmed
        )
        if is_short_signal and position_size >= 0:
            return 0  # SELL (open short or close long)

        return 1  # HOLD
