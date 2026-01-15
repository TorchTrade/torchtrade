"""
Market Regime Features for Context-Aware Trading.

Computes market condition indicators (volatility, trend, volume, price position)
to enable agents to learn context-dependent trading strategies.
"""

from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)


class MarketRegimeFeatures:
    """
    Compute market regime indicators from OHLCV data.

    Features computed:
    1. Volatility regime (0=low, 1=medium, 2=high)
    2. Trend regime (-1=down, 0=sideways, 1=up)
    3. Volume regime (0=low, 1=normal, 2=high)
    4. Price position (0=oversold, 1=neutral, 2=overbought)
    5. Continuous volatility value
    6. Continuous trend strength value
    7. Continuous volume ratio value

    Args:
        volatility_window: Window for volatility calculation (default: 20)
        trend_window: Window for long-term trend MA calculation (default: 50)
        trend_short_window: Window for short-term trend MA calculation (default: 20)
        volume_window: Window for volume analysis (default: 20)
        price_position_window: Window for price position (52-week ~= 252 daily bars, default: 252)
        volatility_thresholds: Percentile thresholds for vol regime [low, high] (default: [0.33, 0.67])
        trend_thresholds: Percentage thresholds for trend regime [down, up] (default: [-0.02, 0.02])
        volume_thresholds: Ratio thresholds for volume regime [low, high] (default: [0.7, 1.3])
        price_position_thresholds: Position thresholds [oversold, overbought] (default: [0.33, 0.67])
    """

    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        trend_short_window: int = 20,
        volume_window: int = 20,
        price_position_window: int = 252,
        volatility_thresholds: tuple = (0.33, 0.67),
        trend_thresholds: tuple = (-0.02, 0.02),
        volume_thresholds: tuple = (0.7, 1.3),
        price_position_thresholds: tuple = (0.33, 0.67),
    ):
        self.vol_window = volatility_window
        self.trend_window = trend_window
        self.trend_short_window = trend_short_window
        self.volume_window = volume_window
        self.price_position_window = price_position_window

        # Thresholds
        self.vol_low_threshold, self.vol_high_threshold = volatility_thresholds
        self.trend_down_threshold, self.trend_up_threshold = trend_thresholds
        self.vol_ratio_low, self.vol_ratio_high = volume_thresholds
        self.price_low_threshold, self.price_high_threshold = price_position_thresholds

        # Minimum data required for all features
        self.min_data_required = max(
            self.vol_window,
            self.trend_window,
            self.trend_short_window,
            self.volume_window,
            self.price_position_window
        )

    def compute_features(
        self,
        prices: torch.Tensor,
        volumes: torch.Tensor,
        highs: Optional[torch.Tensor] = None,
        lows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute regime features from price and volume history.

        Args:
            prices: Historical close prices (1D tensor, oldest to newest)
            volumes: Historical volumes (1D tensor, oldest to newest)
            highs: Optional high prices for more accurate calculations
            lows: Optional low prices for more accurate calculations

        Returns:
            7-element feature tensor:
            [vol_regime, trend_regime, volume_regime, position_regime,
             volatility, trend_strength, volume_ratio]

        Raises:
            ValueError: If insufficient data provided
        """
        # Validate input
        if len(prices) < self.min_data_required:
            raise ValueError(
                f"Insufficient data: need at least {self.min_data_required} bars, "
                f"got {len(prices)}"
            )

        if len(prices) != len(volumes):
            raise ValueError(
                f"Price and volume arrays must have same length: "
                f"prices={len(prices)}, volumes={len(volumes)}"
            )

        # Ensure tensors are float32
        prices = prices.float()
        volumes = volumes.float()

        # 1. Volatility Regime
        vol_regime, volatility = self._compute_volatility_regime(prices)

        # 2. Trend Regime
        trend_regime, trend_strength = self._compute_trend_regime(prices)

        # 3. Volume Regime
        volume_regime, volume_ratio = self._compute_volume_regime(volumes)

        # 4. Price Position Regime
        position_regime = self._compute_price_position_regime(prices)

        # Combine into feature vector
        regime_features = torch.tensor([
            float(vol_regime),
            float(trend_regime),
            float(volume_regime),
            float(position_regime),
            volatility.item(),
            trend_strength.item(),
            volume_ratio.item(),
        ], dtype=torch.float32)

        return regime_features

    def _compute_volatility_regime(self, prices: torch.Tensor) -> tuple:
        """
        Compute volatility regime and continuous volatility value.

        Returns:
            (regime: int, volatility: float)
            regime: 0=low, 1=medium, 2=high
        """
        # Calculate returns
        returns = torch.diff(prices) / prices[:-1]

        # Current volatility (std of recent returns)
        current_vol = returns[-self.vol_window:].std()

        # Historical volatility distribution (rolling windows)
        # Use unfold to create rolling windows efficiently
        if len(returns) >= self.vol_window:
            # Create rolling windows
            hist_vol = returns.unfold(0, self.vol_window, 1).std(dim=1)

            # Calculate percentile of current volatility
            vol_percentile = (hist_vol < current_vol).float().mean()

            # Classify regime
            if vol_percentile < self.vol_low_threshold:
                vol_regime = 0  # Low volatility
            elif vol_percentile < self.vol_high_threshold:
                vol_regime = 1  # Medium volatility
            else:
                vol_regime = 2  # High volatility
        else:
            # Not enough data for historical comparison, use medium as default
            vol_regime = 1

        return vol_regime, current_vol

    def _compute_trend_regime(self, prices: torch.Tensor) -> tuple:
        """
        Compute trend regime and continuous trend strength value.

        Returns:
            (regime: int, trend_strength: float)
            regime: -1=downtrend, 0=sideways, 1=uptrend
        """
        # Short-term and long-term moving averages
        ma_short = prices[-self.trend_short_window:].mean()
        ma_long = prices[-self.trend_window:].mean()

        # Trend strength as percentage difference
        trend_strength = (ma_short - ma_long) / ma_long

        # Classify regime
        if trend_strength > self.trend_up_threshold:
            trend_regime = 1  # Uptrend
        elif trend_strength < self.trend_down_threshold:
            trend_regime = -1  # Downtrend
        else:
            trend_regime = 0  # Sideways

        return trend_regime, trend_strength

    def _compute_volume_regime(self, volumes: torch.Tensor) -> tuple:
        """
        Compute volume regime and continuous volume ratio value.

        Returns:
            (regime: int, volume_ratio: float)
            regime: 0=low, 1=normal, 2=high
        """
        # Current volume vs average volume
        current_volume = volumes[-1]
        avg_volume = volumes[-self.volume_window:].mean()

        # Avoid division by zero
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
        else:
            volume_ratio = torch.tensor(1.0)

        # Classify regime
        if volume_ratio < self.vol_ratio_low:
            volume_regime = 0  # Low volume
        elif volume_ratio > self.vol_ratio_high:
            volume_regime = 2  # High volume
        else:
            volume_regime = 1  # Normal volume

        return volume_regime, volume_ratio

    def _compute_price_position_regime(self, prices: torch.Tensor) -> int:
        """
        Compute price position regime (oversold/neutral/overbought).

        Returns:
            regime: 0=oversold, 1=neutral, 2=overbought
        """
        # Current price position relative to recent range
        window_size = min(self.price_position_window, len(prices))
        recent_prices = prices[-window_size:]

        current_price = prices[-1]
        high_price = recent_prices.max()
        low_price = recent_prices.min()

        # Calculate position (0 to 1)
        if high_price > low_price:
            price_position = (current_price - low_price) / (high_price - low_price)
        else:
            # No range, assume neutral
            price_position = 0.5

        # Classify regime using configurable thresholds
        if price_position < self.price_low_threshold:
            position_regime = 0  # Oversold
        elif price_position > self.price_high_threshold:
            position_regime = 2  # Overbought
        else:
            position_regime = 1  # Neutral

        return position_regime

    def get_feature_names(self) -> list:
        """
        Get names of the 7 regime features.

        Returns:
            List of feature names
        """
        return [
            "vol_regime",       # 0=low, 1=med, 2=high
            "trend_regime",     # -1=down, 0=sideways, 1=up
            "volume_regime",    # 0=low, 1=normal, 2=high
            "position_regime",  # 0=oversold, 1=neutral, 2=overbought
            "volatility",       # Continuous value
            "trend_strength",   # Continuous value
            "volume_ratio",     # Continuous value
        ]
