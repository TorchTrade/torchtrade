"""Market regime transform for context-aware trading.

This module provides a TorchRL transform that computes market regime indicators
from historical OHLCV data, enabling agents to learn context-dependent strategies.
"""

from typing import List, Optional, Tuple
import torch
from torchrl.envs.transforms import Transform
from torchrl.data import CompositeSpec, BoundedTensorSpec
from tensordict import TensorDictBase
import logging

logger = logging.getLogger(__name__)


class _MarketRegimeFeatures:
    """
    Internal class to compute market regime indicators from OHLCV data.

    Features computed:
    1. Volatility regime (0=low, 1=medium, 2=high)
    2. Trend regime (-1=down, 0=sideways, 1=up)
    3. Volume regime (0=low, 1=normal, 2=high)
    4. Price position (0=oversold, 1=neutral, 2=overbought)
    5. Continuous volatility value
    6. Continuous trend strength value
    7. Continuous volume ratio value
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
        """Compute regime features from price and volume history."""
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

        # Compute all regimes
        vol_regime, volatility = self._compute_volatility_regime(prices)
        trend_regime, trend_strength = self._compute_trend_regime(prices)
        volume_regime, volume_ratio = self._compute_volume_regime(volumes)
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
        """Compute volatility regime and continuous volatility value."""
        returns = torch.diff(prices) / prices[:-1]
        current_vol = returns[-self.vol_window:].std()

        if len(returns) >= self.vol_window:
            hist_vol = returns.unfold(0, self.vol_window, 1).std(dim=1)
            vol_percentile = (hist_vol < current_vol).float().mean()

            if vol_percentile < self.vol_low_threshold:
                vol_regime = 0  # Low volatility
            elif vol_percentile < self.vol_high_threshold:
                vol_regime = 1  # Medium volatility
            else:
                vol_regime = 2  # High volatility
        else:
            vol_regime = 1  # Default to medium

        return vol_regime, current_vol

    def _compute_trend_regime(self, prices: torch.Tensor) -> tuple:
        """Compute trend regime and continuous trend strength value."""
        ma_short = prices[-self.trend_short_window:].mean()
        ma_long = prices[-self.trend_window:].mean()
        trend_strength = (ma_short - ma_long) / ma_long

        if trend_strength > self.trend_up_threshold:
            trend_regime = 1  # Uptrend
        elif trend_strength < self.trend_down_threshold:
            trend_regime = -1  # Downtrend
        else:
            trend_regime = 0  # Sideways

        return trend_regime, trend_strength

    def _compute_volume_regime(self, volumes: torch.Tensor) -> tuple:
        """Compute volume regime and continuous volume ratio value."""
        current_volume = volumes[-1]
        avg_volume = volumes[-self.volume_window:].mean()

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
        else:
            volume_ratio = torch.tensor(1.0)

        if volume_ratio < self.vol_ratio_low:
            volume_regime = 0  # Low volume
        elif volume_ratio > self.vol_ratio_high:
            volume_regime = 2  # High volume
        else:
            volume_regime = 1  # Normal volume

        return volume_regime, volume_ratio

    def _compute_price_position_regime(self, prices: torch.Tensor) -> int:
        """Compute price position regime (oversold/neutral/overbought)."""
        window_size = min(self.price_position_window, len(prices))
        recent_prices = prices[-window_size:]

        current_price = prices[-1]
        high_price = recent_prices.max()
        low_price = recent_prices.min()

        if high_price > low_price:
            price_position = (current_price - low_price) / (high_price - low_price)
        else:
            price_position = 0.5  # No range, assume neutral

        if price_position < self.price_low_threshold:
            position_regime = 0  # Oversold
        elif price_position > self.price_high_threshold:
            position_regime = 2  # Overbought
        else:
            position_regime = 1  # Neutral

        return position_regime


class MarketRegimeTransform(Transform):
    """Transform that adds market regime features to observations.

    This transform computes 7 regime indicators from historical price/volume data:
    - Volatility regime (0=low, 1=medium, 2=high)
    - Trend regime (-1=down, 0=sideways, 1=up)
    - Volume regime (0=low, 1=normal, 2=high)
    - Price position (0=oversold, 1=neutral, 2=overbought)
    - Continuous volatility value
    - Continuous trend strength value
    - Continuous volume ratio value

    The transform queries the environment's sampler for historical OHLCV data
    and computes regime features, which are added to the observation.

    Usage:
        from torchrl.envs import TransformedEnv, Compose
        from torchtrade.envs.transforms import MarketRegimeTransform

        # Create environment with regime features
        env = TransformedEnv(
            base_env,
            Compose(
                MarketRegimeTransform(
                    sampler=base_env.sampler,
                    out_keys=["regime_features"],
                    volatility_window=20,
                    trend_window=50,
                ),
            )
        )

        # Regime features will be automatically added to observations
        td = env.reset()
        assert "regime_features" in td

    Args:
        sampler: MarketDataObservationSampler instance for accessing historical data
        out_keys: Output key for regime features (default: ["regime_features"])
        volatility_window: Window for volatility calculation (default: 20)
        trend_window: Window for long-term trend MA (default: 50)
        trend_short_window: Window for short-term trend MA (default: 20)
        volume_window: Window for volume analysis (default: 20)
        price_position_window: Window for price position (default: 252)
        volatility_thresholds: Percentile thresholds [low, high] (default: [0.33, 0.67])
        trend_thresholds: Percentage thresholds [down, up] (default: [-0.02, 0.02])
        volume_thresholds: Ratio thresholds [low, high] (default: [0.7, 1.3])
        price_position_thresholds: Position thresholds [oversold, overbought] (default: [0.33, 0.67])
    """

    def __init__(
        self,
        sampler,  # MarketDataObservationSampler
        out_keys: Optional[List[str]] = None,
        volatility_window: int = 20,
        trend_window: int = 50,
        trend_short_window: int = 20,
        volume_window: int = 20,
        price_position_window: int = 252,
        volatility_thresholds: Tuple[float, float] = (0.33, 0.67),
        trend_thresholds: Tuple[float, float] = (-0.02, 0.02),
        volume_thresholds: Tuple[float, float] = (0.7, 1.3),
        price_position_thresholds: Tuple[float, float] = (0.33, 0.67),
    ):
        """Initialize MarketRegimeTransform.

        Args:
            sampler: Sampler instance for accessing historical OHLCV data
            out_keys: Output keys for regime features
            volatility_window: Window for volatility calculation
            trend_window: Window for long-term MA
            trend_short_window: Window for short-term MA
            volume_window: Window for volume analysis
            price_position_window: Window for price position
            volatility_thresholds: Volatility regime thresholds
            trend_thresholds: Trend regime thresholds
            volume_thresholds: Volume regime thresholds
            price_position_thresholds: Price position thresholds
        """
        if out_keys is None:
            out_keys = ["regime_features"]

        # Initialize with empty in_keys since we don't transform existing keys
        super().__init__(in_keys=[], out_keys=out_keys)

        self.sampler = sampler

        # Initialize regime feature calculator
        self.regime_calculator = _MarketRegimeFeatures(
            volatility_window=volatility_window,
            trend_window=trend_window,
            trend_short_window=trend_short_window,
            volume_window=volume_window,
            price_position_window=price_position_window,
            volatility_thresholds=volatility_thresholds,
            trend_thresholds=trend_thresholds,
            volume_thresholds=volume_thresholds,
            price_position_thresholds=price_position_thresholds,
        )

        # Default regime features when insufficient data
        self.default_features = torch.tensor(
            [1, 0, 1, 1, 0.01, 0.0, 1.0],  # [vol=med, trend=sideways, vol=normal, pos=neutral, ...]
            dtype=torch.float32
        )

        # Cache for transformed spec
        self._transformed_spec = None

    def _compute_regime_features(self) -> torch.Tensor:
        """Compute regime features from sampler's historical data.

        Returns:
            Regime features tensor of shape (7,)
        """
        try:
            # Get historical data from sampler
            max_window = self.regime_calculator.min_data_required
            prices = self.sampler.get_recent_prices(window=max_window)
            volumes = self.sampler.get_recent_volumes(window=max_window)

            # Compute regime features
            features = self.regime_calculator.compute_features(prices, volumes)
            return features

        except ValueError as e:
            # Handle insufficient data at start of episodes
            if "Insufficient data" in str(e):
                logger.debug(f"Not enough data for regime features: {e}. Using defaults.")
                return self.default_features.clone()
            else:
                raise

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Add regime features to tensordict.

        Args:
            tensordict: Input tensordict from environment

        Returns:
            Tensordict with regime features added
        """
        # Compute regime features
        features = self._compute_regime_features()

        # Handle batched observations
        if tensordict.batch_size:
            # Expand features to match batch size
            batch_shape = tensordict.batch_size
            features = features.unsqueeze(0).expand(*batch_shape, -1)

        # Add to tensordict for each out_key
        for out_key in self.out_keys:
            tensordict.set(out_key, features)

        return tensordict

    forward = _call  # Alias for compatibility

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Apply transform to reset observations.

        Args:
            tensordict: Current tensordict (unused)
            tensordict_reset: Reset tensordict

        Returns:
            Transformed reset tensordict
        """
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        """Update observation spec with regime feature dimensions.

        Args:
            observation_spec: Original observation spec

        Returns:
            Updated spec with regime feature keys
        """
        if self._transformed_spec is not None:
            return self._transformed_spec

        spec = observation_spec.clone()

        # Add regime features spec for each out_key
        for out_key in self.out_keys:
            spec.set(
                out_key,
                BoundedTensorSpec(
                    low=-torch.inf,
                    high=torch.inf,
                    shape=(7,),  # 7 regime features
                    dtype=torch.float32,
                )
            )

        self._transformed_spec = spec
        return spec

    def to(self, dest) -> "MarketRegimeTransform":
        """Move transform to specified device or dtype.

        Args:
            dest: Target device or dtype

        Returns:
            Self for method chaining
        """
        # Move default features to new device
        if isinstance(dest, (torch.device, str)) and not isinstance(dest, torch.dtype):
            device = torch.device(dest) if isinstance(dest, str) else dest
            self.default_features = self.default_features.to(device)

        # Clear cached spec since device/dtype changed
        self._transformed_spec = None

        return super().to(dest)
