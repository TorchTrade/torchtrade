"""Market regime transform for context-aware trading.

This module provides a TorchRL transform that computes market regime features
from time series market data to enable context-dependent trading policies.
"""

from typing import List, Optional
import torch
from torchrl.envs.transforms import Transform
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDictBase
import warnings


class MarketRegimeTransform(Transform):
    """Transform that computes market regime features from price/volume data.

    Enables context-aware trading by extracting regime indicators:
    - Volatility regime (low/medium/high)
    - Trend regime (downtrend/sideways/uptrend)
    - Volume regime (low/normal/high)
    - Price position (oversold/neutral/overbought)

    These features help agents learn adaptive strategies for different market conditions.

    Available regime features (7 total):
        - vol_regime: Categorical (0=low, 1=medium, 2=high)
        - trend_regime: Categorical (-1=down, 0=sideways, 1=up)
        - volume_regime: Categorical (0=low, 1=normal, 2=high)
        - position_regime: Categorical (0=oversold, 1=neutral, 2=overbought)
        - volatility: Continuous volatility value
        - trend_strength: Continuous trend strength
        - volume_ratio: Continuous volume ratio

    Usage:
        from torchrl.envs import TransformedEnv, Compose
        from torchtrade.envs.transforms import MarketRegimeTransform

        # Create environment with regime features
        env = TransformedEnv(
            base_env,
            Compose(
                MarketRegimeTransform(
                    in_keys=["market_data_1Minute_12"],
                    price_feature_idx=3,  # Close price index in OHLCV
                    volume_feature_idx=4,  # Volume index in OHLCV
                ),
            )
        )

    Args:
        in_keys: Input keys containing market data (OHLCV observations)
        price_feature_idx: Index of price feature to use (default: 3 for close)
        volume_feature_idx: Index of volume feature (default: 4)
        volatility_window: Window for volatility calculation (default: 20)
        trend_window: Window for trend calculation (default: 50)
        volume_window: Window for volume calculation (default: 20)
        position_window: Window for price position calculation (default: 252)
        vol_percentiles: Percentile thresholds for volatility regimes (default: [0.33, 0.67])
        trend_thresholds: Thresholds for trend regimes (default: [-0.02, 0.02])
        volume_thresholds: Thresholds for volume regimes (default: [0.7, 1.3])
        position_percentiles: Percentile thresholds for price position (default: [0.33, 0.67])

    Attributes:
        regime_features_dim: Dimension of output regime features (always 7)
    """

    def __init__(
        self,
        in_keys: List[str],
        price_feature_idx: int = 3,  # Close price in OHLCV
        volume_feature_idx: int = 4,  # Volume in OHLCV
        volatility_window: int = 20,
        trend_window: int = 50,
        volume_window: int = 20,
        position_window: int = 252,
        vol_percentiles: List[float] = None,
        trend_thresholds: List[float] = None,
        volume_thresholds: List[float] = None,
        position_percentiles: List[float] = None,
    ):
        """Initialize MarketRegimeTransform.

        Args:
            in_keys: Market data keys to process (e.g., ["market_data_1Minute_12"])
            price_feature_idx: Index of price feature in market data
            volume_feature_idx: Index of volume feature in market data
            volatility_window: Lookback window for volatility
            trend_window: Lookback window for trend
            volume_window: Lookback window for volume
            position_window: Lookback window for price position
            vol_percentiles: Volatility regime thresholds
            trend_thresholds: Trend regime thresholds
            volume_thresholds: Volume regime thresholds
            position_percentiles: Price position thresholds
        """
        # Create out_key for regime features
        out_keys = ["regime_features"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.price_feature_idx = price_feature_idx
        self.volume_feature_idx = volume_feature_idx

        # Windows
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volume_window = volume_window
        self.position_window = position_window

        # Thresholds
        self.vol_percentiles = vol_percentiles or [0.33, 0.67]
        self.trend_thresholds = trend_thresholds or [-0.02, 0.02]
        self.volume_thresholds = volume_thresholds or [0.7, 1.3]
        self.position_percentiles = position_percentiles or [0.33, 0.67]

        # Output dimension is fixed (7 features)
        self.regime_features_dim = 7

        # Cache for observation specs
        self._transformed_spec = None

    def _compute_regime_features(
        self,
        prices: torch.Tensor,
        volumes: torch.Tensor
    ) -> torch.Tensor:
        """Compute regime features from price and volume series.

        Args:
            prices: Price series of shape (window_size,)
            volumes: Volume series of shape (window_size,)

        Returns:
            Regime features tensor of shape (7,)
        """
        # Ensure we have enough data
        if len(prices) < max(self.volatility_window, self.trend_window, self.volume_window):
            # Return neutral/zero features if not enough data
            return torch.tensor([1.0, 0.0, 1.0, 1.0, 0.01, 0.0, 1.0], dtype=torch.float32)

        # 1. Volatility Regime
        returns = torch.diff(prices) / (prices[:-1] + 1e-8)

        # Current volatility
        if len(returns) >= self.volatility_window:
            current_vol = returns[-self.volatility_window:].std()
        else:
            current_vol = returns.std()

        # Historical volatility percentile
        if len(returns) >= self.volatility_window * 2:
            # Compute rolling volatility
            vol_windows = returns.unfold(0, self.volatility_window, 1)
            hist_vol = vol_windows.std(dim=1)
            vol_percentile = (hist_vol < current_vol).float().mean()
        else:
            vol_percentile = torch.tensor(0.5)  # Default to medium

        # Classify volatility regime
        if vol_percentile < self.vol_percentiles[0]:
            vol_regime = 0.0  # Low
        elif vol_percentile < self.vol_percentiles[1]:
            vol_regime = 1.0  # Medium
        else:
            vol_regime = 2.0  # High

        # 2. Trend Regime
        if len(prices) >= self.trend_window:
            ma_short = prices[-20:].mean()
            ma_long = prices[-self.trend_window:].mean()
            trend_strength = (ma_short - ma_long) / (ma_long + 1e-8)
        else:
            trend_strength = torch.tensor(0.0)

        if trend_strength > self.trend_thresholds[1]:
            trend_regime = 1.0  # Uptrend
        elif trend_strength < self.trend_thresholds[0]:
            trend_regime = -1.0  # Downtrend
        else:
            trend_regime = 0.0  # Sideways

        # 3. Volume Regime
        if len(volumes) >= self.volume_window:
            current_volume = volumes[-1]
            avg_volume = volumes[-self.volume_window:].mean()
            volume_ratio = current_volume / (avg_volume + 1e-8)
        else:
            volume_ratio = torch.tensor(1.0)

        if volume_ratio < self.volume_thresholds[0]:
            volume_regime = 0.0  # Low
        elif volume_ratio > self.volume_thresholds[1]:
            volume_regime = 2.0  # High
        else:
            volume_regime = 1.0  # Normal

        # 4. Price Position
        if len(prices) >= self.position_window:
            position_prices = prices[-self.position_window:]
        else:
            position_prices = prices

        current_price = prices[-1]
        high_period = position_prices.max()
        low_period = position_prices.min()
        price_range = high_period - low_period

        if price_range > 1e-8:
            price_position = (current_price - low_period) / price_range
        else:
            price_position = torch.tensor(0.5)  # Neutral

        if price_position < self.position_percentiles[0]:
            position_regime = 0.0  # Oversold
        elif price_position > self.position_percentiles[1]:
            position_regime = 2.0  # Overbought
        else:
            position_regime = 1.0  # Neutral

        # Combine into feature vector
        regime_features = torch.tensor([
            vol_regime,
            trend_regime,
            volume_regime,
            position_regime,
            current_vol.item() if isinstance(current_vol, torch.Tensor) else current_vol,
            trend_strength.item() if isinstance(trend_strength, torch.Tensor) else trend_strength,
            volume_ratio.item() if isinstance(volume_ratio, torch.Tensor) else volume_ratio,
        ], dtype=torch.float32)

        return regime_features

    def _extract_price_volume(self, market_data: torch.Tensor) -> tuple:
        """Extract price and volume series from market data observation.

        Args:
            market_data: Market data tensor of shape (window_size, num_features)
                        where features are typically [open, high, low, close, volume, ...]

        Returns:
            Tuple of (prices, volumes) tensors of shape (window_size,)
        """
        if market_data.ndim == 1:
            # Single feature case - assume it's the price
            prices = market_data
            volumes = torch.ones_like(prices)  # Dummy volumes
        elif market_data.ndim == 2:
            # Multi-feature case
            prices = market_data[:, self.price_feature_idx]
            if market_data.shape[1] > self.volume_feature_idx:
                volumes = market_data[:, self.volume_feature_idx]
            else:
                volumes = torch.ones_like(prices)  # Dummy volumes
        else:
            raise ValueError(f"Expected 1D or 2D market data, got shape {market_data.shape}")

        return prices, volumes

    def _apply_transform(self, market_data: torch.Tensor) -> torch.Tensor:
        """Transform market data into regime features.

        Args:
            market_data: Market data tensor of shape (window_size,) or (window_size, num_features)

        Returns:
            Regime features tensor of shape (7,)
        """
        prices, volumes = self._extract_price_volume(market_data)
        return self._compute_regime_features(prices, volumes)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Process tensordict and add regime features.

        Args:
            tensordict: Input tensordict with market data observations

        Returns:
            Tensordict with regime_features added
        """
        # We'll use the first in_key to compute regime features
        # (typically there's one main market data observation)
        if not self.in_keys:
            return tensordict

        in_key = self.in_keys[0]
        if in_key not in tensordict.keys():
            warnings.warn(
                f"Key '{in_key}' not found in tensordict, skipping regime features",
                UserWarning
            )
            return tensordict

        market_data = tensordict.get(in_key)

        # Handle batched observations
        if market_data.ndim > 2:
            # Batched: flatten batch dimensions
            batch_shape = market_data.shape[:-2]
            market_data_flat = market_data.flatten(end_dim=-3)

            # Process batch
            regime_features_list = []
            for data in market_data_flat:
                features = self._apply_transform(data)
                regime_features_list.append(features)

            regime_features = torch.stack(regime_features_list)
            regime_features = regime_features.view(*batch_shape, -1)
        else:
            # Unbatched: single observation
            regime_features = self._apply_transform(market_data)

        # Add regime features to tensordict
        tensordict.set("regime_features", regime_features)

        return tensordict

    forward = _call  # Alias for compatibility

    def _reset(
        self,
        tensordict: TensorDictBase,
        tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Apply transform to reset observations."""
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        """Update observation spec with regime features.

        Args:
            observation_spec: Original observation spec

        Returns:
            Updated spec with regime_features key
        """
        if self._transformed_spec is not None:
            return self._transformed_spec

        spec = observation_spec.clone()

        # Add regime features spec
        spec.set(
            "regime_features",
            UnboundedContinuousTensorSpec(
                shape=(self.regime_features_dim,),
                dtype=torch.float32
            )
        )

        self._transformed_spec = spec
        return spec
