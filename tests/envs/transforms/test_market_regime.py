"""Tests for MarketRegimeTransform."""

import pytest
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec

from torchtrade.envs.transforms import MarketRegimeTransform


class TestMarketRegimeTransformInit:
    """Test MarketRegimeTransform initialization."""

    def test_init_basic(self):
        """Test basic initialization with default parameters."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"]
        )

        assert transform.price_feature_idx == 3  # Close price
        assert transform.volume_feature_idx == 4  # Volume
        assert transform.volatility_window == 20
        assert transform.trend_window == 50
        assert transform.volume_window == 20
        assert transform.position_window == 252
        assert transform.regime_features_dim == 7
        assert transform.out_keys == ["regime_features"]

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            price_feature_idx=0,
            volume_feature_idx=1,
            volatility_window=30,
            trend_window=100,
            volume_window=30,
            position_window=500,
        )

        assert transform.price_feature_idx == 0
        assert transform.volume_feature_idx == 1
        assert transform.volatility_window == 30
        assert transform.trend_window == 100
        assert transform.volume_window == 30
        assert transform.position_window == 500

    def test_init_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            vol_percentiles=[0.25, 0.75],
            trend_thresholds=[-0.01, 0.01],
            volume_thresholds=[0.5, 1.5],
            position_percentiles=[0.3, 0.7],
        )

        assert transform.vol_percentiles == [0.25, 0.75]
        assert transform.trend_thresholds == [-0.01, 0.01]
        assert transform.volume_thresholds == [0.5, 1.5]
        assert transform.position_percentiles == [0.3, 0.7]


class TestMarketRegimeTransformExtractPriceVolume:
    """Test _extract_price_volume method."""

    def test_extract_1d_input(self):
        """Test extraction from 1D price series."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        market_data = torch.randn(100)  # Just prices
        prices, volumes = transform._extract_price_volume(market_data)

        assert prices.shape == (100,)
        assert volumes.shape == (100,)
        assert torch.all(volumes == 1.0)  # Dummy volumes

    def test_extract_2d_input_ohlcv(self):
        """Test extraction from 2D OHLCV data."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            price_feature_idx=3,  # Close
            volume_feature_idx=4,  # Volume
        )

        # Create OHLCV data: (window_size, 5 features)
        market_data = torch.randn(100, 5)
        prices, volumes = transform._extract_price_volume(market_data)

        assert prices.shape == (100,)
        assert volumes.shape == (100,)
        assert torch.all(prices == market_data[:, 3])
        assert torch.all(volumes == market_data[:, 4])

    def test_extract_2d_input_missing_volume(self):
        """Test extraction when volume feature is missing."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            price_feature_idx=0,
            volume_feature_idx=4,
        )

        # Only 2 features (no volume)
        market_data = torch.randn(100, 2)
        prices, volumes = transform._extract_price_volume(market_data)

        assert prices.shape == (100,)
        assert volumes.shape == (100,)
        assert torch.all(volumes == 1.0)  # Dummy volumes

    def test_extract_invalid_shape(self):
        """Test that 3D input raises error."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        market_data = torch.randn(2, 100, 5)  # 3D invalid

        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            transform._extract_price_volume(market_data)


class TestMarketRegimeTransformComputeFeatures:
    """Test _compute_regime_features method."""

    def test_compute_features_insufficient_data(self):
        """Test behavior with insufficient data."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            volatility_window=20,
            trend_window=50,
        )

        # Only 10 data points
        prices = torch.randn(10) + 100
        volumes = torch.ones(10)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        # Should return neutral/default features
        assert features[0] == 1.0  # Medium volatility
        assert features[1] == 0.0  # Sideways
        assert features[2] == 1.0  # Normal volume
        assert features[3] == 1.0  # Neutral position

    def test_compute_features_low_volatility(self):
        """Test detection of low volatility regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            volatility_window=20,
        )

        # Create low volatility price series (small changes)
        prices = torch.linspace(100, 101, 100)  # Very gradual increase
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        # Low volatility should be detected
        assert features[0] <= 1.0  # Low or medium volatility

    def test_compute_features_high_volatility(self):
        """Test detection of high volatility regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            volatility_window=20,
        )

        # Create high volatility price series
        torch.manual_seed(42)
        prices = torch.randn(100) * 10 + 100  # Large swings
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        # Check that volatility was computed
        assert features[4] > 0  # Continuous volatility

    def test_compute_features_uptrend(self):
        """Test detection of uptrend regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            trend_window=50,
            trend_thresholds=[-0.02, 0.02],
        )

        # Create uptrend
        prices = torch.linspace(100, 120, 100)  # Strong uptrend
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[1] == 1.0  # Uptrend
        assert features[5] > 0.02  # Positive trend strength

    def test_compute_features_downtrend(self):
        """Test detection of downtrend regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            trend_window=50,
            trend_thresholds=[-0.02, 0.02],
        )

        # Create downtrend
        prices = torch.linspace(120, 100, 100)  # Strong downtrend
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[1] == -1.0  # Downtrend
        assert features[5] < -0.02  # Negative trend strength

    def test_compute_features_sideways(self):
        """Test detection of sideways regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            trend_window=50,
            trend_thresholds=[-0.02, 0.02],
        )

        # Create sideways market
        prices = torch.ones(100) * 100 + torch.randn(100) * 0.5  # Oscillating around 100
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        # Trend strength should be small
        assert abs(features[5]) < 0.1

    def test_compute_features_high_volume(self):
        """Test detection of high volume regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            volume_window=20,
            volume_thresholds=[0.7, 1.3],
        )

        prices = torch.linspace(100, 110, 100)
        # Normal volume, then spike
        volumes = torch.ones(100) * 1000
        volumes[-1] = 2000  # 2x volume spike

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[2] == 2.0  # High volume
        assert features[6] > 1.3  # High volume ratio

    def test_compute_features_low_volume(self):
        """Test detection of low volume regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            volume_window=20,
            volume_thresholds=[0.7, 1.3],
        )

        prices = torch.linspace(100, 110, 100)
        # Normal volume, then drop
        volumes = torch.ones(100) * 1000
        volumes[-1] = 500  # 0.5x volume drop

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[2] == 0.0  # Low volume
        assert features[6] < 0.7  # Low volume ratio

    def test_compute_features_oversold(self):
        """Test detection of oversold regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            position_window=100,
            position_percentiles=[0.33, 0.67],
        )

        # Create price series with current price at bottom
        prices = torch.linspace(120, 100, 100)  # Declining
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[3] == 0.0  # Oversold (at bottom of range)

    def test_compute_features_overbought(self):
        """Test detection of overbought regime."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            position_window=100,
            position_percentiles=[0.33, 0.67],
        )

        # Create price series with current price at top
        prices = torch.linspace(100, 120, 100)  # Rising
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert features[3] == 2.0  # Overbought (at top of range)


class TestMarketRegimeTransformApplyTransform:
    """Test _apply_transform method."""

    def test_apply_transform_1d(self):
        """Test transformation of 1D price series."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        market_data = torch.linspace(100, 110, 100)
        features = transform._apply_transform(market_data)

        assert features.shape == (7,)
        assert torch.all(torch.isfinite(features))

    def test_apply_transform_2d_ohlcv(self):
        """Test transformation of 2D OHLCV data."""
        transform = MarketRegimeTransform(
            in_keys=["market_data"],
            price_feature_idx=3,
            volume_feature_idx=4,
        )

        # Create OHLCV data
        market_data = torch.randn(100, 5) + 100
        features = transform._apply_transform(market_data)

        assert features.shape == (7,)
        assert torch.all(torch.isfinite(features))


class TestMarketRegimeTransformCall:
    """Test _call method (forward pass with tensordict)."""

    def test_call_unbatched(self):
        """Test forward pass with unbatched observation."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        td = TensorDict({
            "market_data": torch.randn(100, 5) + 100,  # (window, features)
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
        }, batch_size=[])

        td_out = transform._call(td)

        assert "regime_features" in td_out.keys()
        assert td_out["regime_features"].shape == (7,)
        assert "account_state" in td_out.keys()  # Preserved
        assert "market_data" in td_out.keys()  # Not deleted

    def test_call_batched(self):
        """Test forward pass with batched observations."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        # Batched: (batch_size, window, features)
        td = TensorDict({
            "market_data": torch.randn(4, 100, 5) + 100,
            "account_state": torch.randn(4, 7)
        }, batch_size=[4])

        td_out = transform._call(td)

        assert "regime_features" in td_out.keys()
        assert td_out["regime_features"].shape == (4, 7)
        assert "market_data" in td_out.keys()

    def test_call_missing_key(self):
        """Test that missing input key is skipped gracefully."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        td = TensorDict({
            "other_key": torch.tensor([1.0]),
        }, batch_size=[])

        # Should not raise error, just warn
        with pytest.warns(UserWarning, match="not found in tensordict"):
            td_out = transform._call(td)

        # Regime features should not be added
        assert "regime_features" not in td_out.keys()

    def test_call_multiple_in_keys(self):
        """Test with multiple input keys (uses first one)."""
        transform = MarketRegimeTransform(
            in_keys=["market_data_1", "market_data_2"]
        )

        td = TensorDict({
            "market_data_1": torch.randn(100, 5) + 100,
            "market_data_2": torch.randn(50, 5) + 100,
        }, batch_size=[])

        td_out = transform._call(td)

        assert "regime_features" in td_out.keys()
        assert td_out["regime_features"].shape == (7,)


class TestMarketRegimeTransformReset:
    """Test _reset method."""

    def test_reset_unbatched(self):
        """Test _reset applies transform to reset observations."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        td_reset = TensorDict({
            "market_data": torch.randn(100, 5) + 100,
            "account_state": torch.tensor([1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
        }, batch_size=[])

        td = TensorDict({}, batch_size=[])

        td_out = transform._reset(td, td_reset)

        assert "regime_features" in td_out.keys()
        assert td_out["regime_features"].shape == (7,)

    def test_reset_batched(self):
        """Test _reset with batched reset observations."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        td_reset = TensorDict({
            "market_data": torch.randn(4, 100, 5) + 100,
            "account_state": torch.randn(4, 7)
        }, batch_size=[4])

        td = TensorDict({}, batch_size=[4])

        td_out = transform._reset(td, td_reset)

        assert "regime_features" in td_out.keys()
        assert td_out["regime_features"].shape == (4, 7)


class TestMarketRegimeTransformObsSpec:
    """Test observation spec transformation."""

    def test_transform_observation_spec(self):
        """Test spec transformation adds regime_features."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        input_spec = CompositeSpec(
            market_data=BoundedTensorSpec(
                low=-10.0,
                high=10.0,
                shape=(100, 5),
                dtype=torch.float32
            ),
            account_state=BoundedTensorSpec(
                low=-torch.inf,
                high=torch.inf,
                shape=(7,),
                dtype=torch.float32
            )
        )

        output_spec = transform.transform_observation_spec(input_spec)

        assert "regime_features" in output_spec.keys()
        assert output_spec["regime_features"].shape == (7,)
        assert "market_data" in output_spec.keys()  # Not deleted
        assert "account_state" in output_spec.keys()  # Preserved

    def test_transform_observation_spec_caching(self):
        """Test that spec transformation is cached."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        input_spec = CompositeSpec(
            market_data=BoundedTensorSpec(
                low=-10.0,
                high=10.0,
                shape=(100, 5),
                dtype=torch.float32
            )
        )

        # First call
        spec1 = transform.transform_observation_spec(input_spec)
        # Second call should return cached version
        spec2 = transform.transform_observation_spec(input_spec)

        assert spec1 is spec2  # Same object (cached)


class TestMarketRegimeTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_price_range(self):
        """Test behavior when price range is zero."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        # Constant prices
        prices = torch.ones(100) * 100
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert torch.all(torch.isfinite(features))
        # Price position should be neutral (0.5 -> regime 1)
        assert features[3] == 1.0

    def test_near_zero_prices(self):
        """Test behavior with near-zero prices."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        # Very small prices (but positive)
        prices = torch.ones(100) * 0.0001
        volumes = torch.ones(100)

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert torch.all(torch.isfinite(features))

    def test_zero_volumes(self):
        """Test behavior with zero volumes."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        prices = torch.linspace(100, 110, 100)
        volumes = torch.zeros(100)
        volumes[-1] = 1.0  # Only last volume is non-zero

        features = transform._compute_regime_features(prices, volumes)

        assert features.shape == (7,)
        assert torch.all(torch.isfinite(features))

    def test_forward_alias(self):
        """Test that forward is an alias for _call."""
        transform = MarketRegimeTransform(in_keys=["market_data"])

        assert transform.forward is transform._call
