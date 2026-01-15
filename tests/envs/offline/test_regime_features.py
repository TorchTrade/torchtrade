"""
Tests for MarketRegimeTransform and internal regime feature calculations.
"""

import pytest
import torch
import numpy as np

from torchtrade.envs.transforms.market_regime import _MarketRegimeFeatures


@pytest.fixture
def regime_calculator():
    """Create a default regime calculator for testing."""
    return _MarketRegimeFeatures(
        volatility_window=20,
        trend_window=50,
        trend_short_window=20,
        volume_window=20,
        price_position_window=252,
    )


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    # Create 300 bars of price data with some variation
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 300)  # Mean ~0.1%, std ~2%
    prices = base_price * np.exp(np.cumsum(returns))
    return torch.tensor(prices, dtype=torch.float32)


@pytest.fixture
def sample_volumes():
    """Generate sample volume data for testing."""
    np.random.seed(42)
    # Generate random volumes around 1000 with variation
    volumes = np.random.uniform(500, 1500, 300)
    return torch.tensor(volumes, dtype=torch.float32)


@pytest.fixture
def trending_up_prices():
    """Generate uptrending price data."""
    # Strong uptrend
    x = torch.linspace(0, 300, 300)
    prices = 100 + 0.5 * x + torch.randn(300) * 2
    return prices


@pytest.fixture
def trending_down_prices():
    """Generate downtrending price data."""
    # Strong downtrend
    x = torch.linspace(0, 300, 300)
    prices = 200 - 0.5 * x + torch.randn(300) * 2
    return prices


@pytest.fixture
def sideways_prices():
    """Generate sideways/ranging price data."""
    # Sideways market around 100
    prices = 100 + torch.randn(300) * 3
    return prices


@pytest.fixture
def high_volatility_prices():
    """Generate high volatility price data."""
    np.random.seed(42)
    base_price = 100.0
    # Higher standard deviation returns
    returns = np.random.normal(0, 0.05, 300)  # 5% std
    prices = base_price * np.exp(np.cumsum(returns))
    return torch.tensor(prices, dtype=torch.float32)


@pytest.fixture
def low_volatility_prices():
    """Generate low volatility price data."""
    np.random.seed(42)
    base_price = 100.0
    # Lower standard deviation returns
    returns = np.random.normal(0, 0.005, 300)  # 0.5% std
    prices = base_price * np.exp(np.cumsum(returns))
    return torch.tensor(prices, dtype=torch.float32)


class Test_MarketRegimeFeatures:
    """Test suite for _MarketRegimeFeatures class."""

    def test_initialization(self):
        """Test that regime calculator initializes correctly."""
        calc = _MarketRegimeFeatures(
            volatility_window=20,
            trend_window=50,
            trend_short_window=20,
            volume_window=20,
            price_position_window=252,
        )
        assert calc.vol_window == 20
        assert calc.trend_window == 50
        assert calc.trend_short_window == 20
        assert calc.volume_window == 20
        assert calc.price_position_window == 252
        assert calc.min_data_required == 252  # max of all windows

    def test_feature_shape(self, regime_calculator, sample_prices, sample_volumes):
        """Test that computed features have correct shape."""
        features = regime_calculator.compute_features(sample_prices, sample_volumes)
        assert features.shape == (7,)
        assert features.dtype == torch.float32

    def test_feature_names(self, regime_calculator):
        """Test that features have the correct structure (7 features)."""
        # Internal class doesn't expose get_feature_names(), but we know the structure
        # Based on the docstring and compute_features implementation
        expected_features = 7
        sample_prices = torch.linspace(100, 110, 300)
        sample_volumes = torch.ones(300) * 1000
        features = regime_calculator.compute_features(sample_prices, sample_volumes)
        assert len(features) == expected_features

    def test_insufficient_data_error(self, regime_calculator):
        """Test that error is raised with insufficient data."""
        short_prices = torch.tensor([100.0, 101.0, 102.0])
        short_volumes = torch.tensor([1000.0, 1100.0, 1200.0])

        with pytest.raises(ValueError, match="Insufficient data"):
            regime_calculator.compute_features(short_prices, short_volumes)

    def test_mismatched_lengths_error(self, regime_calculator):
        """Test that error is raised when price and volume lengths don't match."""
        prices = torch.randn(300)
        volumes = torch.randn(250)  # Different length

        with pytest.raises(ValueError, match="same length"):
            regime_calculator.compute_features(prices, volumes)

    def test_volatility_regime_classification(self, regime_calculator, low_volatility_prices,
                                             high_volatility_prices, sample_volumes):
        """Test that volatility regime is correctly classified."""
        # Low volatility should give regime 0 or 1
        features_low_vol = regime_calculator.compute_features(
            low_volatility_prices, sample_volumes
        )
        vol_regime_low = int(features_low_vol[0].item())
        assert vol_regime_low in [0, 1], f"Low vol regime should be 0 or 1, got {vol_regime_low}"

        # High volatility should give regime 1 or 2
        features_high_vol = regime_calculator.compute_features(
            high_volatility_prices, sample_volumes
        )
        vol_regime_high = int(features_high_vol[0].item())
        assert vol_regime_high in [1, 2], f"High vol regime should be 1 or 2, got {vol_regime_high}"

    def test_trend_regime_classification(self, regime_calculator, trending_up_prices,
                                        trending_down_prices, sideways_prices, sample_volumes):
        """Test that trend regime is correctly classified."""
        # Uptrend should give regime 1
        features_up = regime_calculator.compute_features(trending_up_prices, sample_volumes)
        trend_regime_up = int(features_up[1].item())
        assert trend_regime_up == 1, f"Uptrend should give regime 1, got {trend_regime_up}"

        # Downtrend should give regime -1
        features_down = regime_calculator.compute_features(trending_down_prices, sample_volumes)
        trend_regime_down = int(features_down[1].item())
        assert trend_regime_down == -1, f"Downtrend should give regime -1, got {trend_regime_down}"

        # Sideways should give regime 0
        features_sideways = regime_calculator.compute_features(sideways_prices, sample_volumes)
        trend_regime_sideways = int(features_sideways[1].item())
        assert trend_regime_sideways == 0, f"Sideways should give regime 0, got {trend_regime_sideways}"

    def test_volume_regime_classification(self, regime_calculator, sample_prices):
        """Test that volume regime is correctly classified."""
        # Create volumes with clear patterns
        volumes_low = torch.ones(300) * 500  # Consistently low volume
        volumes_high = torch.cat([torch.ones(299) * 1000, torch.tensor([2000.0])])  # Spike at end

        # Low volume at the end
        features_low = regime_calculator.compute_features(sample_prices, volumes_low)
        vol_regime = int(features_low[2].item())
        assert vol_regime in [0, 1], f"Low volume should give regime 0 or 1, got {vol_regime}"

        # High volume spike at the end
        features_high = regime_calculator.compute_features(sample_prices, volumes_high)
        vol_regime_high = int(features_high[2].item())
        assert vol_regime_high == 2, f"High volume should give regime 2, got {vol_regime_high}"

    def test_price_position_regime(self, regime_calculator, sample_volumes):
        """Test that price position regime is correctly classified."""
        # Create prices at different positions in range
        base_prices = torch.linspace(100, 200, 300)

        # Price at bottom of range (oversold)
        prices_bottom = base_prices.clone()
        prices_bottom[-1] = 105  # Near the bottom
        features_bottom = regime_calculator.compute_features(prices_bottom, sample_volumes)
        position_regime = int(features_bottom[3].item())
        assert position_regime == 0, f"Bottom position should give regime 0, got {position_regime}"

        # Price at top of range (overbought)
        prices_top = base_prices.clone()
        prices_top[-1] = 195  # Near the top
        features_top = regime_calculator.compute_features(prices_top, sample_volumes)
        position_regime_top = int(features_top[3].item())
        assert position_regime_top == 2, f"Top position should give regime 2, got {position_regime_top}"

        # Price in middle (neutral)
        prices_mid = base_prices.clone()
        prices_mid[-1] = 150  # In the middle
        features_mid = regime_calculator.compute_features(prices_mid, sample_volumes)
        position_regime_mid = int(features_mid[3].item())
        assert position_regime_mid == 1, f"Mid position should give regime 1, got {position_regime_mid}"

    def test_continuous_features(self, regime_calculator, sample_prices, sample_volumes):
        """Test that continuous feature values are reasonable."""
        features = regime_calculator.compute_features(sample_prices, sample_volumes)

        # Volatility (index 4) should be positive
        volatility = features[4].item()
        assert volatility > 0, f"Volatility should be positive, got {volatility}"

        # Trend strength (index 5) should be a reasonable percentage
        trend_strength = features[5].item()
        assert -1 < trend_strength < 1, f"Trend strength should be reasonable, got {trend_strength}"

        # Volume ratio (index 6) should be positive
        volume_ratio = features[6].item()
        assert volume_ratio > 0, f"Volume ratio should be positive, got {volume_ratio}"

    def test_deterministic_output(self, regime_calculator, sample_prices, sample_volumes):
        """Test that features are deterministic for same input."""
        features1 = regime_calculator.compute_features(sample_prices, sample_volumes)
        features2 = regime_calculator.compute_features(sample_prices, sample_volumes)

        assert torch.allclose(features1, features2), "Features should be deterministic"

    def test_custom_thresholds(self):
        """Test that custom thresholds work correctly."""
        calc = _MarketRegimeFeatures(
            volatility_window=20,
            trend_window=50,
            trend_short_window=20,
            volume_window=20,
            price_position_window=252,
            volatility_thresholds=(0.25, 0.75),
            trend_thresholds=(-0.05, 0.05),
            volume_thresholds=(0.5, 1.5),
            price_position_thresholds=(0.2, 0.8),
        )

        assert calc.vol_low_threshold == 0.25
        assert calc.vol_high_threshold == 0.75
        assert calc.trend_down_threshold == -0.05
        assert calc.trend_up_threshold == 0.05
        assert calc.price_low_threshold == 0.2
        assert calc.price_high_threshold == 0.8

    def test_zero_volume_handling(self, regime_calculator, sample_prices):
        """Test that zero volumes are handled gracefully."""
        zero_volumes = torch.zeros(300)

        # Should not crash, should return default volume ratio of 1.0
        features = regime_calculator.compute_features(sample_prices, zero_volumes)
        volume_ratio = features[6].item()
        assert volume_ratio == 1.0, f"Zero volume should give ratio 1.0, got {volume_ratio}"

    def test_no_price_range_handling(self, regime_calculator, sample_volumes):
        """Test that flat prices (no range) are handled gracefully."""
        flat_prices = torch.ones(300) * 100.0  # All same price

        # Should not crash, position should be neutral (1)
        features = regime_calculator.compute_features(flat_prices, sample_volumes)
        position_regime = int(features[3].item())
        assert position_regime == 1, f"Flat prices should give neutral position, got {position_regime}"

    def test_with_minimal_required_data(self):
        """Test with exactly the minimum required data."""
        calc = _MarketRegimeFeatures(
            volatility_window=10,
            trend_window=20,
            trend_short_window=10,
            volume_window=10,
            price_position_window=30,  # This is the max, so min_data_required = 30
        )

        # Create exactly 30 bars
        prices = torch.linspace(100, 110, 30)
        volumes = torch.ones(30) * 1000

        # Should work without errors
        features = calc.compute_features(prices, volumes)
        assert features.shape == (7,)


class TestRegimeFeatureIntegration:
    """Integration tests for regime features with environment."""

    def test_integration_with_transformed_env(self):
        """Test that MarketRegimeTransform works with TransformedEnv."""
        # Import here to avoid circular dependency issues
        from torchtrade.envs.offline.futuresonestepenv import (
            FuturesOneStepEnv,
            FuturesOneStepEnvConfig,
        )
        from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
        from torchtrade.envs.transforms import MarketRegimeTransform
        from torchrl.envs import TransformedEnv
        import pandas as pd

        # Create sample data
        np.random.seed(42)
        n_bars = 500
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        volumes = np.random.uniform(500, 1500, n_bars)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes,
        })

        # Create base config (no regime features in config)
        config = FuturesOneStepEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
        )

        # Create base environment
        base_env = FuturesOneStepEnv(df, config)

        # Wrap with MarketRegimeTransform
        env = TransformedEnv(
            base_env,
            MarketRegimeTransform(
                sampler=base_env.sampler,
                out_keys=["regime_features"],
                volatility_window=20,
                trend_window=50,
                trend_short_window=20,
                volume_window=20,
                price_position_window=100,
            )
        )

        # Check that observation spec includes regime features
        assert "regime_features" in env.observation_spec
        assert env.observation_spec["regime_features"].shape == (7,)

        # Reset and get observation
        td = env.reset()

        # Check that observation contains regime features
        assert "regime_features" in td
        assert td["regime_features"].shape == (7,)

        # Features should be valid
        regime_features = td["regime_features"]
        assert not torch.isnan(regime_features).any(), "Regime features contain NaN"
        assert not torch.isinf(regime_features).any(), "Regime features contain Inf"

        # Discrete regime features should be in valid ranges
        vol_regime = int(regime_features[0].item())
        trend_regime = int(regime_features[1].item())
        volume_regime = int(regime_features[2].item())
        position_regime = int(regime_features[3].item())

        assert vol_regime in [0, 1, 2], f"Invalid vol regime: {vol_regime}"
        assert trend_regime in [-1, 0, 1], f"Invalid trend regime: {trend_regime}"
        assert volume_regime in [0, 1, 2], f"Invalid volume regime: {volume_regime}"
        assert position_regime in [0, 1, 2], f"Invalid position regime: {position_regime}"

    def test_env_without_transform(self):
        """Test that base environment works correctly without regime transform."""
        from torchtrade.envs.offline.futuresonestepenv import (
            FuturesOneStepEnv,
            FuturesOneStepEnvConfig,
        )
        from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
        import pandas as pd

        # Create sample data
        np.random.seed(42)
        n_bars = 500
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        volumes = np.random.uniform(500, 1500, n_bars)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes,
        })

        # Create config without regime features
        config = FuturesOneStepEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
        )

        # Create environment without transform
        env = FuturesOneStepEnv(df, config)

        # Check that observation spec does NOT include regime features
        assert "regime_features" not in env.observation_spec

        # Reset and get observation
        td = env.reset()

        # Check that observation does NOT contain regime features
        assert "regime_features" not in td
