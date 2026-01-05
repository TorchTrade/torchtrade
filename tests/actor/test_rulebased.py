"""Tests for rule-based trading actors."""

import pytest
import torch
from tensordict import TensorDict

from torchtrade.actor import (
    MomentumActor,
    MeanReversionActor,
    BreakoutActor,
    MomentumSLTPActor,
    MeanReversionSLTPActor,
    BreakoutSLTPActor,
    MomentumFuturesActor,
    MeanReversionFuturesActor,
    BreakoutFuturesActor,
    create_expert_ensemble,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def market_data_trending():
    """Create trending market data (prices going up)."""
    # Create 50 bars of upward trending prices
    prices = torch.linspace(100, 110, 50)
    noise = torch.randn(50) * 0.1
    close = prices + noise

    # OHLCV data
    open_prices = close + torch.randn(50) * 0.05
    high = torch.maximum(close, open_prices) + torch.rand(50) * 0.1
    low = torch.minimum(close, open_prices) - torch.rand(50) * 0.1
    volume = torch.rand(50) * 1000 + 500

    # Shape: (50, 5) -> (window_size, features)
    data = torch.stack([close, open_prices, high, low, volume], dim=1)
    return data


@pytest.fixture
def market_data_ranging():
    """Create ranging market data (sideways movement)."""
    # Create 50 bars of sideways movement around 100
    close = 100 + torch.randn(50) * 0.5

    open_prices = close + torch.randn(50) * 0.3
    high = torch.maximum(close, open_prices) + torch.rand(50) * 0.2
    low = torch.minimum(close, open_prices) - torch.rand(50) * 0.2
    volume = torch.rand(50) * 1000 + 500

    data = torch.stack([close, open_prices, high, low, volume], dim=1)
    return data


@pytest.fixture
def observation_spot(market_data_trending):
    """Create observation TensorDict for spot environment."""
    return TensorDict({
        "market_data_5Minute_50": market_data_trending.unsqueeze(0),  # (1, 50, 5)
        "account_state": torch.tensor([[1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]]),
    }, batch_size=[])


@pytest.fixture
def observation_futures(market_data_trending):
    """Create observation TensorDict for futures environment."""
    return TensorDict({
        "market_data_5Minute_50": market_data_trending.unsqueeze(0),
        "account_state": torch.tensor([[1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 10.0, 0.1, 90.0, 0.0]]),
    }, batch_size=[])


# ============================================================================
# Tests for Spot Trading Actors
# ============================================================================


class TestMomentumActor:
    """Tests for MomentumActor."""

    def test_init(self):
        """Test actor initialization."""
        actor = MomentumActor(
            market_data_keys=["market_data_5Minute_50"],
            momentum_window=10,
            volatility_window=20,
        )
        assert actor.momentum_window == 10
        assert actor.volatility_window == 20
        assert actor.action_space_size == 3

    def test_select_action_trending_up(self, observation_spot):
        """Test action selection in uptrend."""
        actor = MomentumActor(
            market_data_keys=["market_data_5Minute_50"],
            momentum_threshold=0.001,  # Low threshold for test
        )
        action = actor.select_action(observation_spot)

        # Expect BUY (action 2) in uptrend
        assert isinstance(action, int)
        assert 0 <= action <= 2
        # Most likely action 2 (buy) due to positive momentum
        assert action == 2

    def test_select_action_high_volatility(self, observation_spot):
        """Test action selection with high volatility threshold."""
        actor = MomentumActor(
            market_data_keys=["market_data_5Minute_50"],
            volatility_threshold=0.001,  # Very low threshold
        )
        action = actor.select_action(observation_spot)

        # Should hold due to high volatility
        assert action == 1

    def test_forward(self, observation_spot):
        """Test forward method sets action in TensorDict."""
        actor = MomentumActor(market_data_keys=["market_data_5Minute_50"])
        result = actor.forward(observation_spot)

        assert "action" in result.keys()
        assert result["action"].shape == (1,)
        assert result["action"].dtype == torch.long

    def test_call(self, observation_spot):
        """Test __call__ method."""
        actor = MomentumActor(market_data_keys=["market_data_5Minute_50"])
        result = actor(observation_spot)

        assert "action" in result.keys()


class TestMeanReversionActor:
    """Tests for MeanReversionActor."""

    def test_init(self):
        """Test actor initialization."""
        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            ma_window=20,
            deviation_threshold=0.02,
        )
        assert actor.ma_window == 20
        assert actor.deviation_threshold == 0.02

    def test_select_action_oversold(self):
        """Test action selection when price is oversold."""
        # Create data where current price is below MA
        prices = torch.ones(50) * 100
        prices[-1] = 95  # Last price significantly lower

        data = prices.unsqueeze(1).repeat(1, 5)  # (50, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            deviation_threshold=0.02,
        )
        action = actor.select_action(obs)

        # Expect BUY (action 2) when oversold
        assert action == 2

    def test_select_action_overbought(self):
        """Test action selection when price is overbought."""
        prices = torch.ones(50) * 100
        prices[-1] = 105  # Last price significantly higher

        data = prices.unsqueeze(1).repeat(1, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        actor = MeanReversionActor(
            market_data_keys=["market_data_5Minute_50"],
            deviation_threshold=0.02,
        )
        action = actor.select_action(obs)

        # Expect SELL (action 0) when overbought
        assert action == 0


class TestBreakoutActor:
    """Tests for BreakoutActor."""

    def test_init(self):
        """Test actor initialization."""
        actor = BreakoutActor(
            market_data_keys=["market_data_5Minute_50"],
            bb_window=20,
            bb_std=2.0,
        )
        assert actor.bb_window == 20
        assert actor.bb_std == 2.0

    def test_select_action_breakout_upward(self):
        """Test action selection on upward breakout."""
        # Create data with breakout above upper band
        prices = torch.ones(50) * 100
        prices[-1] = 110  # Breakout price

        data = prices.unsqueeze(1).repeat(1, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        actor = BreakoutActor(
            market_data_keys=["market_data_5Minute_50"],
            bb_std=2.0,
        )
        action = actor.select_action(obs)

        # Expect BUY (action 2) on upward breakout
        assert action == 2

    def test_select_action_within_bands(self):
        """Test action selection when price is within bands."""
        prices = torch.ones(50) * 100

        data = prices.unsqueeze(1).repeat(1, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        actor = BreakoutActor(market_data_keys=["market_data_5Minute_50"])
        action = actor.select_action(obs)

        # Expect HOLD (action 1) within bands
        assert action == 1


# ============================================================================
# Tests for SLTP Actors
# ============================================================================


class TestSLTPActors:
    """Tests for SLTP environment actors."""

    def test_momentum_sltp_init(self):
        """Test MomentumSLTPActor initialization."""
        actor = MomentumSLTPActor(
            market_data_keys=["market_data_5Minute_50"],
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.05, 0.10],
        )
        assert actor.stoploss_levels == [-0.02, -0.05]
        assert actor.takeprofit_levels == [0.05, 0.10]
        # Action map: 0=HOLD + 2*2=4 SL/TP combos = 5 total actions
        assert actor.action_space_size == 5

    def test_action_map_creation(self):
        """Test that action map is created correctly."""
        actor = MomentumSLTPActor(
            market_data_keys=["market_data_5Minute_50"],
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.05, 0.10],
        )

        assert actor.action_map[0] == (None, None)  # HOLD
        assert actor.action_map[1] == (-0.02, 0.05)
        assert actor.action_map[2] == (-0.02, 0.10)
        assert actor.action_map[3] == (-0.05, 0.05)
        assert actor.action_map[4] == (-0.05, 0.10)

    def test_momentum_sltp_select_action(self, observation_spot):
        """Test MomentumSLTPActor action selection."""
        actor = MomentumSLTPActor(
            market_data_keys=["market_data_5Minute_50"],
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.05, 0.10],
            momentum_threshold=0.001,
        )
        action = actor.select_action(observation_spot)

        # Action should be in valid range
        assert 0 <= action < 5

    def test_mean_reversion_sltp(self, observation_spot):
        """Test MeanReversionSLTPActor."""
        actor = MeanReversionSLTPActor(
            market_data_keys=["market_data_5Minute_50"],
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
        )
        action = actor.select_action(observation_spot)

        # Should return valid action (0=HOLD or 1=BUY with SL/TP)
        assert action in [0, 1]

    def test_breakout_sltp(self, observation_spot):
        """Test BreakoutSLTPActor."""
        actor = BreakoutSLTPActor(
            market_data_keys=["market_data_5Minute_50"],
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.05, 0.10],
        )
        action = actor.select_action(observation_spot)

        assert 0 <= action < 5


# ============================================================================
# Tests for Futures Actors
# ============================================================================


class TestFuturesActors:
    """Tests for futures environment actors."""

    def test_momentum_futures_init(self):
        """Test MomentumFuturesActor initialization."""
        actor = MomentumFuturesActor(
            market_data_keys=["market_data_5Minute_50"],
        )
        assert actor.action_space_size == 3  # SHORT, HOLD, LONG

    def test_momentum_futures_trending_up(self, observation_futures):
        """Test futures actor in uptrend."""
        actor = MomentumFuturesActor(
            market_data_keys=["market_data_5Minute_50"],
            momentum_threshold=0.001,
        )
        action = actor.select_action(observation_futures)

        # Expect LONG (action 2) in uptrend
        assert action == 2

    def test_mean_reversion_futures_oversold(self):
        """Test futures mean reversion on oversold condition."""
        prices = torch.ones(50) * 100
        prices[-1] = 95

        data = prices.unsqueeze(1).repeat(1, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 10),  # 10 elements for futures
        }, batch_size=[])

        actor = MeanReversionFuturesActor(
            market_data_keys=["market_data_5Minute_50"],
            deviation_threshold=0.02,
        )
        action = actor.select_action(obs)

        # Expect LONG (action 2) when oversold
        assert action == 2

    def test_mean_reversion_futures_overbought(self):
        """Test futures mean reversion on overbought condition."""
        prices = torch.ones(50) * 100
        prices[-1] = 105

        data = prices.unsqueeze(1).repeat(1, 5)

        obs = TensorDict({
            "market_data_5Minute_50": data.unsqueeze(0),
            "account_state": torch.zeros(1, 10),
        }, batch_size=[])

        actor = MeanReversionFuturesActor(
            market_data_keys=["market_data_5Minute_50"],
            deviation_threshold=0.02,
        )
        action = actor.select_action(obs)

        # Expect SHORT (action 0) when overbought
        assert action == 0

    def test_breakout_futures(self, observation_futures):
        """Test BreakoutFuturesActor."""
        actor = BreakoutFuturesActor(
            market_data_keys=["market_data_5Minute_50"],
        )
        action = actor.select_action(observation_futures)

        # Should return valid action (0=SHORT, 1=HOLD, 2=LONG)
        assert action in [0, 1, 2]


# ============================================================================
# Tests for Ensemble Creation
# ============================================================================


class TestExpertEnsemble:
    """Tests for expert ensemble creation."""

    def test_create_spot_ensemble(self):
        """Test creating spot trading ensemble."""
        experts = create_expert_ensemble(
            market_data_keys=["market_data_5Minute_50"],
            env_type="spot",
        )

        assert len(experts) == 3
        assert isinstance(experts[0], MomentumActor)
        assert isinstance(experts[1], MeanReversionActor)
        assert isinstance(experts[2], BreakoutActor)

    def test_create_sltp_ensemble(self):
        """Test creating SLTP ensemble."""
        experts = create_expert_ensemble(
            market_data_keys=["market_data_5Minute_50"],
            env_type="sltp",
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.05, 0.10],
        )

        assert len(experts) == 3
        assert isinstance(experts[0], MomentumSLTPActor)
        assert isinstance(experts[1], MeanReversionSLTPActor)
        assert isinstance(experts[2], BreakoutSLTPActor)

    def test_create_futures_ensemble(self):
        """Test creating futures ensemble."""
        experts = create_expert_ensemble(
            market_data_keys=["market_data_5Minute_50"],
            env_type="futures",
        )

        assert len(experts) == 3
        assert isinstance(experts[0], MomentumFuturesActor)
        assert isinstance(experts[1], MeanReversionFuturesActor)
        assert isinstance(experts[2], BreakoutFuturesActor)

    def test_invalid_env_type(self):
        """Test error handling for invalid environment type."""
        with pytest.raises(ValueError, match="Unknown env_type"):
            create_expert_ensemble(env_type="invalid")

    def test_ensemble_with_same_observation(self, observation_spot):
        """Test that all experts in ensemble can process same observation."""
        experts = create_expert_ensemble(
            market_data_keys=["market_data_5Minute_50"],
            env_type="spot",
        )

        for expert in experts:
            result = expert(observation_spot.clone())
            assert "action" in result.keys()
            assert 0 <= result["action"].item() <= 2


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_short_window(self):
        """Test with window shorter than requested."""
        # Only 5 bars of data
        data = torch.randn(5, 5) + 100

        obs = TensorDict({
            "market_data_5Minute_5": data.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        actor = MomentumActor(
            market_data_keys=["market_data_5Minute_5"],
            momentum_window=10,  # Larger than available data
        )
        action = actor.select_action(obs)

        # Should handle gracefully
        assert 0 <= action <= 2

    def test_feature_extraction(self, market_data_trending):
        """Test feature extraction from market data."""
        actor = MomentumActor(market_data_keys=["market_data_5Minute_50"])

        obs = TensorDict({
            "market_data_5Minute_50": market_data_trending.unsqueeze(0),
            "account_state": torch.zeros(1, 7),
        }, batch_size=[])

        market_data = actor.extract_market_data(obs)

        assert "5Minute" in market_data
        assert market_data["5Minute"].shape == (50, 5)

        # Test getting specific feature
        closes = actor.get_feature(market_data["5Minute"], "close")
        assert closes.shape == (50,)

    def test_debug_mode(self, observation_spot, capsys):
        """Test debug mode prints information."""
        actor = MomentumActor(
            market_data_keys=["market_data_5Minute_50"],
            debug=True,
        )

        actor.forward(observation_spot)
        captured = capsys.readouterr()

        # Should print debug info
        assert "MomentumActor" in captured.out or "Momentum" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with actual environments."""

    def test_with_seqlongonly_env(self):
        """Test actors with SeqLongOnlyEnv."""
        pytest.importorskip("pandas")

        from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
        from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
        import pandas as pd

        # Create minimal OHLCV data
        df = pd.DataFrame({
            "0": pd.date_range("2020-01-01", periods=1000, freq="1min"),
            "1": torch.linspace(100, 110, 1000),  # open
            "2": torch.linspace(100, 110, 1000) + 0.5,  # high
            "3": torch.linspace(100, 110, 1000) - 0.5,  # low
            "4": torch.linspace(100, 110, 1000),  # close
            "5": torch.ones(1000) * 1000,  # volume
        })

        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[50],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        )

        env = SeqLongOnlyEnv(df, config)

        # Create actor
        market_data_keys = [k for k in env.observation_spec.keys() if k.startswith("market_data_")]
        actor = MomentumActor(market_data_keys=market_data_keys)

        # Run one episode
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < 10:
            obs = actor(obs.clone())
            obs = env.step(obs)
            done = obs.get("done", torch.tensor([False])).item()
            steps += 1

        # Should complete without errors
        assert steps > 0
