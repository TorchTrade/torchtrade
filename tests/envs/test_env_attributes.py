"""
Tests to ensure all TorchTrade environments have required attributes.

This test suite verifies that all implemented environments (offline, alpaca, binance)
have the necessary attributes for LLM actor integration:
- account_state: list of account state variable names
- market_data_keys: list of market data observation keys
"""

import pandas as pd

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


# Aliases for backwards compatibility in tests
SeqLongOnlyEnv = SequentialTradingEnv
SeqLongOnlyEnvConfig = SequentialTradingEnvConfig
SeqLongOnlySLTPEnv = SequentialTradingEnvSLTP
SeqLongOnlySLTPEnvConfig = SequentialTradingEnvSLTPConfig
LongOnlyOneStepEnv = OneStepTradingEnv
LongOnlyOneStepEnvConfig = OneStepTradingEnvConfig
SeqFuturesEnv = SequentialTradingEnv
SeqFuturesEnvConfig = SequentialTradingEnvConfig
SeqFuturesSLTPEnv = SequentialTradingEnvSLTP
SeqFuturesSLTPEnvConfig = SequentialTradingEnvSLTPConfig
FuturesOneStepEnv = OneStepTradingEnv
FuturesOneStepEnvConfig = OneStepTradingEnvConfig


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


# Expected account state configuration (unified 6-element structure)
UNIFIED_ACCOUNT_STATE = [
    "exposure_pct", "position_direction", "unrealized_pnlpct",
    "holding_time", "leverage", "distance_to_liquidation"
]


class TestOfflineEnvironmentAttributes:
    """Test offline environment attributes."""

    def test_seqlongonly_has_attributes(self, sample_ohlcv_df):
        """Test SeqLongOnlyEnv has account_state and market_data_keys."""
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        # Check attributes exist
        assert hasattr(env, "account_state"), "SeqLongOnlyEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "SeqLongOnlyEnv missing market_data_keys attribute"

        # Check they are lists
        assert isinstance(env.account_state, list), "account_state should be a list"
        assert isinstance(env.market_data_keys, list), "market_data_keys should be a list"

        # Check account_state content
        assert env.account_state == UNIFIED_ACCOUNT_STATE, f"Expected {UNIFIED_ACCOUNT_STATE}, got {env.account_state}"

        # Check market_data_keys is not empty
        assert len(env.market_data_keys) > 0, "market_data_keys should not be empty"

        # Check getter methods exist
        assert hasattr(env, "get_account_state"), "SeqLongOnlyEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "SeqLongOnlyEnv missing get_market_data_keys method"

        # Check getter methods are callable
        assert callable(env.get_account_state), "get_account_state should be callable"
        assert callable(env.get_market_data_keys), "get_market_data_keys should be callable"

        # Check getter methods return correct values
        assert env.get_account_state() == env.account_state, "get_account_state() should return same as attribute"
        assert env.get_market_data_keys() == env.market_data_keys, "get_market_data_keys() should return same as attribute"

        # Check return types
        assert isinstance(env.get_account_state(), list), "get_account_state() should return a list"
        assert isinstance(env.get_market_data_keys(), list), "get_market_data_keys() should return a list"

    def test_seqlongonlysltp_has_attributes(self, sample_ohlcv_df):
        """Test SeqLongOnlySLTPEnv has account_state and market_data_keys."""
        config = SeqLongOnlySLTPEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.06],
        )
        env = SeqLongOnlySLTPEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert hasattr(env, "account_state"), "SeqLongOnlySLTPEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "SeqLongOnlySLTPEnv missing market_data_keys attribute"
        assert isinstance(env.account_state, list)
        assert isinstance(env.market_data_keys, list)
        assert env.account_state == UNIFIED_ACCOUNT_STATE
        assert len(env.market_data_keys) > 0

        # Check getter methods
        assert hasattr(env, "get_account_state"), "SeqLongOnlySLTPEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "SeqLongOnlySLTPEnv missing get_market_data_keys method"
        assert callable(env.get_account_state)
        assert callable(env.get_market_data_keys)
        assert env.get_account_state() == env.account_state
        assert env.get_market_data_keys() == env.market_data_keys
        assert isinstance(env.get_account_state(), list)
        assert isinstance(env.get_market_data_keys(), list)

    def test_longonlyonestepenv_has_attributes(self, sample_ohlcv_df):
        """Test LongOnlyOneStepEnv has account_state and market_data_keys."""
        config = LongOnlyOneStepEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert hasattr(env, "account_state"), "LongOnlyOneStepEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "LongOnlyOneStepEnv missing market_data_keys attribute"
        assert isinstance(env.account_state, list)
        assert isinstance(env.market_data_keys, list)
        assert env.account_state == UNIFIED_ACCOUNT_STATE
        assert len(env.market_data_keys) > 0

        # Check getter methods
        assert hasattr(env, "get_account_state"), "LongOnlyOneStepEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "LongOnlyOneStepEnv missing get_market_data_keys method"
        assert callable(env.get_account_state)
        assert callable(env.get_market_data_keys)
        assert env.get_account_state() == env.account_state
        assert env.get_market_data_keys() == env.market_data_keys
        assert isinstance(env.get_account_state(), list)
        assert isinstance(env.get_market_data_keys(), list)

    def test_seqfutures_has_attributes(self, sample_ohlcv_df):
        """Test SeqFuturesEnv has account_state and market_data_keys with 10 elements."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=10.0,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert hasattr(env, "account_state"), "SeqFuturesEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "SeqFuturesEnv missing market_data_keys attribute"
        assert isinstance(env.account_state, list)
        assert isinstance(env.market_data_keys, list)
        assert env.account_state == UNIFIED_ACCOUNT_STATE, f"Expected {UNIFIED_ACCOUNT_STATE}, got {env.account_state}"
        assert len(env.market_data_keys) > 0

        # Check getter methods
        assert hasattr(env, "get_account_state"), "SeqFuturesEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "SeqFuturesEnv missing get_market_data_keys method"
        assert callable(env.get_account_state)
        assert callable(env.get_market_data_keys)
        assert env.get_account_state() == env.account_state
        assert env.get_market_data_keys() == env.market_data_keys
        assert isinstance(env.get_account_state(), list)
        assert isinstance(env.get_market_data_keys(), list)

    def test_seqfuturessltp_has_attributes(self, sample_ohlcv_df):
        """Test SeqFuturesSLTPEnv has account_state and market_data_keys with 10 elements."""
        config = SeqFuturesSLTPEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=10.0,
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.06],
        )
        env = SeqFuturesSLTPEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert hasattr(env, "account_state"), "SeqFuturesSLTPEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "SeqFuturesSLTPEnv missing market_data_keys attribute"
        assert isinstance(env.account_state, list)
        assert isinstance(env.market_data_keys, list)
        assert env.account_state == UNIFIED_ACCOUNT_STATE
        assert len(env.market_data_keys) > 0

        # Check getter methods
        assert hasattr(env, "get_account_state"), "SeqFuturesSLTPEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "SeqFuturesSLTPEnv missing get_market_data_keys method"
        assert callable(env.get_account_state)
        assert callable(env.get_market_data_keys)
        assert env.get_account_state() == env.account_state
        assert env.get_market_data_keys() == env.market_data_keys
        assert isinstance(env.get_account_state(), list)
        assert isinstance(env.get_market_data_keys(), list)

    def test_futuresonestepenv_has_attributes(self, sample_ohlcv_df):
        """Test FuturesOneStepEnv has account_state and market_data_keys with 10 elements."""
        config = FuturesOneStepEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=10.0,
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.06],
        )
        env = FuturesOneStepEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert hasattr(env, "account_state"), "FuturesOneStepEnv missing account_state attribute"
        assert hasattr(env, "market_data_keys"), "FuturesOneStepEnv missing market_data_keys attribute"
        assert isinstance(env.account_state, list)
        assert isinstance(env.market_data_keys, list)
        assert env.account_state == UNIFIED_ACCOUNT_STATE
        assert len(env.market_data_keys) > 0

        # Check getter methods
        assert hasattr(env, "get_account_state"), "FuturesOneStepEnv missing get_account_state method"
        assert hasattr(env, "get_market_data_keys"), "FuturesOneStepEnv missing get_market_data_keys method"
        assert callable(env.get_account_state)
        assert callable(env.get_market_data_keys)
        assert env.get_account_state() == env.account_state
        assert env.get_market_data_keys() == env.market_data_keys
        assert isinstance(env.get_account_state(), list)
        assert isinstance(env.get_market_data_keys(), list)


class TestAlpacaEnvironmentAttributes:
    """Test Alpaca live trading environment attributes."""

    def test_alpaca_torch_env_has_attributes(self):
        """Test AlpacaTorchTradingEnv has account_state and market_data_keys."""
        # Note: We can't easily instantiate Alpaca envs without API credentials,
        # but we can import and check the class structure
        from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

        # Check that the config exists (at minimum)
        assert AlpacaTorchTradingEnv is not None
        assert AlpacaTradingEnvConfig is not None

        # TODO: Add actual instantiation test with mocked Alpaca API when possible

    def test_alpaca_sltp_torch_env_has_attributes(self):
        """Test AlpacaSLTPTorchTradingEnv has account_state and market_data_keys."""
        from torchtrade.envs.live.alpaca.env_sltp import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig

        assert AlpacaSLTPTorchTradingEnv is not None
        assert AlpacaSLTPTradingEnvConfig is not None

        # TODO: Add actual instantiation test with mocked Alpaca API when possible


class TestBinanceEnvironmentAttributes:
    """Test Binance live trading environment attributes."""

    def test_binance_futures_env_has_attributes(self):
        """Test BinanceFuturesTorchTradingEnv has account_state and market_data_keys."""
        from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

        assert BinanceFuturesTorchTradingEnv is not None
        assert BinanceFuturesTradingEnvConfig is not None

        # TODO: Add actual instantiation test with mocked Binance API when possible


class TestAccountStateStructure:
    """Test that account_state structure matches documentation."""

    def test_unified_account_state_length(self, sample_ohlcv_df):
        """Test all environments use unified 6-element account state."""
        # Test spot mode
        spot_config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
        )
        spot_env = SeqLongOnlyEnv(sample_ohlcv_df, spot_config, feature_preprocessing_fn=simple_feature_fn)
        assert len(spot_env.account_state) == 6, f"Spot environments should have 6-element account state, got {len(spot_env.account_state)}"

        # Test futures mode
        futures_config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=10,  # Futures mode
        )
        futures_env = SeqFuturesEnv(sample_ohlcv_df, futures_config, feature_preprocessing_fn=simple_feature_fn)
        assert len(futures_env.account_state) == 6, f"Futures environments should have 6-element account state, got {len(futures_env.account_state)}"


class TestMarketDataKeys:
    """Test market_data_keys generation."""

    def test_market_data_keys_format(self, sample_ohlcv_df):
        """Test that market_data_keys follow the expected format."""
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        # Check format: "market_data_{timeframe}_{window_size}"
        assert len(env.market_data_keys) == 2, "Should have 2 market data keys for 2 timeframes"

        for key in env.market_data_keys:
            assert key.startswith("market_data_"), f"Market data key should start with 'market_data_', got {key}"
            parts = key.split("_")
            assert len(parts) >= 3, f"Market data key should have at least 3 parts, got {parts}"

    def test_market_data_keys_count_matches_timeframes(self, sample_ohlcv_df):
        """Test that number of market_data_keys matches number of timeframes."""
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(15, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000.0,
            transaction_fee=0.001,
            leverage=1,  # Spot mode
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, feature_preprocessing_fn=simple_feature_fn)

        assert len(env.market_data_keys) == 3, "Should have 3 market data keys for 3 timeframes"
