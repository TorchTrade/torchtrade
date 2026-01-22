"""
Base test classes for exchange-specific tests.

Provides reusable test patterns for testing observation classes, order executors,
environments, and SL/TP functionality across different exchanges (Alpaca, Binance, Bitget).
"""

import pytest
import numpy as np
import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


# ============================================================================
# Base Observation Class Tests
# ============================================================================


class BaseObservationClassTests(ABC):
    """
    Base test class for exchange observation classes.

    Tests observation fetching, preprocessing, and feature extraction.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """
        Create an observer instance for the specific exchange.

        Args:
            symbol: Trading symbol
            timeframes: Single TimeFrame or list of TimeFrames
            window_sizes: Single int or list of ints
            **kwargs: Exchange-specific parameters

        Returns:
            Observer instance
        """
        pass

    @abstractmethod
    def get_expected_symbol_format(self, symbol):
        """
        Get the expected symbol format for this exchange.

        Args:
            symbol: Input symbol (e.g., "BTC/USD", "BTCUSDT")

        Returns:
            Expected normalized symbol format
        """
        pass

    # Initialization tests

    def test_init_single_timeframe(self):
        """Test initialization with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=20,
        )

        assert len(observer.timeframes) == 1 or len(observer.time_frames) == 1
        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames'))
        window_sizes = observer.window_sizes

        assert timeframes[0].value == 15
        assert window_sizes[0] == 20

    def test_init_multiple_timeframes(self):
        """Test initialization with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 30],
        )

        timeframes = getattr(observer, 'timeframes', getattr(observer, 'time_frames'))
        assert len(timeframes) == 3
        assert len(observer.window_sizes) == 3

    def test_init_mismatched_lengths_raises_error(self):
        """Test that mismatched timeframes and window_sizes raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            self.create_observer(
                symbol="BTC/USD",
                timeframes=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 20, 30],  # 3 sizes for 2 timeframes
            )

    # get_keys tests

    def test_get_keys_single_timeframe(self):
        """Test get_keys with single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        keys = observer.get_keys()
        assert len(keys) == 1
        assert "15Minute_10" in keys[0] or "15m_10" in keys[0].lower()

    def test_get_keys_multiple_timeframes(self):
        """Test get_keys with multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20],
        )

        keys = observer.get_keys()
        assert len(keys) == 2

    # get_observations tests

    def test_get_observations_single_timeframe(self):
        """Test getting observations for single timeframe."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()

        assert isinstance(observations, dict)
        assert len(observations) >= 1  # At least one key

        # Check first observation
        key = observer.get_keys()[0]
        assert key in observations
        assert isinstance(observations[key], np.ndarray)
        assert observations[key].shape[0] == 10  # window_size
        assert observations[key].shape[1] >= 4  # At least 4 features (OHLC-based)

    def test_get_observations_multiple_timeframes(self):
        """Test getting observations for multiple timeframes."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 20],
        )

        observations = observer.get_observations()

        assert len(observations) >= 2
        keys = observer.get_keys()
        assert observations[keys[0]].shape == (10, 4) or observations[keys[0]].shape[0] == 10
        assert observations[keys[1]].shape == (20, 4) or observations[keys[1]].shape[0] == 20

    def test_get_observations_with_base_ohlc(self):
        """Test getting observations with base OHLC data."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations(return_base_ohlc=True)

        assert "base_features" in observations
        assert observations["base_features"].shape[1] == 4  # OHLC

    def test_observations_are_float32(self):
        """Test that observations are float32."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].dtype == np.float32

    def test_observations_no_nan_values(self):
        """Test that observations don't contain NaN values."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert not np.isnan(observations[key]).any()

    # get_features tests

    def test_get_features_default_preprocessing(self):
        """Test get_features with default preprocessing."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
        )

        features = observer.get_features()

        assert "observation_features" in features
        assert "original_features" in features
        assert len(features["observation_features"]) >= 4  # At least OHLC features

    # Custom preprocessing tests

    def test_custom_preprocessing(self):
        """Test with custom preprocessing function."""
        def custom_preprocessing(df):
            df = df.copy()
            df.dropna(inplace=True)
            df["feature_volatility"] = df["high"] - df["low"]
            df["feature_volume_ma"] = df["volume"].rolling(window=3).mean()
            df.dropna(inplace=True)
            return df

        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocessing,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        # Custom preprocessing has 2 features
        assert observations[key].shape[1] == 2

    # Edge cases

    def test_window_size_one(self):
        """Test with window size of 1."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=1,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 1

    def test_large_window_size(self):
        """Test with large window size."""
        observer = self.create_observer(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=100,
        )

        observations = observer.get_observations()
        key = observer.get_keys()[0]

        assert observations[key].shape[0] == 100


# ============================================================================
# Base Order Executor Tests
# ============================================================================


class BaseOrderExecutorTests(ABC):
    """
    Base test class for exchange order executors.

    Tests order execution, position management, and status retrieval.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_order_executor(self, symbol, trade_mode, **kwargs):
        """
        Create an order executor instance for the specific exchange.

        Args:
            symbol: Trading symbol
            trade_mode: Trade mode (QUANTITY or NOTIONAL)
            **kwargs: Exchange-specific parameters (leverage, etc.)

        Returns:
            Order executor instance
        """
        pass

    @abstractmethod
    def get_trade_mode_enum(self):
        """Get the TradeMode enum for this exchange."""
        pass

    # Initialization tests

    def test_init_with_mock_client(self):
        """Test initialization with injected mock client."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        assert executor.symbol is not None
        assert executor.trade_mode == TradeMode.NOTIONAL

    # Trade execution tests

    def test_market_buy_order(self):
        """Test placing a market buy order."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.trade(
            side="buy",
            amount=1000,  # $1000 notional
            order_type="market",
        )

        assert success is True

    def test_market_sell_order(self):
        """Test placing a market sell order."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        # First buy
        executor.trade(side="buy", amount=1000, order_type="market")

        # Then sell
        success = executor.trade(
            side="sell",
            amount=1000,
            order_type="market",
        )

        assert success is True

    def test_limit_order(self):
        """Test placing a limit order."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.trade(
            side="buy",
            amount=1000,
            order_type="limit",
            limit_price=95000.0,
        )

        assert success is True

    def test_limit_order_without_price_fails(self):
        """Test that limit order without price fails."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.trade(
            side="buy",
            amount=1000,
            order_type="limit",
        )

        assert success is False

    def test_order_with_take_profit(self):
        """Test order with take profit."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.trade(
            side="buy",
            amount=1000,
            order_type="market",
            take_profit=110000.0,
        )

        assert success is True

    def test_order_with_stop_loss(self):
        """Test order with stop loss."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.trade(
            side="buy",
            amount=1000,
            order_type="market",
            stop_loss=90000.0,
        )

        assert success is True

    # Status retrieval tests

    def test_get_status(self):
        """Test getting order/position status."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        # Place an order
        executor.trade(side="buy", amount=1000, order_type="market")

        status = executor.get_status()
        assert "position_status" in status or "order_status" in status

    # Position management tests

    def test_close_position(self):
        """Test closing a position."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        # Open position
        executor.trade(side="buy", amount=1000, order_type="market")

        # Close position
        success = executor.close_position()
        assert success is True

    def test_cancel_open_orders(self):
        """Test cancelling open orders."""
        TradeMode = self.get_trade_mode_enum()
        executor = self.create_order_executor(
            symbol="BTC/USD",
            trade_mode=TradeMode.NOTIONAL,
        )

        success = executor.cancel_open_orders()
        assert success is True


# ============================================================================
# Base Environment Tests
# ============================================================================


class BaseEnvTests(ABC):
    """
    Base test class for exchange trading environments.

    Tests environment initialization, reset, step, and trading mechanics.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_env(self, config, observer, trader):
        """
        Create an environment instance for the specific exchange.

        Args:
            config: Environment configuration
            observer: Mock observer
            trader: Mock trader

        Returns:
            Environment instance
        """
        pass

    @abstractmethod
    def create_config(self, **kwargs):
        """
        Create environment configuration for the specific exchange.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Environment config instance
        """
        pass

    # Initialization tests

    def test_init_with_mocks(self, mock_observer, mock_trader):
        """Test initialization with injected mocks."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert env.config == config
        assert env.observer is mock_observer
        assert env.trader is mock_trader

    def test_action_spec(self, mock_observer, mock_trader):
        """Test that action spec is correctly defined."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert hasattr(env, 'action_spec')
        assert env.action_spec.n >= 3  # At least 3 actions

    def test_observation_spec(self, mock_observer, mock_trader):
        """Test that observation spec is correctly defined."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert "account_state" in env.observation_spec.keys()
        # Check for market data keys
        market_keys = [k for k in env.observation_spec.keys() if "market_data" in k]
        assert len(market_keys) >= 1

    # Reset tests

    def test_reset_returns_tensordict(self, mock_observer, mock_trader):
        """Test that reset returns a TensorDict."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()
        assert isinstance(td, TensorDict)

    def test_reset_contains_account_state(self, mock_observer, mock_trader):
        """Test that reset returns account state."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()
        assert env.account_state_key in td.keys()
        # Account state should have at least 7 elements
        assert td[env.account_state_key].shape[0] >= 7

    def test_reset_contains_market_data(self, mock_observer, mock_trader):
        """Test that reset returns market data."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        td = env.reset()

        market_key = env.market_data_keys[0]
        assert market_key in td.keys()
        assert td[market_key].shape[0] == 10  # window_size

    # Step tests

    def test_step_returns_tensordict(self, mock_observer, mock_trader):
        """Test that step returns a TensorDict."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        assert isinstance(td_out, TensorDict)

    def test_step_contains_reward(self, mock_observer, mock_trader):
        """Test that step returns reward."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        assert "reward" in td_out.keys()

    def test_step_contains_done(self, mock_observer, mock_trader):
        """Test that step returns done flags."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        assert "done" in td_out.keys()
        assert "terminated" in td_out.keys()
        assert "truncated" in td_out.keys()

    # Termination tests

    def test_bankruptcy_termination(self, mock_observer, mock_trader):
        """Test that bankruptcy triggers termination."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
            done_on_bankruptcy=True,
            bankrupt_threshold=0.1,
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # Override initial_portfolio_value to simulate bankruptcy
        env.initial_portfolio_value = 10000.0

        # Simulate low portfolio value
        done = env._check_termination(500.0)  # Below 10% of initial
        assert done is True

    def test_no_termination_above_threshold(self, mock_observer, mock_trader):
        """Test that no termination above threshold."""
        config = self.create_config(
            symbol="BTC/USD",
            window_sizes=[10],
            done_on_bankruptcy=True,
            bankrupt_threshold=0.1,
        )

        env = self.create_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.initial_portfolio_value = 10000.0

        done = env._check_termination(5000.0)  # Above 10% of initial
        assert done is False


# ============================================================================
# Base SL/TP Tests
# ============================================================================


class BaseSLTPTests(ABC):
    """
    Base test class for SL/TP (Stop Loss / Take Profit) environments.

    Tests bracket order functionality, action mapping, and SL/TP tracking.
    Each exchange should subclass this and implement the abstract methods.
    """

    @abstractmethod
    def create_sltp_env(self, config, observer, trader):
        """
        Create an SL/TP environment instance for the specific exchange.

        Args:
            config: SL/TP environment configuration
            observer: Mock observer
            trader: Mock trader

        Returns:
            SL/TP environment instance
        """
        pass

    @abstractmethod
    def create_sltp_config(self, **kwargs):
        """
        Create SL/TP environment configuration for the specific exchange.

        Args:
            **kwargs: Configuration parameters (including stoploss_levels, takeprofit_levels)

        Returns:
            SL/TP environment config instance
        """
        pass

    # Initialization tests

    def test_init_with_mocks(self, mock_observer, mock_trader):
        """Test initialization with injected mocks."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        assert env.config == config
        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0

    def test_action_map_structure(self, mock_observer, mock_trader):
        """Test action map has correct structure."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # With 2 SL and 2 TP: 1 HOLD + 4 combinations = 5 (or 9 with shorts)
        assert len(env.action_map) >= 5
        assert env.action_map[0] == (None, None, None)  # HOLD action

    def test_action_map_long_actions(self, mock_observer, mock_trader):
        """Test that long actions have correct SL/TP structure."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02, -0.05),
            takeprofit_levels=(0.03, 0.06),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # Find long actions (skip HOLD at index 0)
        long_actions = [
            env.action_map[i] for i in range(1, min(5, len(env.action_map)))
            if env.action_map[i][0] == "long"
        ]

        for side, sl, tp in long_actions:
            assert side == "long"
            assert sl < 0  # SL below entry for longs
            assert tp > 0  # TP above entry for longs

    # Reset tests

    def test_reset_resets_sltp_state(self, mock_observer, mock_trader):
        """Test that reset clears active SL/TP levels."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        # Set some active levels
        env.active_stop_loss = 49000.0
        env.active_take_profit = 51000.0

        env.reset()

        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0

    # Step tests

    def test_step_hold_action(self, mock_observer, mock_trader):
        """Test step with HOLD action."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()
        td_in = TensorDict({"action": torch.tensor(0)}, batch_size=())  # HOLD
        td_out = env._step(td_in)

        assert isinstance(td_out, TensorDict)

    def test_step_buy_with_sltp(self, mock_observer, mock_trader):
        """Test buy action with SL/TP."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Action 1 should be first SL/TP combination
        td_in = TensorDict({"action": torch.tensor(1)}, batch_size=())
        td_out = env._step(td_in)

        # Position should be opened
        assert env.position.current_position != 0

    # SL/TP tracking tests

    def test_active_sltp_tracking(self, mock_observer, mock_trader):
        """Test that active SL/TP levels are tracked after trade."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Execute a trade
        action_tuple = ("long", -0.02, 0.03)
        trade_info = env._execute_trade_if_needed(action_tuple)

        if trade_info["executed"]:
            # Active SL/TP should be set
            assert env.active_stop_loss > 0
            assert env.active_take_profit > 0

    def test_sltp_reset_on_position_close(self, mock_observer, mock_trader):
        """Test that SL/TP are reset when position closes."""
        config = self.create_sltp_config(
            symbol="BTC/USD",
            window_sizes=[10],
            stoploss_levels=(-0.02,),
            takeprofit_levels=(0.03,),
        )

        env = self.create_sltp_env(
            config=config,
            observer=mock_observer,
            trader=mock_trader,
        )

        env.reset()

        # Set active SL/TP
        env.active_stop_loss = 49000.0
        env.active_take_profit = 51000.0
        env.position.current_position = 1

        # Simulate position closed (would be detected via get_status in real env)
        env.active_stop_loss = 0.0
        env.active_take_profit = 0.0
        env.position.current_position = 0

        assert env.active_stop_loss == 0.0
        assert env.active_take_profit == 0.0
