"""Example usage of MarketRegimeTransform.

This example demonstrates how to use the MarketRegimeTransform to add
market regime features to TorchTrade environments for context-aware trading.
"""

import torch
import pandas as pd
from torchrl.envs import TransformedEnv, Compose
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.transforms import MarketRegimeTransform
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


def create_sample_data(num_points=1000):
    """Create sample OHLCV data for demonstration."""
    timestamps = pd.date_range(start="2024-01-01", periods=num_points, freq="1min")

    # Generate synthetic price data
    torch.manual_seed(42)
    prices = 100 + torch.cumsum(torch.randn(num_points) * 0.5, dim=0).numpy()

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": torch.randint(1000, 5000, (num_points,)).numpy(),
    })

    return df


def example_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=" * 80)
    print("Example 1: Basic MarketRegimeTransform Usage")
    print("=" * 80)

    # Create sample data
    df = create_sample_data(5000)

    # Create base environment
    config = SeqLongOnlyEnvConfig(
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[12],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Wrap with MarketRegimeTransform
    env = TransformedEnv(
        base_env,
        Compose(
            MarketRegimeTransform(
                in_keys=["market_data_1Minute_12"],
                # Uses default parameters:
                # - price_feature_idx=3 (close price)
                # - volume_feature_idx=4 (volume)
                # - volatility_window=20
                # - trend_window=50
                # - volume_window=20
            ),
        )
    )

    # Reset environment
    td = env.reset()

    print("\nObservation keys:", list(td.keys()))
    print("\nRegime features shape:", td["regime_features"].shape)
    print("Regime features values:", td["regime_features"])
    print("\nRegime feature breakdown:")
    print(f"  [0] Volatility regime: {td['regime_features'][0].item():.1f} (0=low, 1=med, 2=high)")
    print(f"  [1] Trend regime: {td['regime_features'][1].item():.1f} (-1=down, 0=sideways, 1=up)")
    print(f"  [2] Volume regime: {td['regime_features'][2].item():.1f} (0=low, 1=normal, 2=high)")
    print(f"  [3] Position regime: {td['regime_features'][3].item():.1f} (0=oversold, 1=neutral, 2=overbought)")
    print(f"  [4] Volatility (continuous): {td['regime_features'][4].item():.6f}")
    print(f"  [5] Trend strength (continuous): {td['regime_features'][5].item():.6f}")
    print(f"  [6] Volume ratio (continuous): {td['regime_features'][6].item():.6f}")

    # Take a few steps
    print("\n" + "-" * 80)
    print("Taking 5 steps to see regime changes...")
    print("-" * 80)

    for i in range(5):
        action = torch.tensor([1])  # Hold
        td = env.step(td.set("action", action))

        print(f"\nStep {i+1}:")
        print(f"  Volatility regime: {td['next']['regime_features'][0].item():.1f}")
        print(f"  Trend regime: {td['next']['regime_features'][1].item():.1f}")
        print(f"  Volume regime: {td['next']['regime_features'][2].item():.1f}")
        print(f"  Position regime: {td['next']['regime_features'][3].item():.1f}")

        if td["next"]["done"].item():
            break


def example_custom_parameters():
    """Example 2: Using custom regime detection parameters."""
    print("\n\n" + "=" * 80)
    print("Example 2: Custom Regime Detection Parameters")
    print("=" * 80)

    # Create sample data
    df = create_sample_data(5000)

    # Create base environment
    config = SeqLongOnlyEnvConfig(
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[12],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Wrap with MarketRegimeTransform using custom parameters
    env = TransformedEnv(
        base_env,
        Compose(
            MarketRegimeTransform(
                in_keys=["market_data_1Minute_12"],
                price_feature_idx=3,
                volume_feature_idx=4,
                # Custom windows for more responsive regime detection
                volatility_window=10,  # Shorter window = more responsive
                trend_window=30,       # Shorter window = more responsive
                volume_window=10,
                position_window=100,
                # Custom thresholds
                vol_percentiles=[0.25, 0.75],      # More extreme volatility buckets
                trend_thresholds=[-0.01, 0.01],     # Tighter trend detection
                volume_thresholds=[0.5, 1.5],       # Wider volume buckets
                position_percentiles=[0.3, 0.7],    # Custom position ranges
            ),
        )
    )

    # Reset and inspect
    td = env.reset()

    print("\nCustom configuration applied:")
    print("  - Shorter windows (10-30 bars) for more responsive detection")
    print("  - Tighter trend thresholds (±1%) for more trend sensitivity")
    print("  - Wider volume thresholds (0.5x-1.5x) for clearer extremes")

    print("\nInitial regime features:", td["regime_features"])


def example_multi_timeframe():
    """Example 3: Using with multi-timeframe observations."""
    print("\n\n" + "=" * 80)
    print("Example 3: Multi-Timeframe Regime Features")
    print("=" * 80)

    # Create sample data
    df = create_sample_data(10000)

    # Create base environment with multiple timeframes
    config = SeqLongOnlyEnvConfig(
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
        ],
        window_sizes=[12, 12, 12],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Add regime features based on the primary (execution) timeframe
    env = TransformedEnv(
        base_env,
        Compose(
            MarketRegimeTransform(
                # Use the execution timeframe for regime detection
                in_keys=["market_data_1Minute_12"],
            ),
        )
    )

    # Reset and inspect
    td = env.reset()

    print("\nEnvironment has multiple timeframe observations:")
    market_data_keys = [k for k in td.keys() if k.startswith("market_data")]
    for key in market_data_keys:
        print(f"  {key}: shape {td[key].shape}")

    print("\nRegime features computed from:", "market_data_1Minute_12")
    print("Regime features:", td["regime_features"])

    print("\nNote: You could also create separate regime transforms for each timeframe")
    print("      if you want regime-aware features at multiple time scales.")


def example_regime_conditional_policy():
    """Example 4: Using regime features for conditional policy."""
    print("\n\n" + "=" * 80)
    print("Example 4: Regime-Conditional Trading Logic")
    print("=" * 80)

    # Create sample data
    df = create_sample_data(5000)

    # Create environment with regime transform
    config = SeqLongOnlyEnvConfig(
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[12],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,
    )
    base_env = SeqLongOnlyEnv(df, config)

    env = TransformedEnv(
        base_env,
        Compose(
            MarketRegimeTransform(
                in_keys=["market_data_1Minute_12"],
            ),
        )
    )

    # Simple regime-aware trading logic
    td = env.reset()

    print("\nDemonstrating regime-aware decision making:")
    print("-" * 80)

    for i in range(10):
        regime = td["regime_features"]

        # Extract regime components
        vol_regime = regime[0].item()
        trend_regime = regime[1].item()
        volume_regime = regime[2].item()
        position_regime = regime[3].item()

        # Simple regime-conditional logic
        if trend_regime == 1.0 and vol_regime <= 1.0:
            # Uptrend + low/medium volatility = aggressive long
            action = 2  # Buy
            strategy = "Aggressive Long (uptrend + low vol)"
        elif trend_regime == -1.0 and vol_regime <= 1.0:
            # Downtrend + low/medium volatility = exit
            action = 0  # Sell
            strategy = "Exit (downtrend)"
        elif vol_regime == 2.0:
            # High volatility = conservative/hold
            action = 1  # Hold
            strategy = "Hold (high volatility)"
        else:
            # Default to hold
            action = 1
            strategy = "Hold (neutral)"

        print(f"\nStep {i+1}:")
        print(f"  Volatility: {['Low', 'Med', 'High'][int(vol_regime)]}")
        print(f"  Trend: {['Down', 'Sideways', 'Up'][int(trend_regime + 1)]}")
        print(f"  Volume: {['Low', 'Normal', 'High'][int(volume_regime)]}")
        print(f"  Position: {['Oversold', 'Neutral', 'Overbought'][int(position_regime)]}")
        print(f"  → Strategy: {strategy}")
        print(f"  → Action: {['SELL', 'HOLD', 'BUY'][action]}")

        # Take action
        td = env.step(td.set("action", torch.tensor([action])))

        if td["next"]["done"].item():
            break

    print("\n" + "=" * 80)
    print("This demonstrates how regime features enable context-aware policies!")
    print("In RL training, the neural network will learn much more sophisticated")
    print("regime-action associations than this simple rule-based example.")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_parameters()
    example_multi_timeframe()
    example_regime_conditional_policy()

    print("\n\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. MarketRegimeTransform adds 7 regime features to observations")
    print("2. Features include both categorical (regimes) and continuous values")
    print("3. Can be applied to any TorchTrade environment using TransformedEnv")
    print("4. Configurable windows and thresholds for different trading styles")
    print("5. Enables agents to learn context-dependent trading strategies")
    print("\nFor RL training, simply add this transform to your environment pipeline")
    print("and your policy network will automatically receive regime features!")
