"""Example usage of ChronosEmbeddingTransform with TorchTrade environments.

This example demonstrates how to use Amazon's Chronos time series models
to embed market data observations before feeding them to an RL policy.

Installation:
    pip install torchtrade[chronos]

    Or install chronos manually:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

Usage:
    python examples/transforms/chronos_embedding_example.py
"""

import torch
import pandas as pd
from torchrl.envs import TransformedEnv, Compose, InitTracker, RewardSum
from torchrl.collectors import SyncDataCollector
from torchrl.modules import TensorDictModule
import torch.nn as nn

from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.transforms import ChronosEmbeddingTransform


def create_sample_data(num_points=5000):
    """Create sample OHLCV data for demonstration."""
    dates = pd.date_range("2024-01-01", periods=num_points, freq="1min")

    # Generate synthetic price data with trend and noise
    base_price = 100.0
    trend = pd.Series(range(num_points)) * 0.01
    noise = pd.Series([i * 0.1 for i in range(num_points)])

    df = pd.DataFrame({
        "timestamp": dates,
        "open": base_price + trend + noise,
        "high": base_price + trend + noise + 1.0,
        "low": base_price + trend + noise - 1.0,
        "close": base_price + trend + noise + 0.5,
        "volume": 1000.0 + pd.Series(range(num_points)),
    })

    return df


def example_basic_usage():
    """Basic example: Single timeframe with Chronos embedding."""
    print("=" * 60)
    print("Example 1: Basic Chronos Embedding")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()

    # Create base environment
    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[1],
        window_sizes=[12],
        execute_on=(5, "Minute"),
        initial_cash=1000,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Wrap with Chronos embedding transform
    env = TransformedEnv(
        base_env,
        Compose(
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_12"],
                out_keys=["chronos_embedding"],
                model_name="amazon/chronos-t5-large",  # Best performance
                aggregation="mean",  # Average across features
                del_keys=True,  # Remove original market data
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            InitTracker(),
            RewardSum(),
        )
    )

    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")
    print(f"Reward spec: {env.reward_spec}")

    # Reset and check observation shape
    td = env.reset()
    print(f"\nObservation keys: {list(td.keys())}")
    print(f"Chronos embedding shape: {td['chronos_embedding'].shape}")
    print(f"Account state shape: {td['account_state'].shape}")

    # Take a few steps
    for i in range(5):
        td["action"] = torch.tensor(1)  # HOLD
        td = env.step(td)
        print(f"Step {i+1}: Reward = {td['reward'].item():.6f}, Done = {td['done'].item()}")

        if td["done"].item():
            break

    env.close()


def example_multi_timeframe():
    """Example: Multiple timeframes with separate embeddings."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Timeframe Chronos Embeddings")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()

    # Create environment with multiple timeframes
    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[1, 5, 15],  # 1min, 5min, 15min
        window_sizes=[12, 8, 6],
        execute_on=(5, "Minute"),
        initial_cash=1000,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Embed all timeframes separately
    env = TransformedEnv(
        base_env,
        Compose(
            ChronosEmbeddingTransform(
                in_keys=[
                    "market_data_1Minute_12",
                    "market_data_5Minute_8",
                    "market_data_15Minute_6"
                ],
                out_keys=[
                    "emb_1min",
                    "emb_5min",
                    "emb_15min"
                ],
                model_name="amazon/chronos-t5-large",
                aggregation="mean",
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            InitTracker(),
            RewardSum(),
        )
    )

    td = env.reset()
    print(f"\nObservation keys: {list(td.keys())}")
    print(f"1min embedding shape: {td['emb_1min'].shape}")
    print(f"5min embedding shape: {td['emb_5min'].shape}")
    print(f"15min embedding shape: {td['emb_15min'].shape}")

    env.close()


def example_with_policy():
    """Example: Using Chronos embeddings with a simple MLP policy."""
    print("\n" + "=" * 60)
    print("Example 3: Chronos Embedding with RL Policy")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()

    # Create environment with Chronos embedding
    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[1, 5],
        window_sizes=[12, 8],
        execute_on=(5, "Minute"),
        initial_cash=1000,
    )
    base_env = SeqLongOnlyEnv(df, config)

    env = TransformedEnv(
        base_env,
        Compose(
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_12"],
                out_keys=["chronos_embedding"],
                model_name="amazon/chronos-t5-small",  # Smaller for faster demo
                aggregation="mean",
                device="cpu"  # Use CPU for this example
            ),
            InitTracker(),
        )
    )

    # Get embedding dimension from spec
    obs_spec = env.observation_spec
    embedding_dim = obs_spec["chronos_embedding"].shape[0]
    account_state_dim = obs_spec["account_state"].shape[0]

    # Create simple MLP policy
    class SimpleTradingPolicy(nn.Module):
        def __init__(self, embedding_dim, account_dim, action_dim=3):
            super().__init__()

            # Combine Chronos embedding with account state
            input_dim = embedding_dim + account_dim

            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, chronos_embedding, account_state):
            # Concatenate embeddings and account state
            x = torch.cat([chronos_embedding, account_state], dim=-1)
            logits = self.net(x)
            return logits

    policy_module = SimpleTradingPolicy(embedding_dim, account_state_dim)

    # Wrap in TensorDictModule
    policy = TensorDictModule(
        module=policy_module,
        in_keys=["chronos_embedding", "account_state"],
        out_keys=["logits"]
    )

    print(f"\nPolicy architecture:")
    print(policy_module)
    print(f"\nInput: Chronos embedding ({embedding_dim}) + Account state ({account_state_dim})")
    print(f"Output: Action logits (3)")

    # Test policy
    td = env.reset()
    td = policy(td)

    print(f"\nPolicy output logits: {td['logits']}")

    env.close()


def example_aggregation_strategies():
    """Example: Different aggregation strategies for multi-feature embeddings."""
    print("\n" + "=" * 60)
    print("Example 4: Aggregation Strategies")
    print("=" * 60)

    df = create_sample_data()

    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[1],
        window_sizes=[12],
        execute_on=(5, "Minute"),
        initial_cash=1000,
    )

    # Test different aggregations
    for aggregation in ["mean", "max", "concat"]:
        print(f"\n--- Aggregation: {aggregation} ---")

        base_env = SeqLongOnlyEnv(df, config)

        env = TransformedEnv(
            base_env,
            ChronosEmbeddingTransform(
                in_keys=["market_data_1Minute_12"],
                out_keys=["chronos_embedding"],
                model_name="amazon/chronos-t5-tiny",  # Tiny for fast testing
                aggregation=aggregation,
                device="cpu"
            )
        )

        td = env.reset()
        embedding_shape = td["chronos_embedding"].shape

        print(f"Embedding shape: {embedding_shape}")
        print(f"Description:")
        if aggregation == "mean":
            print("  - Averages embeddings across all features")
            print("  - Output: (embedding_dim,)")
            print("  - Best for: Compact representation")
        elif aggregation == "max":
            print("  - Takes max across all features")
            print("  - Output: (embedding_dim,)")
            print("  - Best for: Capturing extreme values")
        else:  # concat
            print("  - Concatenates all feature embeddings")
            print("  - Output: (embedding_dim * num_features,)")
            print("  - Best for: Preserving all information")

        env.close()


def example_model_sizes():
    """Example: Different Chronos model sizes."""
    print("\n" + "=" * 60)
    print("Example 5: Chronos Model Sizes")
    print("=" * 60)

    print("\nAvailable Chronos models:")
    print("  - chronos-t5-tiny  (8M params)   - Fast, for testing/CI")
    print("  - chronos-t5-mini  (20M params)  - Small deployments")
    print("  - chronos-t5-small (46M params)  - Balanced")
    print("  - chronos-t5-base  (200M params) - Standard")
    print("  - chronos-t5-large (710M params) - Best performance (default)")

    print("\nRecommendations:")
    print("  - Development/Testing: chronos-t5-tiny or chronos-t5-small")
    print("  - Production: chronos-t5-large or chronos-t5-base")
    print("  - Resource-constrained: chronos-t5-mini")

    print("\nExample: Using smaller model for faster initialization")

    df = create_sample_data(num_points=1000)  # Smaller dataset

    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[1],
        window_sizes=[12],
        execute_on=(5, "Minute"),
        initial_cash=1000,
    )
    base_env = SeqLongOnlyEnv(df, config)

    # Use tiny model for demo
    env = TransformedEnv(
        base_env,
        ChronosEmbeddingTransform(
            in_keys=["market_data_1Minute_12"],
            out_keys=["chronos_embedding"],
            model_name="amazon/chronos-t5-tiny",  # Fast loading
            aggregation="mean",
            device="cpu"
        )
    )

    print("\nInitializing with chronos-t5-tiny...")
    td = env.reset()
    print(f"Success! Embedding shape: {td['chronos_embedding'].shape}")

    env.close()


if __name__ == "__main__":
    print("\nChronos Embedding Transform Examples")
    print("=" * 60)
    print("NOTE: First run will download Chronos models from HuggingFace")
    print("      This may take a few minutes depending on your connection")
    print("=" * 60)

    try:
        # Run all examples
        example_basic_usage()
        example_multi_timeframe()
        example_aggregation_strategies()
        example_model_sizes()
        example_with_policy()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install chronos-forecasting:")
        print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        print("Or install with optional extra:")
        print("  pip install torchtrade[chronos]")
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise
