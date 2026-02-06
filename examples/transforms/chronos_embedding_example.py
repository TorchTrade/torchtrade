"""Example usage of ChronosEmbeddingTransform with TorchTrade environments.

This example demonstrates how to use Amazon's Chronos time series models
to embed market data observations before feeding them to an RL policy.

Available Chronos Models:
    - chronos-t5-tiny (8M params) - Fast, for testing/CI
    - chronos-t5-mini (20M params) - Small deployments
    - chronos-t5-small (46M params) - Balanced
    - chronos-t5-base (200M params) - Standard
    - chronos-t5-large (710M params) - Best performance (default)

Aggregation Strategies:
    - mean: Averages embeddings across features (compact)
    - max: Takes max across features (captures extremes)
    - concat: Concatenates all features (preserves all information)

Installation:
    pip install torchtrade[chronos]

Usage:
    python examples/transforms/chronos_embedding_example.py
"""

import torch
from torchrl.envs import TransformedEnv, Compose, InitTracker, RewardSum
from tensordict.nn import TensorDictModule
import torch.nn as nn

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.transforms import ChronosEmbeddingTransform
import datasets


def example_basic_and_multitimeframe():
    """Example 1: Basic usage and multi-timeframe embeddings."""
    print("=" * 60)
    print("Example 1: Basic & Multi-Timeframe Chronos Embeddings")
    print("=" * 60)

    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = df["train"].to_pandas()

    # Create environment with multiple timeframes
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=["1Min", "5Min", "15Min"],  # 1min, 5min, 15min
        window_sizes=[12, 8, 6],
        execute_on="5Min",
        initial_cash=1000,
    )
    base_env = SequentialTradingEnv(df, config)

    # Transform all timeframes with Chronos embeddings
    env = TransformedEnv(
        base_env,
        Compose(
            InitTracker(),
            RewardSum(),
        )
    )
    env.append_transform(ChronosEmbeddingTransform(
                in_keys=[
                    "market_data_1Minute_12",
                    "market_data_5Minute_8",
                    "market_data_15Minute_6"
                ],
                out_keys=["embedding_1min", "embedding_5min", "embedding_15min"],
                model_name="amazon/chronos-t5-large",
                aggregation="mean",
                device="cuda" if torch.cuda.is_available() else "cpu"
            ))

    print(f"\nObservation spec:")
    print(f"  Keys: {list(env.observation_spec.keys())}")

    # Reset and check shapes
    td = env.reset()
    print(f"\nObservation shapes:")
    print(f"  1min embedding: {td['embedding_1min'].shape}")
    print(f"  5min embedding: {td['embedding_5min'].shape}")
    print(f"  15min embedding: {td['embedding_15min'].shape}")
    print(f"  Account state: {td['account_state'].shape}")

    # Take a few steps
    print(f"\nRunning 5 steps...")
    for i in range(5):
        td["action"] = torch.tensor(1)  # HOLD
        td = env.step(td)
        #print(td)

    del env

def example_with_policy():
    """Example 2: Using Chronos embeddings with an RL policy."""
    print("\n" + "=" * 60)
    print("Example 2: Chronos Embedding with RL Policy")
    print("=" * 60)

    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = df["train"].to_pandas()

    # Create environment with Chronos embedding
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=["5Min", "15Min"],
        window_sizes=[12, 8],
        execute_on="5Min",
        initial_cash=1000,
    )
    base_env = SequentialTradingEnv(df, config)

    env = TransformedEnv(
        base_env,
    )
    env.append_transform(
        ChronosEmbeddingTransform(
            in_keys=["market_data_5Minute_12", "market_data_15Minute_8"],
            out_keys=["embedding_5min", "embedding_15min"],
            model_name="amazon/chronos-t5-small",  # Smaller for demo
            aggregation="mean",
            device="cpu"
        )
    )
    # Get dimensions from spec
    obs_spec = env.observation_spec
    embedding_dim = env.transform[0].embedding_dim

    account_dim = obs_spec["account_state"].shape[0]

    # Create simple MLP policy
    class TradingPolicy(nn.Module):
        """Simple MLP policy using Chronos embeddings + account state."""
        def __init__(self, embedding_dim, account_dim, action_dim=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*embedding_dim + account_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, embd1, embd2, account_state):
            x = torch.cat([embd1, embd2, account_state], dim=-1)
            return self.net(x)

    policy_module = TradingPolicy(embedding_dim, account_dim)
    policy = TensorDictModule(
        module=policy_module,
        in_keys=["embedding_5min", "embedding_15min", "account_state"],
        out_keys=["logits"]
    )

    print(f"\nPolicy architecture:")
    print(f"  Input: Chronos (2 x {embedding_dim}) + Account ({account_dim})")
    print(f"  Hidden: 128 -> 64")
    print(f"  Output: 3 actions (sell/hold/buy)")

    # Test policy
    td = env.reset()
    td = policy(td)
    print(f"\nPolicy output logits: {td['logits']}")

    del env


if __name__ == "__main__":
    print("\nChronos Embedding Transform Examples")
    print("=" * 60)
    print("NOTE: First run downloads models from HuggingFace")
    print("      This may take a few minutes")
    print("=" * 60)

    try:
        example_basic_and_multitimeframe()
        example_with_policy()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nInstall chronos-forecasting:")
        print("  pip install torchtrade[chronos]")
    except Exception as e:
        print(f"\nError: {e}")
        raise
