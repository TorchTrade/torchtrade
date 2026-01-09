"""
Basic inference example using LocalLLMActor with unsloth.

This example demonstrates how to use a local quantized LLM (Qwen3-0.6B)
for trading decisions instead of cloud APIs.

Requirements:
    pip install unsloth
    pip install torch transformers

Note: This example can run on CPU or GPU. For Raspberry Pi deployment,
      ensure you have at least 2GB RAM available.
"""

import torch
import numpy as np
from torchrl.data import TensorDict
from torchtrade.actor.local_llm_actor import LocalLLMActor


def create_sample_tensordict():
    """Create a sample tensordict with mock market data."""
    # Account state: [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpct]
    account_state = torch.tensor([[10000.0, 0.0, 0.0, 0.0, 101500.0, 0.0]])

    # Market data for different timeframes (shape: [1, window_size, 5])
    # Features: [close, open, high, low, volume]
    market_data_1m = torch.tensor([
        [101450.0, 101440.0, 101480.0, 101420.0, 1.5],
        [101470.0, 101450.0, 101490.0, 101430.0, 2.1],
        [101490.0, 101470.0, 101510.0, 101460.0, 1.8],
        [101500.0, 101490.0, 101520.0, 101480.0, 2.3],
        [101480.0, 101500.0, 101510.0, 101470.0, 1.9],
        [101460.0, 101480.0, 101490.0, 101450.0, 2.0],
        [101470.0, 101460.0, 101485.0, 101455.0, 1.7],
        [101490.0, 101470.0, 101505.0, 101465.0, 2.2],
        [101510.0, 101490.0, 101530.0, 101485.0, 2.5],
        [101500.0, 101510.0, 101520.0, 101490.0, 2.1],
        [101520.0, 101500.0, 101540.0, 101495.0, 2.4],
        [101530.0, 101520.0, 101550.0, 101515.0, 2.6],
    ]).unsqueeze(0)  # Add batch dimension

    market_data_5m = torch.tensor([
        [101400.0, 101380.0, 101450.0, 101370.0, 8.5],
        [101450.0, 101400.0, 101480.0, 101390.0, 9.2],
        [101480.0, 101450.0, 101510.0, 101440.0, 8.8],
        [101500.0, 101480.0, 101520.0, 101470.0, 9.5],
        [101490.0, 101500.0, 101510.0, 101470.0, 8.9],
        [101470.0, 101490.0, 101500.0, 101460.0, 8.7],
        [101490.0, 101470.0, 101510.0, 101465.0, 9.1],
        [101530.0, 101490.0, 101550.0, 101485.0, 10.2],
    ]).unsqueeze(0)

    market_data_15m = torch.tensor([
        [101300.0, 101250.0, 101380.0, 101240.0, 25.3],
        [101380.0, 101300.0, 101420.0, 101290.0, 26.8],
        [101420.0, 101380.0, 101480.0, 101370.0, 27.5],
        [101480.0, 101420.0, 101520.0, 101410.0, 28.2],
        [101500.0, 101480.0, 101530.0, 101470.0, 27.9],
        [101490.0, 101500.0, 101520.0, 101470.0, 26.5],
        [101510.0, 101490.0, 101540.0, 101485.0, 28.8],
        [101530.0, 101510.0, 101560.0, 101505.0, 29.5],
    ]).unsqueeze(0)

    market_data_1h = torch.tensor([
        [101000.0, 100950.0, 101150.0, 100920.0, 95.5],
        [101150.0, 101000.0, 101250.0, 100980.0, 102.3],
        [101250.0, 101150.0, 101350.0, 101130.0, 98.7],
        [101350.0, 101250.0, 101420.0, 101230.0, 105.2],
        [101420.0, 101350.0, 101480.0, 101340.0, 103.8],
        [101480.0, 101420.0, 101520.0, 101410.0, 107.5],
        [101500.0, 101480.0, 101540.0, 101470.0, 104.2],
        [101530.0, 101500.0, 101570.0, 101495.0, 110.8],
    ] + [[101530.0, 101500.0, 101570.0, 101495.0, 110.8]] * 16).unsqueeze(0)  # Pad to 24

    tensordict = TensorDict({
        "account_state": account_state,
        "market_data_1Minute_12": market_data_1m,
        "market_data_5Minute_8": market_data_5m,
        "market_data_15Minute_8": market_data_15m,
        "market_data_1Hour_24": market_data_1h,
    }, batch_size=[1])

    return tensordict


def main():
    """Run a simple inference example."""
    print("=" * 80)
    print("Local LLM Trading Actor - Inference Example")
    print("=" * 80)
    print()

    # Create the actor
    print("Loading LocalLLMActor with Qwen3-0.6B (4-bit quantized)...")
    print("This may take a minute on first run as the model is downloaded...")
    print()

    actor = LocalLLMActor(
        model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        debug=True,  # Show prompts and responses
    )

    print("\nModel loaded successfully!")
    print()

    # Create sample data
    print("Creating sample market data...")
    tensordict = create_sample_tensordict()
    print("Sample data created.")
    print()

    # Run inference
    print("=" * 80)
    print("Running inference...")
    print("=" * 80)
    print()

    result = actor(tensordict)

    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Action taken: {result.get('action').item()}")
    print(f"Action meaning: {['sell', 'hold', 'buy'][result.get('action').item()]}")

    if "thinking" in result.keys():
        print(f"\nModel's reasoning:")
        print(result.get("thinking"))

    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
