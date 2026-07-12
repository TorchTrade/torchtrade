"""Minimal GRPO fine-tuning of a local LLM trading actor on daily bars.

GPU-required (validated on the DGX Spark). Requires: pip install -e ".[llm]".
Run:  python examples/llm/train/grpo_finetune.py

To customize features, reward, prompts, loss, or the training method (full/lora/qlora),
see docs/guides/llm-grpo-training.md — every one is a single keyword argument.
"""
import datasets
import pandas as pd

from torchtrade.envs.offline import OneStepTradingEnvConfig
from torchtrade.llm.train import LLMTrainer


def main():
    df = datasets.load_dataset(
        "Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025", split="train"
    ).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Daily bars — an LLM reasons over higher-level structure, so train on >= daily.
    config = OneStepTradingEnvConfig(
        symbol="BTC/USD", time_frames=["1Day"], window_sizes=[30], execute_on="1Day",
        action_levels=[-1, 0, 1], random_start=False,
    )

    # Defaults do the right thing: model unsloth/Qwen3-8B-Base-bnb-4bit (one 4-bit checkpoint
    # serves both the vLLM rollout engine and the QLoRA trainer; the adapter is hot-swapped
    # each step, ~50MB, not the 16GB base), method="qlora". We opt into constrain_actions=True
    # (guided decoding forces a parseable <answer>N</answer> every time — recommended) and a larger
    # GRPO group; K is cheap thanks to the chunked log-probs (see the guide's Memory section).
    adapter = LLMTrainer(df=df, config=config, num_generations=8, max_steps=100,
                         constrain_actions=True).train()
    print(f"Trained adapter saved to {adapter} — load it into LocalLLMActor for eval/live.")


if __name__ == "__main__":
    main()
