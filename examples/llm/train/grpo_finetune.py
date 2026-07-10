"""Fine-tune a local LLM trading actor with GRPO on historical bars.

GPU-required (validated on the DGX Spark). Requires: pip install -e ".[llm]".
The trainer scores each decision with OneStepTradingEnv (the reward oracle); training uses
the one-step contextual-bandit form because GRPO needs K samples of the SAME bar.

Run:  VLLM_ALLOW_INSECURE_SERIALIZATION=1 python examples/llm/train/grpo_finetune.py
"""
import datasets
import pandas as pd

from torchtrade.envs.offline import OneStepTradingEnvConfig
from torchtrade.train import LLMTrainer


def simple_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    return df


def main():
    ds = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = ds["train"].to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    config = OneStepTradingEnvConfig(
        symbol="BTC/USD", time_frames=["1Hour"], window_sizes=[48], execute_on="1Hour",
        initial_cash=10000, transaction_fee=0.0, slippage=0.0, action_levels=[-1, 0, 1],
        include_base_features=False, random_start=False,
    )

    trainer = LLMTrainer(
        df=df, config=config, feature_preprocessing_fn=simple_features,
        feature_keys=["close", "volume"],
        model="Qwen/Qwen2.5-0.5B-Instruct",
        method="qlora",          # "full" | "lora" | "qlora"
        num_generations=8,       # GRPO group size (K completions per bar)
        max_steps=100, lr=1e-5,
        use_wandb=True,          # logs loss/mean_reward per step
        output_dir="./llm_grpo_out",
    )
    adapter = trainer.train()
    print(f"Trained adapter saved to {adapter} — load it into LocalLLMActor for eval/live.")


if __name__ == "__main__":
    main()
