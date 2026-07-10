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


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """You decide what the LLM sees. Return `features_*` columns; whichever you list in
    `feature_keys` are rendered into the trading prompt (same hook LocalLLMActor uses at
    inference, so training prompts == inference prompts)."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_return"] = df["close"].pct_change().fillna(0.0)
    df["features_sma_ratio"] = (df["close"] / df["close"].rolling(10).mean()).fillna(1.0)
    return df


def main():
    ds = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = ds["train"].to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Daily bars: an LLM reasons over higher-level structure, so train on >= daily timeframes
    # (a decision over 1m/1h noise gives it nothing to latch onto). The sampler resamples the
    # 1m source to 1D; window_sizes=[30] gives the model ~a month of daily context per decision.
    config = OneStepTradingEnvConfig(
        symbol="BTC/USD", time_frames=["1Day"], window_sizes=[30], execute_on="1Day",
        initial_cash=10000, transaction_fee=0.0, slippage=0.0, action_levels=[-1, 0, 1],
        include_base_features=False, random_start=False,
    )

    trainer = LLMTrainer(
        df=df, config=config,
        feature_preprocessing_fn=daily_features,          # what features to COMPUTE
        feature_keys=["features_return", "features_sma_ratio", "close"],  # what the LLM SEES
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
