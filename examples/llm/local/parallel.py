"""Parallel multi-symbol inference with a batched LocalLLMActor.

Runs one LLM policy across N offline envs (e.g. different symbols/windows) in
a single batched generation pass per step via ParallelEnv.

Usage:
    python examples/llm/local/parallel.py

Requirements:
    pip install -e ".[llm]"   # needs a CUDA GPU for vllm
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import datasets
import pandas as pd
import torch
from torchrl.envs import ParallelEnv

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=False)
    for c in ["open", "high", "low", "close", "volume"]:
        df[f"features_{c}"] = df[c]
    df.dropna(inplace=True)
    return df


def make_env(df):
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD", time_frames=["1Hour"], window_sizes=[48],
        execute_on="1Hour", initial_cash=10000, transaction_fee=0.0, slippage=0.0,
        action_levels=[-1, 0, 1], include_base_features=False, random_start=True,
    )
    return SequentialTradingEnv(df, config, feature_preprocessing_fn=simple_preprocessing)


def main():
    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")["train"].to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    n_envs = 4
    penv = ParallelEnv(n_envs, lambda: make_env(df))

    ref = make_env(df)
    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct", backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        market_data_keys=ref.market_data_keys, account_state_labels=ref.account_state,
        action_levels=ref.action_levels, symbol="BTC/USD", execute_on="1Hour",
    )

    td = penv.reset()  # batch_size=[n_envs]
    for step in range(5):
        td = actor(td)                       # ONE batched generation -> n_envs actions
        print(f"step {step}: actions={td['action'].tolist()}")
        td = penv.step(td)["next"]

    penv.close()


if __name__ == "__main__":
    main()
