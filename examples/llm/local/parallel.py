"""Parallel-env batched inference with a batched LocalLLMActor.

Runs one LLM policy across N offline envs in a SINGLE batched generation pass
per step (via ParallelEnv). Here the N envs share one data source with
different random start windows; to trade multiple symbols instead, build each
env from a different dataframe/symbol -- the batched actor handles it the same
way (one generation pass -> N actions).

Usage:
    python examples/llm/local/parallel.py

Requirements:
    pip install -e ".[llm]"   # needs a CUDA GPU for vllm
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import functools

import datasets
import pandas as pd
import torch
from torchrl.envs import EnvCreator, ParallelEnv

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=False)
    for c in ["open", "high", "low", "close", "volume"]:
        df[f"features_{c}"] = df[c]
    df.dropna(inplace=True)
    return df


def make_config():
    return SequentialTradingEnvConfig(
        symbol="BTC/USD", time_frames=["1Hour"], window_sizes=[48],
        execute_on="1Hour", initial_cash=10000, transaction_fee=0.0, slippage=0.0,
        action_levels=[-1, 0, 1], include_base_features=False, random_start=True,
    )


def make_env(df, config):
    return SequentialTradingEnv(df, config, feature_preprocessing_fn=simple_preprocessing)


def main():
    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")["train"].to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    config = make_config()
    n_envs = 4
    # N parallel envs over one data source (different random start windows).
    # For true multi-symbol, pass a different (df, config) per env instead.
    penv = ParallelEnv(n_envs, EnvCreator(functools.partial(make_env, df, config)))

    ref = make_env(df, config)
    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct", backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        market_data_keys=ref.market_data_keys, account_state_labels=ref.account_state,
        action_levels=ref.action_levels, symbol=config.symbol, execute_on=config.execute_on,
    )

    td = penv.reset()  # batch_size=[n_envs]
    for step in range(5):
        td = actor(td)                       # ONE batched generation -> n_envs actions
        print(f"step {step}: actions={td['action'].tolist()}")
        td = penv.step(td)["next"]
        if td["done"].any():
            print("an env reached done; stopping demo loop")
            break

    penv.close()


if __name__ == "__main__":
    main()
