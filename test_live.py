"""
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
torch.set_float32_matmul_precision("high")
from trading_envs.alpaca.torch_env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import torch
import os
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
import functools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    device = torch.device("cpu")

    total_eval_steps = 1000
    max_rollout_steps = 20 # 20  calc for execute on timeframe
    policy_type = "random"

    torch.manual_seed(42)
    np.random.seed(42)
    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[6, 12, 24], # 30],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute), # 5 min
    )

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"), api_secret=os.getenv("SECRET_KEY")
    )

    scratch_dir = None
    storage_cls = (
        functools.partial(LazyTensorStorage, device=device)
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    # Create replay buffer
    replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=3,
            storage=storage_cls(
                1000000,
            ),
            batch_size=1,
            shared=False,
        )


    # Run Evaluation
    total_collected = 0
    pbar = tqdm(total=total_eval_steps, desc="Evaluating", unit="steps")
    for i in range(1000):
        with set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), torch.no_grad(), timeit("eval"):
            eval_rollout = env.rollout(
                max_rollout_steps,
                auto_cast_to_device=True,
                break_when_any_done=True, # we want to continue sample until we reach the required steps
            )

        episode_end = (
            eval_rollout["next", "done"]
            if eval_rollout["next", "done"].any()
            else eval_rollout["next", "truncated"]
        )
        episode_rewards = eval_rollout["next", "episode_reward"][episode_end]
        episode_length = eval_rollout["next", "step_count"][episode_end]
        print("*** Evaluation Stats: ***")
        print(f"Episode rewards: {episode_rewards.mean()}")
        print(f"Episode rewards std: {episode_rewards.std()}")
        print(f"Episode count: {len(episode_rewards)}")
        print(f"Episode length: {episode_length.sum() / len(episode_length)}")
        # could do some preprocessing here
        eval_rollout = eval_rollout.cpu().reshape(-1)
        steps_collected = eval_rollout.batch_size[0]
        total_collected += steps_collected
        pbar.update(steps_collected)
        pbar.set_postfix({
            'collected': f'{total_collected}/{total_eval_steps}'
        })
        replay_buffer.extend(eval_rollout)
        replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")
        if total_collected >= total_eval_steps:
            break

    pbar.close()
    

    

if __name__ == "__main__":
    main()