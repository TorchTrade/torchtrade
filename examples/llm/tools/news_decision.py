"""LocalLLMActor with the Google News tool making one decision with news context.

Requires: pip install -e ".[llm]"  (adds feedparser) + network access.
Run:      python examples/llm/tools/news_decision.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from tensordict import TensorDict

from torchtrade.actor import LocalLLMActor
from torchtrade.actor.tools import GoogleNewsTool

SYMBOL = "BTC/USD"


def main():
    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=True,
        market_data_keys=["market_data_1Hour_48"],
        account_state_labels=[
            "exposure_pct", "position_direction", "unrealized_pnl_pct",
            "holding_time", "leverage", "distance_to_liquidation",
        ],
        action_levels=[-1, 0, 1],
        symbol=SYMBOL,
        execute_on="1Hour",
        tools=[GoogleNewsTool(symbol=SYMBOL)],
        max_tool_iters=3,
    )
    td = TensorDict({
        "market_data_1Hour_48": torch.randn(1, 48, 5),
        "account_state": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]),
    }, batch_size=[])
    out = actor(td)
    print(f"\nDecision: action={out['action'].item()} "
          f"(target {actor.action_levels[out['action'].item()] * 100:+.0f}%)")


if __name__ == "__main__":
    main()
