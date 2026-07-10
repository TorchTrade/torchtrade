import torch
from tensordict import TensorDict
from torchtrade.actor import LocalLLMActor          # concrete BaseLLMActor (mocked vllm in tests)
from torchtrade.train.prompts import build_training_prompt

MARKET_KEYS = ["market_data_1Hour_48"]
ACCOUNT_LABELS = ["exposure_pct", "position_direction", "unrealized_pnl_pct",
                  "holding_time", "leverage", "distance_to_liquidation"]


def _actor(**kw):
    return LocalLLMActor(model="test-model", backend="vllm",
                         market_data_keys=MARKET_KEYS, account_state_labels=ACCOUNT_LABELS,
                         action_levels=[-1, 0, 1], symbol="BTC/USD", **kw)


def _td():
    return TensorDict({"market_data_1Hour_48": torch.randn(1, 48, 5),
                       "account_state": torch.tensor([[0.5, 1.0, 0.02, 5.0, 1.0, 1.0]])},
                      batch_size=[])


def test_default_prompt_matches_actor_builders():
    actor = _actor()
    sysp, userp = build_training_prompt(actor, _td())
    assert sysp == actor._build_system_prompt()
    assert "exposure_pct" in userp and "market_data_1Hour_48" in userp


def test_overrides_apply():
    actor = _actor()
    sysp, userp = build_training_prompt(
        actor, _td(),
        system_prompt="CUSTOM SYS",
        user_prompt_fn=lambda a, td: "CUSTOM USER",
    )
    assert sysp == "CUSTOM SYS"
    assert userp == "CUSTOM USER"
