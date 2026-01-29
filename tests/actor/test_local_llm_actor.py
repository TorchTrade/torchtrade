"""Tests for LLM actors with the environment-driven API."""

from unittest.mock import patch, Mock
from types import ModuleType
import sys

import pytest
import torch
from tensordict import TensorDict


# ============================================================================
# Fixtures
# ============================================================================

MARKET_DATA_KEYS = ["market_data_1Hour_48"]
ACCOUNT_STATE_LABELS = [
    "exposure_pct", "position_direction", "unrealized_pnlpct",
    "holding_time", "leverage", "distance_to_liquidation",
]
ACTION_LEVELS_FUTURES = [-1.0, 0.0, 1.0]
ACTION_LEVELS_SPOT = [0.0, 0.5, 1.0]


@pytest.fixture(autouse=True)
def mock_vllm_backend():
    """Mock vllm so LocalLLMActor can be instantiated without GPU."""
    vllm_module = ModuleType("vllm")

    class MockVLLM:
        def __init__(self, *a, **kw):
            pass
        def get_tokenizer(self):
            tok = Mock()
            tok.apply_chat_template = Mock(side_effect=lambda msgs, **kw:
                f"{msgs[0]['content']}\n\n{msgs[1]['content']}")
            return tok
        def generate(self, prompts, sampling_params):
            out = Mock()
            out.text = "<think>analysis</think><answer>1</answer>"
            req = Mock()
            req.outputs = [out]
            return [req]

    vllm_module.LLM = MockVLLM
    vllm_module.SamplingParams = Mock
    sys.modules["vllm"] = vllm_module
    yield
    sys.modules.pop("vllm", None)


@pytest.fixture
def actor():
    from torchtrade.actor import LocalLLMActor
    return LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=ACTION_LEVELS_FUTURES,
        symbol="BTC/USD",
        execute_on="1Hour",
    )


@pytest.fixture
def sample_td():
    return TensorDict({
        "market_data_1Hour_48": torch.randn(1, 48, 5),
        "account_state": torch.tensor([[0.5, 1.0, 0.02, 5.0, 1.0, 1.0]]),
    }, batch_size=[])


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("response,expected_idx", [
    ("<think>go long</think><answer>2</answer>", 2),
    ("<answer>0</answer>", 0),
    ("<ANSWER> 1 </ANSWER>", 1),  # case-insensitive, whitespace
    ("<answer>99</answer>", 0),   # out of range → default 0
    ("no tags here", 0),          # missing tag → default 0
], ids=["valid-long", "valid-short", "case-whitespace", "out-of-range", "no-tag"])
def test_extract_action(actor, response, expected_idx):
    """Action extraction maps <answer>N</answer> to correct index, with safe fallbacks."""
    assert actor._extract_action(response) == expected_idx


def test_forward_produces_required_keys(actor, sample_td):
    """Forward pass populates action, thinking, system_prompt, user_prompt."""
    with patch.object(actor, "generate", return_value="<think>reasoning</think><answer>2</answer>"):
        result = actor.forward(sample_td)

    assert result["action"].item() == 2
    assert result["action"].dtype == torch.long
    assert "reasoning" in result["thinking"]
    assert "BTC/USD" in result["system_prompt"]
    assert "exposure_pct" in result["user_prompt"]


def test_system_prompt_reflects_action_levels(actor):
    """System prompt describes actions from action_levels, not hardcoded buy/sell/hold."""
    prompt = actor._build_system_prompt()
    assert "target -100%" in prompt   # action_levels[0] = -1
    assert "target 0%" in prompt      # action_levels[1] = 0
    assert "target +100%" in prompt   # action_levels[2] = 1
    assert "buy" not in prompt.lower()
    assert "sell" not in prompt.lower()
    assert "hold" not in prompt.lower()


def test_account_state_uses_provided_labels(actor, sample_td):
    """Account state section uses labels from env, not hardcoded ones."""
    result = actor._construct_account_state(sample_td)
    for label in ACCOUNT_STATE_LABELS:
        assert label in result
    # Old labels should not appear
    assert "cash:" not in result
    assert "margin_ratio:" not in result


@pytest.mark.parametrize("action_levels", [
    [0.0, 1.0],
    [0.0, 0.5, 1.0],
    [-1.0, -0.5, 0.0, 0.5, 1.0],
], ids=["2-action", "3-action-spot", "5-action"])
def test_action_descriptions_match_levels(action_levels):
    """Each action_level gets a corresponding description."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=action_levels,
    )
    assert len(actor._action_descriptions) == len(action_levels)
    prompt = actor._build_system_prompt()
    assert f"0 to {len(action_levels) - 1}" in prompt
