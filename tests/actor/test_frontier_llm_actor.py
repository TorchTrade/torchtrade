"""Tests for FrontierLLMActor (OpenAI API-based).

BaseLLMActor behavior (prompt construction, action extraction, forward pass)
is already covered by test_local_llm_actor.py. These tests cover only
the FrontierLLMActor-specific logic: OpenAI client setup and generate().
"""

from unittest.mock import patch, MagicMock

import pytest
import torch
from tensordict import TensorDict

# Skip entire module if openai is not installed
openai = pytest.importorskip("openai")


MARKET_DATA_KEYS = ["market_data_1Hour_48"]
ACCOUNT_STATE_LABELS = [
    "exposure_pct", "position_direction", "unrealized_pnlpct",
    "holding_time", "leverage", "distance_to_liquidation",
]
ACTION_LEVELS = [-1.0, 0.0, 1.0]


@pytest.fixture
def mock_openai():
    """Mock OpenAI client so no real API calls are made."""
    with patch("openai.OpenAI") as cls:
        client = MagicMock()
        response = MagicMock()
        response.output_text = "<think>analysis</think><answer>2</answer>"
        client.responses.create.return_value = response
        cls.return_value = client
        yield client


@pytest.fixture
def actor(mock_openai):
    from torchtrade.actor import FrontierLLMActor
    return FrontierLLMActor(
        model="gpt-4o-mini",
        api_key="test-key",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=ACTION_LEVELS,
    )


@pytest.fixture
def sample_td():
    return TensorDict({
        "market_data_1Hour_48": torch.randn(1, 48, 5),
        "account_state": torch.tensor([[0.5, 1.0, 0.02, 5.0, 1.0, 1.0]]),
    }, batch_size=[])


def test_generate_calls_openai_api(actor, mock_openai):
    """generate() delegates to OpenAI responses.create with correct args."""
    result = actor.generate("system prompt", "user prompt")

    mock_openai.responses.create.assert_called_once_with(
        model="gpt-4o-mini",
        instructions="system prompt",
        input="user prompt",
    )
    assert result == "<think>analysis</think><answer>2</answer>"


def test_forward_end_to_end(actor, sample_td):
    """Full forward pass produces correct action from mocked API response."""
    result = actor.forward(sample_td)

    assert result["action"].item() == 2
    assert result["action"].dtype == torch.long
    assert "analysis" in result["thinking"]


@pytest.mark.parametrize("api_key,expect_dotenv,expected_resolved_key", [
    ("explicit-key", False, "explicit-key"),
    (None, True, "env-key"),
], ids=["explicit-key", "dotenv-fallback"])
def test_api_key_resolution(api_key, expect_dotenv, expected_resolved_key):
    """Uses explicit key when provided, falls back to .env otherwise."""
    with patch("openai.OpenAI") as mock_cls, \
         patch("dotenv.dotenv_values") as mock_dotenv:
        mock_dotenv.return_value = {"OPENAI_API_KEY": "env-key"}
        mock_cls.return_value = MagicMock()
        from torchtrade.actor import FrontierLLMActor

        FrontierLLMActor(
            model="gpt-4o-mini",
            api_key=api_key,
            market_data_keys=MARKET_DATA_KEYS,
            account_state_labels=ACCOUNT_STATE_LABELS,
            action_levels=ACTION_LEVELS,
        )

        assert mock_dotenv.call_count == (1 if expect_dotenv else 0)
        assert mock_cls.call_count == 1
        assert mock_cls.call_args.kwargs["api_key"] == expected_resolved_key
