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


@pytest.mark.parametrize("response,expected_msg_fragment", [
    ("<answer>99</answer>", "out of range"),
    ("no tag", "No <answer> tag"),
], ids=["out-of-range", "no-tag"])
def test_extract_action_warns_unconditionally(actor, caplog, response, expected_msg_fragment):
    """Warnings on bad responses are emitted regardless of debug flag."""
    assert actor.debug is False
    with caplog.at_level("WARNING", logger="torchtrade.actor.base_llm_actor"):
        actor._extract_action(response)
    assert any(expected_msg_fragment in r.message for r in caplog.records)


@pytest.mark.parametrize("bad_shape", [
    (48, 4),       # 2D with wrong feature count
    (2, 48, 5),    # 3D with leading dim != 1 (squeeze(0) is a no-op)
], ids=["wrong-feature-count", "batched-3d"])
def test_market_data_shape_mismatch_raises(actor, bad_shape):
    """Bad market data shapes raise ValueError instead of being silently dropped.

    Note: 1D is now a valid path (renders as labeled rows for envs like
    PolymarketBetEnv), so it's tested separately via
    test_flat_1d_market_state_renders_as_labeled_rows.
    """
    td = TensorDict({
        "market_data_1Hour_48": torch.randn(*bad_shape),
        "account_state": torch.tensor([[0.5, 1.0, 0.02, 5.0, 1.0, 1.0]]),
    }, batch_size=[])
    with pytest.raises(ValueError, match="Unexpected market data shape"):
        actor._construct_market_data(td)


def test_vllm_import_error_propagates(monkeypatch):
    """If vllm cannot be imported, construction raises (no silent transformers fallback)."""
    # Setting sys.modules[name] = None makes `from name import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "vllm", None)
    from torchtrade.actor import LocalLLMActor
    with pytest.raises(ImportError):
        LocalLLMActor(
            model="test-model",
            backend="vllm",
            market_data_keys=MARKET_DATA_KEYS,
            account_state_labels=ACCOUNT_STATE_LABELS,
            action_levels=ACTION_LEVELS_FUTURES,
        )


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


# ----- Behaviors needed for envs that don't expose account_state / use flat
# 1D market_state (e.g. PolymarketBetEnv). These let users plug FrontierLLMActor
# / LocalLLMActor in directly without writing a Polymarket-specific subclass.
# ----------------------------------------------------------------------------


def test_account_state_block_omitted_when_key_missing():
    """Envs that omit the account_state key (e.g. PolymarketBetEnv) skip the block."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=["market_state"],
        account_state_labels=[],
        action_levels=[0, 1],
    )
    td = TensorDict({"market_state": torch.tensor([0.5, 0.02, 1500.0, 12000.0])}, batch_size=[])
    assert actor._construct_account_state(td) == ""


def test_flat_1d_market_state_renders_as_labeled_rows():
    """A 1D ``market_state`` is rendered as one labeled row per feature."""
    from torchtrade.actor import LocalLLMActor
    feature_keys = ["yes_price", "spread", "volume_24h", "liquidity"]
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=["market_state"],
        account_state_labels=[],
        action_levels=[0, 1],
        feature_keys=feature_keys,
    )
    td = TensorDict({"market_state": torch.tensor([0.51, 0.03, 1690.0, 18110.0])}, batch_size=[])
    rendered = actor._construct_market_data(td)
    for key in feature_keys:
        assert key in rendered
    assert "0.5100" in rendered
    assert "1690.0000" in rendered


def test_flat_1d_market_state_label_mismatch_raises():
    """1D market state with feature_keys length mismatch raises (no silent f0..fN fallback)."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=["market_state"],
        account_state_labels=[],
        action_levels=[0, 1],
        feature_keys=["yes_price", "spread"],  # 2 keys
    )
    td = TensorDict({"market_state": torch.tensor([0.5, 0.02, 1500.0, 12000.0])}, batch_size=[])  # 4 values
    with pytest.raises(ValueError, match="Unexpected market data shape"):
        actor._construct_market_data(td)


def test_action_descriptions_override_used_in_system_prompt():
    """Explicit ``action_descriptions`` replaces the auto-generated lines."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=["market_state"],
        account_state_labels=[],
        action_levels=[0, 1],
        action_descriptions=[
            "Action 0 → bet DOWN",
            "Action 1 → bet UP",
        ],
    )
    prompt = actor._build_system_prompt()
    assert "bet DOWN" in prompt
    assert "bet UP" in prompt
    # Auto-generated phrases must NOT leak in
    assert "long" not in prompt
    assert "flat/no position" not in prompt


# ============================================================================
# Custom prompt injection (system_prompt + user_prompt_fn) — issue #223
# ============================================================================


def _make_actor(**overrides):
    from torchtrade.actor import LocalLLMActor
    return LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=ACTION_LEVELS_FUTURES,
        symbol="BTC/USD",
        execute_on="1Hour",
        **overrides,
    )


def test_system_prompt_string_override(sample_td):
    """A static string replaces the default system prompt verbatim."""
    actor = _make_actor(system_prompt="CUSTOM SYSTEM")
    with patch.object(actor, "generate", return_value="<answer>1</answer>") as mock_gen:
        result = actor.forward(sample_td)

    mock_gen.assert_called_once()
    assert mock_gen.call_args.args[0] == "CUSTOM SYSTEM"
    assert result["system_prompt"] == "CUSTOM SYSTEM"


def test_system_prompt_callable_receives_actor(sample_td):
    """A callable receives the actor and its return value is used."""
    def build(actor):
        return f"trade {actor.symbol} on {actor.execute_on} ({len(actor.action_levels)} actions)"

    actor = _make_actor(system_prompt=build)
    with patch.object(actor, "generate", return_value="<answer>1</answer>") as mock_gen:
        actor.forward(sample_td)

    assert mock_gen.call_args.args[0] == "trade BTC/USD on 1Hour (3 actions)"


def test_user_prompt_fn_override(sample_td):
    """user_prompt_fn replaces default user prompt construction."""
    def build(actor, td):
        return f"custom: action_levels={actor.action_levels}, has_account={('account_state' in td)}"

    actor = _make_actor(user_prompt_fn=build)
    with patch.object(actor, "generate", return_value="<answer>2</answer>") as mock_gen:
        result = actor.forward(sample_td)

    user_prompt = mock_gen.call_args.args[1]
    assert user_prompt.startswith("custom: action_levels=[-1.0, 0.0, 1.0]")
    assert user_prompt == result["user_prompt"]
    # Default account-state header should NOT be present
    assert "Current account state:" not in user_prompt


def test_system_prompt_empty_string_is_used_verbatim(sample_td):
    """Empty string is a valid override (pin `if override is None`, not `if override`)."""
    actor = _make_actor(system_prompt="")
    with patch.object(actor, "generate", return_value="<answer>0</answer>") as mock_gen:
        actor.forward(sample_td)
    assert mock_gen.call_args.args[0] == ""


def test_forward_polymarket_shape_with_user_prompt_fn():
    """End-to-end forward() on a Polymarket-shaped envelope: no account_state,
    1D market_state, action_descriptions override, and a user_prompt_fn.

    Verifies the merge x #223 integration the PR is selling — the four features
    compose correctly through the full forward pipeline.
    """
    from torchtrade.actor import LocalLLMActor
    feature_keys = ["yes_price", "spread", "volume_24h", "liquidity"]
    actor = LocalLLMActor(
        model="test-model",
        backend="vllm",
        market_data_keys=["market_state"],
        account_state_labels=[],
        action_levels=[0, 1],
        feature_keys=feature_keys,
        action_descriptions=["Action 0 → bet DOWN", "Action 1 → bet UP"],
        user_prompt_fn=lambda a, td: f"yes_price={td['market_state'][0].item():.2f}",
    )
    td = TensorDict({"market_state": torch.tensor([0.51, 0.03, 1690.0, 18110.0])}, batch_size=[])
    with patch.object(actor, "generate", return_value="<think>up</think><answer>1</answer>") as mock_gen:
        result = actor.forward(td)

    system_prompt, user_prompt = mock_gen.call_args.args
    # action_descriptions override flowed through to the system prompt
    assert "bet DOWN" in system_prompt
    assert "bet UP" in system_prompt
    # user_prompt_fn replaced the default user prompt (no account_state/market-data tables)
    assert user_prompt == "yes_price=0.51"
    # forward() wrote the action correctly
    assert result["action"].item() == 1
