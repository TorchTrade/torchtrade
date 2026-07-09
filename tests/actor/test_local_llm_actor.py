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
            reqs = []
            for _ in prompts:
                out = Mock()
                out.text = "<think>analysis</think><answer>1</answer>"
                req = Mock()
                req.outputs = [out]
                reqs.append(req)
            return reqs

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
    with patch.object(actor, "generate_batch", return_value=["<think>reasoning</think><answer>2</answer>"]):
        result = actor.forward(sample_td)

    assert result["action"].item() == 2
    assert result["action"].dtype == torch.long
    assert "reasoning" in result["thinking"]
    assert "BTC/USD" in result["system_prompt"]
    assert "exposure_pct" in result["user_prompt"]


def test_forward_clamps_out_of_range_action(actor, sample_td):
    """Actor passes num_actions=len(action_levels) to the parser, so the first
    out-of-range index (3 for a 3-action actor) clamps to 0 — pins the
    actor->parser wiring to exactly len(action_levels)."""
    with patch.object(actor, "generate_batch", return_value=["<answer>3</answer>"]):
        result = actor.forward(sample_td)
    assert result["action"].item() == 0


def test_vllm_stop_includes_tool_tag():
    """The tool loop needs generation to halt at </tool> as well as </answer>.

    Also pins include_stop_str_in_output=True: vLLM defaults it to False, which
    strips the stop string from the returned text — but the action/tool parsers'
    regexes require the closing tag, so without this flag every live response
    fails to parse and silently defaults to action 0.
    """
    captured = {}

    class CapturingSamplingParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    sys.modules["vllm"].SamplingParams = CapturingSamplingParams

    from torchtrade.actor import LocalLLMActor
    LocalLLMActor(
        model="test-model", backend="vllm",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=ACTION_LEVELS_FUTURES,
    )
    assert "</tool>" in captured["stop"]
    assert "</answer>" in captured["stop"]
    assert captured["include_stop_str_in_output"] is True


def test_tools_with_transformers_backend_raises():
    """Tool use requires vLLM's </tool> stop; the transformers backend can't halt
    there, so configuring tools with it must fail loud rather than degrade silently."""
    from torchtrade.actor import LocalLLMActor
    from torchtrade.actor.tools import GoogleNewsTool
    with pytest.raises(ValueError, match="Tool use requires backend='vllm'"):
        LocalLLMActor(
            model="test-model", backend="transformers",
            market_data_keys=MARKET_DATA_KEYS,
            account_state_labels=ACCOUNT_STATE_LABELS,
            action_levels=ACTION_LEVELS_FUTURES,
            tools=[GoogleNewsTool(symbol="BTC/USD")],
        )


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
    with patch.object(actor, "generate_batch", return_value=["<answer>1</answer>"]) as mock_gen:
        result = actor.forward(sample_td)

    mock_gen.assert_called_once()
    assert mock_gen.call_args.args[0] == "CUSTOM SYSTEM"
    assert result["system_prompt"] == "CUSTOM SYSTEM"


def test_system_prompt_callable_receives_actor(sample_td):
    """A callable receives the actor and its return value is used."""
    def build(actor):
        return f"trade {actor.symbol} on {actor.execute_on} ({len(actor.action_levels)} actions)"

    actor = _make_actor(system_prompt=build)
    with patch.object(actor, "generate_batch", return_value=["<answer>1</answer>"]) as mock_gen:
        actor.forward(sample_td)

    assert mock_gen.call_args.args[0] == "trade BTC/USD on 1Hour (3 actions)"


def test_user_prompt_fn_override(sample_td):
    """user_prompt_fn replaces default user prompt construction."""
    def build(actor, td):
        return f"custom: action_levels={actor.action_levels}, has_account={('account_state' in td)}"

    actor = _make_actor(user_prompt_fn=build)
    with patch.object(actor, "generate_batch", return_value=["<answer>2</answer>"]) as mock_gen:
        result = actor.forward(sample_td)

    user_prompt = mock_gen.call_args.args[1][0]
    assert user_prompt.startswith("custom: action_levels=[-1.0, 0.0, 1.0]")
    assert user_prompt == result["user_prompt"]
    # Default account-state header should NOT be present
    assert "Current account state:" not in user_prompt


def test_system_prompt_empty_string_is_used_verbatim(sample_td):
    """Empty string is a valid override (pin `if override is None`, not `if override`)."""
    actor = _make_actor(system_prompt="")
    with patch.object(actor, "generate_batch", return_value=["<answer>0</answer>"]) as mock_gen:
        actor.forward(sample_td)
    assert mock_gen.call_args.args[0] == ""


def test_execute_on_timeframe_object_renders_obs_key_freq():
    """Regression: env configs normalize execute_on to a TimeFrame object in
    __post_init__, and LLM examples pass config.execute_on into the actor.
    Plain str(TimeFrame(...)) leaks "TimeFrame(1, TimeFrameUnit.Hour)" into the
    prompt — must use obs_key_freq() to render "1Hour"-style strings.
    """
    from torchtrade.actor import LocalLLMActor
    from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

    def make(tf):
        return LocalLLMActor(
            model="test-model",
            backend="vllm",
            market_data_keys=MARKET_DATA_KEYS,
            account_state_labels=ACCOUNT_STATE_LABELS,
            action_levels=ACTION_LEVELS_FUTURES,
            symbol="BTC/USD",
            execute_on=tf,
        )

    actor = make(TimeFrame(1, TimeFrameUnit.Hour))
    assert actor.execute_on == "1Hour"
    assert "1Hour" in actor._build_system_prompt()

    assert make(TimeFrame(5, TimeFrameUnit.Minute)).execute_on == "5Minute"


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
    with patch.object(actor, "generate_batch", return_value=["<think>up</think><answer>1</answer>"]) as mock_gen:
        result = actor.forward(td)

    system_prompt, user_prompts = mock_gen.call_args.args
    # action_descriptions override flowed through to the system prompt
    assert "bet DOWN" in system_prompt
    assert "bet UP" in system_prompt
    # user_prompt_fn replaced the default user prompt (no account_state/market-data tables)
    assert user_prompts[0] == "yes_price=0.51"
    # forward() wrote the action correctly
    assert result["action"].item() == 1


def test_vllm_generate_batch_returns_one_per_prompt(actor):
    """generate_batch feeds all prompts to vllm.generate once and returns N texts."""
    prompts = ["p0", "p1", "p2"]
    with patch.object(actor.llm, "generate", wraps=actor.llm.generate) as spy:
        out = actor.generate_batch("sys", prompts)
    assert len(out) == 3
    assert all("<answer>1</answer>" in o for o in out)
    # one batched call, not three
    spy.assert_called_once()
    called_prompts = spy.call_args.args[0]
    assert len(called_prompts) == 3


def test_batched_forward_end_to_end_vllm(actor):
    """A [N]-batched td -> N DISTINCT actions via one vllm call (per-element alignment)."""
    def gen(prompts, sampling_params):
        reqs = []
        for i, _ in enumerate(prompts):
            out = Mock(); out.text = f"<think>x</think><answer>{i % 3}</answer>"
            req = Mock(); req.outputs = [out]
            reqs.append(req)
        return reqs
    td = TensorDict({
        "market_data_1Hour_48": torch.randn(3, 48, 5),
        "account_state": torch.randn(3, 6),
    }, batch_size=[3])
    with patch.object(actor.llm, "generate", side_effect=gen):
        result = actor.forward(td)
    assert result["action"].shape == torch.Size([3])
    assert result["action"].tolist() == [0, 1, 2]   # distinct -> proves per-element mapping
    assert len(result["thinking"]) == 3


# ============================================================================
# Tool config, system-prompt tools block, dispatch helpers (C1, Task 3)
# ============================================================================


from torchtrade.actor.tools import Tool


class _EchoTool(Tool):
    name = "echo"
    description = "echo(text): returns the text"
    def run(self, text="hi", **kw):
        return f"echoed: {text}"


class _BoomTool(Tool):
    name = "boom"
    description = "boom(): always fails"
    def run(self, **kw):
        raise RuntimeError("kaboom")


@pytest.fixture
def tool_actor():
    from torchtrade.actor import LocalLLMActor
    return LocalLLMActor(
        model="test-model", backend="vllm",
        market_data_keys=MARKET_DATA_KEYS,
        account_state_labels=ACCOUNT_STATE_LABELS,
        action_levels=ACTION_LEVELS_FUTURES,
        symbol="BTC/USD",
        tools=[_EchoTool(), _BoomTool()],
        max_tool_iters=2,
    )


def test_tools_prompt_lists_tools_and_protocol(tool_actor):
    prompt = tool_actor._resolve_system_prompt()
    assert "echo(text)" in prompt                 # tool description present
    assert '<tool name=' in prompt                # protocol shown
    assert "<answer>" in prompt                   # base contract retained


def test_run_tool_calls_success_and_errors(tool_actor):
    out = tool_actor._run_tool_calls([
        {"name": "echo", "args": {"text": "yo"}, "tag": None},
        {"name": "nope", "args": {}, "tag": None},     # unknown tool
        {"name": "boom", "args": {}, "tag": None},     # raises
    ])
    assert out.startswith("<tool_results>") and out.rstrip().endswith("</tool_results>")
    assert "echoed: yo" in out
    assert "Tool nope (call 2) failed" in out and "unknown tool" in out.lower()
    assert "Tool boom (call 3) failed" in out and "kaboom" in out


def test_no_tools_actor_prompt_unchanged(actor):
    # actor fixture has no tools -> system prompt has no tool section
    assert "<tool name=" not in actor._resolve_system_prompt()


# ============================================================================
# Multi-turn tool loop (C1, Task 4)
# ============================================================================


def test_tool_loop_call_then_answer(tool_actor, sample_td):
    """A tool call is executed, its result injected, then the model answers."""
    calls = [
        ['<tool name="echo">{"text": "news"}</tool>'],   # round 1: tool call
        ["<answer>2</answer>"],                            # round 2: final answer
    ]
    with patch.object(tool_actor, "generate_batch", side_effect=calls) as gen:
        result = tool_actor.forward(sample_td)
    assert result["action"].item() == 2
    assert gen.call_count == 2                            # one extra generation for the tool round


def test_tool_loop_batching_only_regenerates_pending(tool_actor):
    """A1 preservation: only the tool-calling conversation is re-generated."""
    td = TensorDict({
        "market_data_1Hour_48": torch.randn(3, 48, 5),
        "account_state": torch.zeros(3, 6),
    }, batch_size=[3])
    round1 = ['<answer>0</answer>', '<tool name="echo">{}</tool>', '<answer>1</answer>']
    round2 = ['<answer>2</answer>']                       # only the 1 pending conv
    seen = []
    def fake_gen(system, prompts):
        seen.append(list(prompts))
        return round1 if len(seen) == 1 else round2
    with patch.object(tool_actor, "generate_batch", side_effect=fake_gen):
        result = tool_actor.forward(td)
    assert len(seen) == 2
    assert len(seen[1]) == 1                              # 2nd call regenerates ONLY the pending one
    assert result["action"].tolist() == [0, 2, 1]


def test_tool_loop_max_iters_defaults_to_zero(tool_actor, sample_td):
    """Runaway tool caller hits the cap and safely defaults to action 0."""
    with patch.object(tool_actor, "generate_batch",
                      return_value=['<tool name="echo">{}</tool>']) as gen:
        result = tool_actor.forward(sample_td)
    assert result["action"].item() == 0                  # never emitted <answer>
    assert gen.call_count == 1 + tool_actor.max_tool_iters


def test_tool_loop_tool_failure_does_not_crash(tool_actor, sample_td):
    calls = [['<tool name="boom">{}</tool>'], ["<answer>1</answer>"]]
    with patch.object(tool_actor, "generate_batch", side_effect=calls):
        result = tool_actor.forward(sample_td)            # must not raise
    assert result["action"].item() == 1
