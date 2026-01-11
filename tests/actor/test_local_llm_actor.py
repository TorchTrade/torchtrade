"""Tests for LocalLLMActor with mocked LLM backends."""

from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
from tensordict import TensorDict

from torchtrade.envs.offline.utils import build_sltp_action_map


def futures_sltp_action_map(stoploss_levels, takeprofit_levels):
    """Wrapper for backward compatibility in tests."""
    return build_sltp_action_map(stoploss_levels, takeprofit_levels, include_short_positions=True)


# ============================================================================
# Mock Classes
# ============================================================================


class MockVLLMOutput:
    """Mock for vllm output object."""
    def __init__(self, text):
        self.text = text


class MockVLLMRequestOutput:
    """Mock for vllm RequestOutput."""
    def __init__(self, text):
        self.outputs = [MockVLLMOutput(text)]


class MockVLLM:
    """Mock for vllm.LLM class."""
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model", "test-model")

    def get_tokenizer(self):
        """Mock tokenizer."""
        tokenizer = Mock()
        tokenizer.apply_chat_template = Mock(side_effect=lambda msgs, **kw:
            f"{msgs[0]['content']}\n\n{msgs[1]['content']}")
        return tokenizer

    def generate(self, prompts, sampling_params):
        """Mock generate that returns canned responses."""
        # Default response for testing
        response_text = "<think>Analyzing market data...</think><answer>hold</answer>"
        return [MockVLLMRequestOutput(response_text)]


class MockTransformersPipeline:
    """Mock for transformers pipeline."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prompt, **kwargs):
        """Mock generation."""
        response_text = "<think>Analyzing market data...</think><answer>hold</answer>"
        return [{"generated_text": response_text}]


class MockTokenizer:
    """Mock tokenizer for transformers."""
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def from_pretrained(*args, **kwargs):
        tokenizer = MockTokenizer()
        tokenizer.apply_chat_template = Mock(side_effect=lambda msgs, **kw:
            f"{msgs[0]['content']}\n\n{msgs[1]['content']}")
        return tokenizer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_vllm_backend():
    """Mock vllm backend for testing."""
    import sys
    from types import ModuleType

    # Create fake vllm module
    vllm_module = ModuleType("vllm")
    vllm_module.LLM = MockVLLM
    vllm_module.SamplingParams = Mock

    # Install in sys.modules
    sys.modules["vllm"] = vllm_module

    try:
        yield
    finally:
        # Cleanup
        sys.modules.pop("vllm", None)


@pytest.fixture
def mock_transformers_backend():
    """Mock transformers backend for testing."""
    import sys
    from types import ModuleType

    # Create fake transformers module
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = MockTokenizer
    transformers_module.AutoModelForCausalLM = Mock()
    transformers_module.AutoModelForCausalLM.from_pretrained = Mock(return_value=Mock())
    transformers_module.pipeline = MockTransformersPipeline
    transformers_module.BitsAndBytesConfig = Mock

    # Install in sys.modules
    sys.modules["transformers"] = transformers_module

    try:
        yield
    finally:
        # Cleanup
        sys.modules.pop("transformers", None)


@pytest.fixture
def sample_tensordict_standard():
    """Create sample TensorDict with standard 7-element account state."""
    return TensorDict({
        "market_data_1Minute_12": torch.randn(1, 12, 5),  # (batch, window, features)
        "market_data_5Minute_8": torch.randn(1, 8, 5),
        "account_state": torch.tensor([[1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0]]),
    }, batch_size=[])


@pytest.fixture
def sample_tensordict_futures():
    """Create sample TensorDict with futures 10-element account state."""
    return TensorDict({
        "market_data_1Minute_12": torch.randn(1, 12, 5),
        "market_data_5Minute_8": torch.randn(1, 8, 5),
        # 10 elements: cash, position_size, position_value, entry_price, current_price,
        # unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time
        "account_state": torch.tensor([[1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0, 0.2, 95.0, 5.0]]),
    }, batch_size=[])


@pytest.fixture
def sltp_action_map():
    """Create sample SLTP action map."""
    return futures_sltp_action_map(
        stoploss_levels=(-0.02, -0.05),
        takeprofit_levels=(0.05, 0.1)
    )


# ============================================================================
# Tests
# ============================================================================


class TestLocalLLMActorInitialization:
    """Tests for LocalLLMActor initialization."""

    def test_init_default_parameters(self, mock_vllm_backend):
        """Test initialization with default parameters."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        assert actor.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert actor.backend == "vllm"
        assert actor.device == "cuda"
        assert actor.action_space_type == "standard"
        assert actor.action_dict == {"buy": 2, "sell": 0, "hold": 1}

    def test_init_custom_parameters(self, mock_vllm_backend):
        """Test initialization with custom parameters."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            model="custom/model",
            backend="vllm",
            device="cpu",
            temperature=0.5,
            max_tokens=256,
            action_space_type="standard"
        )

        assert actor.model_name == "custom/model"
        assert actor.device == "cpu"
        assert actor.temperature == 0.5
        assert actor.max_tokens == 256

    def test_init_sltp_action_space(self, mock_vllm_backend, sltp_action_map):
        """Test initialization with SLTP action space."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            action_space_type="futures_sltp",
            action_map=sltp_action_map
        )

        assert actor.action_space_type == "futures_sltp"
        assert actor.action_map == sltp_action_map
        assert actor.action_dict is None  # SLTP uses numeric actions

    def test_init_sltp_without_action_map_raises_error(self, mock_vllm_backend):
        """Test that SLTP without action_map raises ValueError."""
        from torchtrade.actor import LocalLLMActor

        with pytest.raises(ValueError, match="action_map required"):
            LocalLLMActor(action_space_type="futures_sltp")

    def test_init_transformers_backend(self, mock_transformers_backend):
        """Test initialization with transformers backend."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(backend="transformers")

        assert actor.backend == "transformers"
        assert actor.llm is not None

    def test_init_vllm_fallback_to_transformers(self):
        """Test that vllm falls back to transformers when unavailable."""
        import sys
        from types import ModuleType
        from torchtrade.actor import LocalLLMActor

        # Mock vllm to raise ImportError
        vllm_module = ModuleType("vllm")
        vllm_module.LLM = Mock(side_effect=ImportError("vllm not available"))
        sys.modules["vllm"] = vllm_module

        # Mock transformers as available
        transformers_module = ModuleType("transformers")
        transformers_module.AutoTokenizer = MockTokenizer
        transformers_module.AutoModelForCausalLM = Mock()
        transformers_module.AutoModelForCausalLM.from_pretrained = Mock(return_value=Mock())
        transformers_module.pipeline = MockTransformersPipeline
        transformers_module.BitsAndBytesConfig = Mock
        sys.modules["transformers"] = transformers_module

        try:
            actor = LocalLLMActor(backend="vllm")
            # Should have fallen back to transformers
            assert actor.backend == "transformers"
        finally:
            sys.modules.pop("vllm", None)
            sys.modules.pop("transformers", None)


class TestLocalLLMActorPromptConstruction:
    """Tests for prompt construction methods."""

    def test_construct_account_state_standard(self, mock_vllm_backend, sample_tensordict_standard):
        """Test account state construction for standard 7-element state."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor.construct_account_state(sample_tensordict_standard)

        assert "Current account state:" in result
        assert "cash:" in result
        assert "position_size:" in result
        assert "holding_time:" in result
        # Should NOT have futures-specific fields
        assert "leverage:" not in result
        assert "liquidation_price:" not in result

    def test_construct_account_state_futures(self, mock_vllm_backend, sample_tensordict_futures):
        """Test account state construction for futures 10-element state."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor.construct_account_state(sample_tensordict_futures)

        assert "Current account state:" in result
        assert "cash:" in result
        assert "leverage:" in result
        assert "margin_ratio:" in result
        assert "liquidation_price:" in result
        assert "holding_time:" in result

    def test_construct_market_data(self, mock_vllm_backend, sample_tensordict_standard):
        """Test market data construction."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor.construct_market_data(sample_tensordict_standard)

        assert "Current market data:" in result
        assert "market_data_1Minute_12:" in result
        assert "market_data_5Minute_8:" in result
        assert "close" in result
        assert "open" in result
        assert "high" in result
        assert "low" in result
        assert "volume" in result

    def test_construct_prompt(self, mock_vllm_backend, sample_tensordict_standard):
        """Test full prompt construction."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor.construct_prompt(sample_tensordict_standard)

        # Should contain both account state and market data
        assert "Current account state:" in result
        assert "Current market data:" in result

    def test_format_system_prompt_standard(self, mock_vllm_backend, sample_tensordict_standard):
        """Test system prompt formatting for standard environment."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor._format_system_prompt(sample_tensordict_standard)

        assert "trading agent" in result.lower()
        assert "buy, sell, or hold" in result.lower()

    def test_format_system_prompt_futures(self, mock_vllm_backend, sample_tensordict_futures):
        """Test system prompt formatting for futures environment."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        result = actor._format_system_prompt(sample_tensordict_futures)

        assert "futures" in result.lower()
        assert "leverage" in result.lower()
        assert "liquidation" in result.lower()


class TestLocalLLMActorActionExtraction:
    """Tests for action extraction and mapping."""

    def test_extract_action_buy(self, mock_vllm_backend):
        """Test extracting buy action from response."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<think>Market looks good</think><answer>buy</answer>"

        action_str = actor.extract_action(response)
        assert action_str == "buy"

    def test_extract_action_sell(self, mock_vllm_backend):
        """Test extracting sell action from response."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<think>Overbought</think><answer>sell</answer>"

        action_str = actor.extract_action(response)
        assert action_str == "sell"

    def test_extract_action_hold(self, mock_vllm_backend):
        """Test extracting hold action from response."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<think>Uncertain</think><answer>hold</answer>"

        action_str = actor.extract_action(response)
        assert action_str == "hold"

    def test_extract_action_case_insensitive(self, mock_vllm_backend):
        """Test action extraction is case-insensitive."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<think>...</think><ANSWER>BUY</ANSWER>"

        action_str = actor.extract_action(response)
        assert action_str.lower() == "buy"

    def test_extract_action_numeric_sltp(self, mock_vllm_backend, sltp_action_map):
        """Test extracting numeric action for SLTP."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            action_space_type="futures_sltp",
            action_map=sltp_action_map
        )
        response = "<think>Going long with tight SL</think><answer>1</answer>"

        action_str = actor.extract_action(response)
        assert action_str == "1"

    def test_extract_action_no_answer_tag(self, mock_vllm_backend):
        """Test fallback when no answer tag found."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "I think we should buy but forgot the tags"

        action_str = actor.extract_action(response)
        assert action_str == "hold"  # Default fallback

    def test_extract_thinking(self, mock_vllm_backend):
        """Test extracting thinking from response."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<think>This is my reasoning</think><answer>buy</answer>"

        thinking = actor.extract_thinking(response)
        assert thinking == "This is my reasoning"

    def test_extract_thinking_none(self, mock_vllm_backend):
        """Test extracting thinking when not present."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        response = "<answer>buy</answer>"

        thinking = actor.extract_thinking(response)
        assert thinking is None

    def test_map_action_to_index_standard(self, mock_vllm_backend):
        """Test mapping action strings to indices for standard space."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        assert actor.map_action_to_index("buy") == 2
        assert actor.map_action_to_index("sell") == 0
        assert actor.map_action_to_index("hold") == 1

    def test_map_action_to_index_unknown(self, mock_vllm_backend):
        """Test mapping unknown action defaults to hold."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        # Unknown action should default to hold (1)
        assert actor.map_action_to_index("invalid") == 1

    def test_map_action_to_index_sltp(self, mock_vllm_backend, sltp_action_map):
        """Test mapping numeric actions for SLTP."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            action_space_type="futures_sltp",
            action_map=sltp_action_map
        )

        assert actor.map_action_to_index("0") == 0  # Hold/Close
        assert actor.map_action_to_index("1") == 1  # First long position
        assert actor.map_action_to_index("5") == 5  # First short position


class TestLocalLLMActorForward:
    """Tests for the full forward pass."""

    def test_forward_standard_action_space(self, mock_vllm_backend, sample_tensordict_standard):
        """Test forward pass with standard 3-action space."""
        from torchtrade.actor import LocalLLMActor

        # Create actor and mock generate to return buy action
        actor = LocalLLMActor()

        # Mock the generate method to return a buy action
        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<think>Bullish signal</think><answer>buy</answer>"

            result = actor.forward(sample_tensordict_standard)

            assert "action" in result.keys()
            assert result["action"].item() == 2  # buy = 2
            assert "thinking" in result.keys()
            assert "Bullish signal" in result["thinking"]

    def test_forward_sell_action(self, mock_vllm_backend, sample_tensordict_standard):
        """Test forward pass returns sell action correctly."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<think>Bearish</think><answer>sell</answer>"

            result = actor.forward(sample_tensordict_standard)

            assert result["action"].item() == 0  # sell = 0

    def test_forward_hold_action(self, mock_vllm_backend, sample_tensordict_standard):
        """Test forward pass returns hold action correctly."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<think>Uncertain</think><answer>hold</answer>"

            result = actor.forward(sample_tensordict_standard)

            assert result["action"].item() == 1  # hold = 1

    def test_forward_sltp_action_space(self, mock_vllm_backend, sample_tensordict_futures, sltp_action_map):
        """Test forward pass with SLTP action space."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            action_space_type="futures_sltp",
            action_map=sltp_action_map
        )

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<think>Going long</think><answer>1</answer>"

            result = actor.forward(sample_tensordict_futures)

            assert result["action"].item() == 1

    def test_forward_futures_account_state(self, mock_vllm_backend, sample_tensordict_futures):
        """Test forward pass correctly handles futures account state."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<answer>hold</answer>"

            result = actor.forward(sample_tensordict_futures)

            assert "action" in result.keys()
            # Verify generate was called (meaning prompt was constructed without error)
            mock_gen.assert_called_once()

    def test_forward_no_thinking_tag(self, mock_vllm_backend, sample_tensordict_standard):
        """Test forward pass when response has no thinking tag."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<answer>buy</answer>"

            result = actor.forward(sample_tensordict_standard)

            assert "action" in result.keys()
            # thinking should not be in result if not in response
            assert "thinking" not in result.keys()

    def test_call_delegates_to_forward(self, mock_vllm_backend, sample_tensordict_standard):
        """Test that __call__ delegates to forward."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<answer>hold</answer>"

            # Call via __call__
            result1 = actor(sample_tensordict_standard)
            # Call via forward
            result2 = actor.forward(sample_tensordict_standard.clone())

            # Both should work and produce actions
            assert "action" in result1.keys()
            assert "action" in result2.keys()


class TestLocalLLMActorBackends:
    """Tests for different backend implementations."""

    def test_generate_vllm_backend(self, mock_vllm_backend, sample_tensordict_standard):
        """Test generation with vllm backend."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(backend="vllm")

        # Should not raise error
        system_prompt = actor._format_system_prompt(sample_tensordict_standard)
        user_prompt = actor.construct_prompt(sample_tensordict_standard)
        response = actor.generate(system_prompt, user_prompt)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_transformers_backend(self, mock_transformers_backend, sample_tensordict_standard):
        """Test generation with transformers backend."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(backend="transformers")

        # Should not raise error
        system_prompt = actor._format_system_prompt(sample_tensordict_standard)
        user_prompt = actor.construct_prompt(sample_tensordict_standard)
        response = actor.generate(system_prompt, user_prompt)

        assert isinstance(response, str)
        assert len(response) > 0


class TestLocalLLMActorActionSpaceDescription:
    """Tests for action space description building."""

    def test_build_action_space_description_standard(self, mock_vllm_backend):
        """Test building action space description for standard space."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()
        instructions, format_str = actor._build_action_space_description()

        assert "buy, sell, or hold" in instructions.lower()
        assert format_str == "buy/sell/hold"

    def test_build_action_space_description_sltp(self, mock_vllm_backend, sltp_action_map):
        """Test building action space description for SLTP space."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            action_space_type="futures_sltp",
            action_map=sltp_action_map
        )
        instructions, format_str = actor._build_action_space_description()

        assert "0: Hold/Close" in instructions
        assert "long" in instructions.lower()
        assert "short" in instructions.lower()
        assert "SL=" in instructions
        assert "TP=" in instructions
        assert format_str == "action_number"


class TestLocalLLMActorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_account_state_size_raises_error(self, mock_vllm_backend):
        """Test that unknown account state size raises ValueError."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        # Create tensordict with wrong account state size
        bad_tensordict = TensorDict({
            "market_data_1Minute_12": torch.randn(1, 12, 5),
            "account_state": torch.tensor([[1.0, 2.0, 3.0]]),  # Only 3 elements
        }, batch_size=[])

        with pytest.raises(ValueError, match="Unknown account state size"):
            actor.construct_account_state(bad_tensordict)

    def test_debug_mode_prints_prompts(self, mock_vllm_backend, sample_tensordict_standard, capsys):
        """Test that debug mode prints prompts and responses."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(debug=True)

        with patch.object(actor, 'generate') as mock_gen:
            mock_gen.return_value = "<answer>buy</answer>"

            actor.forward(sample_tensordict_standard)

            captured = capsys.readouterr()
            assert "SYSTEM PROMPT" in captured.out
            assert "USER PROMPT" in captured.out
            assert "RESPONSE" in captured.out

    def test_missing_market_data_keys(self, mock_vllm_backend):
        """Test handling of tensordict with missing market data keys."""
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor()

        # TensorDict with no market data keys
        td = TensorDict({
            "account_state": torch.tensor([[1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]]),
        }, batch_size=[])

        # Should not crash, just return empty market data section
        result = actor.construct_market_data(td)
        assert "Current market data:" in result


# ============================================================================
# Integration Tests (with real models)
# ============================================================================


@pytest.mark.slow
class TestLocalLLMActorIntegration:
    """Integration tests with real small models.

    These tests actually load models and generate text, so they are:
    - Marked with @pytest.mark.slow (skip by default)
    - Skip if vllm/transformers not installed
    - Use smallest available model to minimize resource usage
    """

    @pytest.mark.skipif(
        not any([
            __import__("importlib.util").util.find_spec("vllm"),
            __import__("importlib.util").util.find_spec("transformers")
        ]),
        reason="Neither vllm nor transformers available"
    )
    def test_integration_real_model_generate(self):
        """Test LocalLLMActor with real small model end-to-end.

        This test:
        1. Loads a real small model (Qwen/Qwen2.5-0.5B-Instruct)
        2. Creates a sample TensorDict
        3. Generates a real trading decision
        4. Validates the output format

        Note: This test takes ~30s-60s depending on hardware and downloads the model (~500MB).
        """
        from torchtrade.actor import LocalLLMActor

        # Use smallest available model
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        # Try vllm first, fall back to transformers
        try:
            actor = LocalLLMActor(
                model=model_name,
                backend="vllm",
                device="cuda" if torch.cuda.is_available() else "cpu",
                debug=False,
                temperature=0.1,  # Low temperature for deterministic output
                max_tokens=128,
            )
        except ImportError:
            # Fall back to transformers
            actor = LocalLLMActor(
                model=model_name,
                backend="transformers",
                device="cuda" if torch.cuda.is_available() else "cpu",
                debug=False,
                temperature=0.1,
                max_tokens=128,
            )

        # Create sample tensordict
        td = TensorDict({
            "market_data_1Minute_12": torch.randn(1, 12, 5),
            "market_data_5Minute_8": torch.randn(1, 8, 5),
            "account_state": torch.tensor([[1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0]]),
        }, batch_size=[])

        # Generate action (this actually runs inference)
        result = actor(td)

        # Validate output
        assert "action" in result.keys(), "Result should contain 'action' key"
        assert result["action"].dtype == torch.long, "Action should be long tensor"
        assert result["action"].item() in [0, 1, 2], "Action should be valid index (0=sell, 1=hold, 2=buy)"

        # If model generated thinking, it should be a string
        if "thinking" in result.keys():
            assert isinstance(result["thinking"], str), "Thinking should be a string"
            assert len(result["thinking"]) > 0, "Thinking should not be empty"

    @pytest.mark.skipif(
        not __import__("importlib.util").util.find_spec("transformers"),
        reason="transformers not available"
    )
    def test_integration_transformers_backend(self):
        """Test LocalLLMActor with real transformers backend.

        This specifically tests the transformers pipeline path with a real model.
        """
        from torchtrade.actor import LocalLLMActor

        actor = LocalLLMActor(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            backend="transformers",
            device="cpu",  # CPU only to avoid CUDA issues in CI
            debug=False,
            temperature=0.1,
            max_tokens=64,
        )

        assert actor.backend == "transformers"
        assert actor.llm is not None

        # Test generation
        td = TensorDict({
            "market_data_1Minute_12": torch.randn(1, 12, 5),
            "account_state": torch.tensor([[1000.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0]]),
        }, batch_size=[])

        result = actor(td)
        assert "action" in result.keys()
        assert result["action"].item() in [0, 1, 2]
