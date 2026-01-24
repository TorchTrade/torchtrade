"""Tests for LLMActor (OpenAI-based) covering critical gaps identified in PR review."""

from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import warnings
from tensordict import TensorDict

# Check if openai is available
try:
    from torchtrade.actor.llm_actor import _LLMModule, LLMActor
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="openai package not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    if not OPENAI_AVAILABLE:
        pytest.skip("openai not available")

    with patch('torchtrade.actor.llm_actor.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "<think>Market analysis</think><answer>half_invested</answer>"
        mock_client.responses.create.return_value = mock_response
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_dotenv_values():
    """Mock environment variables."""
    if not OPENAI_AVAILABLE:
        pytest.skip("openai not available")

    with patch('torchtrade.actor.llm_actor.dotenv_values') as mock_env:
        mock_env.return_value = {"OPENAI_API_KEY": "test-api-key"}
        yield mock_env


@pytest.fixture
def sample_market_data_standard():
    """Sample market data tensor with 5 features (standard OHLCV)."""
    return torch.randn(12, 5)  # 12 timesteps, 5 features


@pytest.fixture
def sample_market_data_custom():
    """Sample market data tensor with 3 custom features."""
    return torch.randn(10, 3)  # 10 timesteps, 3 features


@pytest.fixture
def sample_account_state():
    """Sample account state tensor (7 elements)."""
    return torch.tensor([1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0])


# ============================================================================
# Tests for _LLMModule: Custom Features (Priority 10)
# ============================================================================


class TestLLMModuleCustomFeatures:
    """Tests for custom feature_keys configurations (addresses critical gap #1)."""

    def test_default_feature_keys(self, mock_openai_client, mock_dotenv_values):
        """Test that default feature_keys is set correctly."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert module.feature_keys == ['close', 'open', 'high', 'low', 'volume']

    def test_custom_feature_keys_3_features(self, mock_openai_client, mock_dotenv_values):
        """Test with 3 custom features (close, volume, rsi)."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=['close', 'volume', 'rsi']
        )

        assert module.feature_keys == ['close', 'volume', 'rsi']

    def test_custom_feature_keys_8_features(self, mock_openai_client, mock_dotenv_values):
        """Test with 8 features including technical indicators."""
        feature_keys = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width']
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=feature_keys
        )

        assert module.feature_keys == feature_keys

    def test_construct_prompt_with_3_custom_features(self, mock_openai_client, mock_dotenv_values):
        """Test prompt construction with 3 custom features doesn't crash."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=['close', 'volume', 'rsi']
        )

        market_data = torch.randn(10, 3)  # 10 timesteps, 3 features
        account_state = torch.tensor([1000.0, 0.5])

        prompt = module.construct_prompt(market_data, account_state)

        # Verify header has correct features
        assert 'close' in prompt
        assert 'volume' in prompt
        assert 'rsi' in prompt
        # Verify no hardcoded 5-feature assumption
        assert prompt.count('|') > 0  # Has column separators

    def test_construct_prompt_with_8_custom_features(self, mock_openai_client, mock_dotenv_values):
        """Test prompt construction with 8 features."""
        feature_keys = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width']
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=feature_keys
        )

        market_data = torch.randn(10, 8)
        account_state = torch.tensor([1000.0, 0.5])

        prompt = module.construct_prompt(market_data, account_state)

        # Verify all 8 features are in header
        for feature in feature_keys:
            assert feature in prompt

    def test_assertion_catches_mismatched_shape_too_few_features(self, mock_openai_client, mock_dotenv_values):
        """Test that shape validation catches data with too few features."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=['close', 'open', 'high', 'low', 'volume']  # Expects 5
        )

        market_data = torch.randn(10, 3)  # Only 3 features
        account_state = torch.tensor([1000.0, 0.5])

        with pytest.raises(AssertionError, match="Expected market data shape"):
            module.construct_prompt(market_data, account_state)

    def test_assertion_catches_mismatched_shape_too_many_features(self, mock_openai_client, mock_dotenv_values):
        """Test that shape validation catches data with too many features."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=['close', 'volume', 'rsi']  # Expects 3
        )

        market_data = torch.randn(10, 5)  # 5 features instead of 3
        account_state = torch.tensor([1000.0, 0.5])

        with pytest.raises(AssertionError, match="Expected market data shape"):
            module.construct_prompt(market_data, account_state)

    def test_prompt_format_correctness_header_matches_rows(self, mock_openai_client, mock_dotenv_values):
        """Test that header columns match data row columns."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            feature_keys=['close', 'volume', 'rsi']
        )

        market_data = torch.ones(5, 3)  # 5 timesteps, 3 features, all ones for easy checking
        account_state = torch.tensor([1000.0, 0.5])

        prompt = module.construct_prompt(market_data, account_state)

        # Extract the market data section
        lines = prompt.split('\n')

        # Find header line (contains feature names)
        header_line = None
        for line in lines:
            if 'close' in line and 'volume' in line and 'rsi' in line:
                header_line = line
                break

        assert header_line is not None, "Header line should be found"

        # Count columns in header
        header_cols = len([col for col in header_line.split('|') if col.strip()])

        # Find a data row (line with numbers)
        data_line = None
        for line in lines:
            if '1.0' in line:  # Our test data is all ones
                data_line = line
                break

        assert data_line is not None, "Data line should be found"

        # Count columns in data row
        data_cols = len([col for col in data_line.split('|') if col.strip()])

        # Should match
        assert header_cols == data_cols, f"Header has {header_cols} cols but data has {data_cols} cols"


# ============================================================================
# Tests for LLMActor: Extraction Failure Handling (Priority 8)
# ============================================================================


class TestLLMActorExtractionFailures:
    """Tests for LLM response extraction failures and fallback behavior."""

    def test_extract_action_success(self, mock_openai_client, mock_dotenv_values):
        """Test successful action extraction."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        response = "<think>Analysis</think><answer>half_invested</answer>"
        action = module.extract_action(response)

        assert action == "half_invested"

    def test_extract_action_failure_uses_initial_action(self, mock_openai_client, mock_dotenv_values):
        """Test that extraction failure on first call uses initial_action."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            initial_action="half_invested"
        )

        # First call with malformed response
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action = module.extract_action("No tags here just rambling")

            assert action == "half_invested"
            assert len(w) == 1
            assert "Failed to extract action" in str(w[0].message)

    def test_extract_action_failure_uses_last_successful(self, mock_openai_client, mock_dotenv_values):
        """Test that extraction failure after success uses last successful action."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        # First successful extraction
        module.extract_action("<answer>fully_invested</answer>")
        assert module.last_action == "fully_invested"

        # Second call with malformed response should use last successful
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action = module.extract_action("Malformed response")

            assert action == "fully_invested"
            assert len(w) == 1
            assert "Maintaining previous position: fully_invested" in str(w[0].message)

    def test_extract_action_default_fallback_is_all_cash(self, mock_openai_client, mock_dotenv_values):
        """Test that default fallback when initial_action not specified is 'all_cash'."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        # No successful extractions yet, should use default
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            action = module.extract_action("Bad response")

            assert action == "all_cash"

    def test_extract_action_warning_emitted(self, mock_openai_client, mock_dotenv_values):
        """Test that warning is emitted when extraction fails."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            module.extract_action("No answer tag")

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Failed to extract action from LLM response" in str(w[0].message)

    def test_forward_with_invalid_action_name_raises_keyerror(self, mock_openai_client, mock_dotenv_values):
        """Test that invalid action name raises KeyError."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        # Mock extract_action to return invalid action
        with patch.object(module, 'extract_action', return_value='invalid_action'):
            market_data = torch.randn(10, 5)
            account_state = torch.tensor([1000.0, 0.5])

            with pytest.raises(KeyError):
                module.forward(market_data, account_state)

    def test_custom_action_dict_extraction(self, mock_openai_client, mock_dotenv_values):
        """Test extraction with custom action_dict."""
        custom_actions = {
            "exit_all": 0,
            "quarter": 1,
            "half": 2,
            "three_quarters": 3,
            "full": 4
        }

        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            action_dict=custom_actions,
            initial_action="quarter"
        )

        # Test successful extraction
        action = module.extract_action("<answer>three_quarters</answer>")
        assert action == "three_quarters"

        # Test fallback
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            action = module.extract_action("bad response")
            assert action == "three_quarters"  # Uses last successful


# ============================================================================
# Tests for LLMActor: TorchRL Spec Compatibility (Priority 7)
# ============================================================================


class TestLLMActorSpecCompatibility:
    """Tests for TorchRL spec validation and compatibility."""

    def test_action_spec_shape(self, mock_openai_client, mock_dotenv_values):
        """Test that action spec has correct shape."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert actor.spec["action"].shape == torch.Size([1])

    def test_action_spec_n_matches_action_dict_length(self, mock_openai_client, mock_dotenv_values):
        """Test that action spec n matches number of actions."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        # Default has 3 actions
        assert actor.spec["action"].n == 3

    def test_action_spec_n_with_custom_action_dict(self, mock_openai_client, mock_dotenv_values):
        """Test action spec n with custom action_dict."""
        custom_actions = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            action_dict=custom_actions
        )

        assert actor.spec["action"].n == 5

    def test_thinking_spec_is_unbounded(self, mock_openai_client, mock_dotenv_values):
        """Test that thinking spec is Unbounded."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        from torchrl.data import Unbounded
        assert isinstance(actor.spec["thinking"], Unbounded)

    def test_spec_shape_is_empty(self, mock_openai_client, mock_dotenv_values):
        """Test that composite spec shape is empty."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert actor.spec.shape == torch.Size([])

    def test_output_conforms_to_spec(self, mock_openai_client, mock_dotenv_values):
        """Test that forward output conforms to spec."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        action_tensor, thinking = actor.module.forward(market_data, account_state)

        # Action should be long tensor with shape [1]
        assert action_tensor.dtype == torch.long
        assert action_tensor.shape == torch.Size([1])
        assert 0 <= action_tensor.item() < 3  # Valid index

        # Thinking should be string
        assert isinstance(thinking, str)


# ============================================================================
# Tests for Account State Handling
# ============================================================================


class TestLLMActorAccountState:
    """Tests for account state tensor validation."""

    def test_construct_prompt_standard_account_state_7_elements(self, mock_openai_client, mock_dotenv_values):
        """Test prompt construction with 7-element account state (standard)."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position_size', 'position_value', 'entry_price',
                          'current_price', 'unrealized_pnlpct', 'holding_time']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0])

        prompt = module.construct_prompt(market_data, account_state)

        assert 'cash: 1000.0' in prompt
        assert 'position_size: 0.5' in prompt
        assert 'holding_time: 5.0' in prompt

    def test_construct_prompt_futures_account_state_10_elements(self, mock_openai_client, mock_dotenv_values):
        """Test prompt construction with 10-element account state (futures)."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position_size', 'position_value', 'entry_price',
                          'current_price', 'unrealized_pnlpct', 'leverage',
                          'margin_ratio', 'liquidation_price', 'holding_time']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5, 50.0, 100.0, 102.0, 0.02, 5.0, 0.2, 95.0, 5.0])

        prompt = module.construct_prompt(market_data, account_state)

        assert 'leverage: 5.0' in prompt
        assert 'margin_ratio: 0.2' in prompt
        assert 'liquidation_price: 95.0' in prompt

    def test_assertion_catches_account_state_size_mismatch(self, mock_openai_client, mock_dotenv_values):
        """Test that assertion catches account state size mismatch."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position_size', 'position_value']  # Expects 3
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])  # Only 2 elements

        with pytest.raises(AssertionError, match="Expected account state shape"):
            module.construct_prompt(market_data, account_state)


# ============================================================================
# Tests for Multiple Market Data Keys
# ============================================================================


class TestLLMActorMultipleMarketData:
    """Tests for handling multiple market data timeframes."""

    def test_construct_prompt_with_multiple_timeframes(self, mock_openai_client, mock_dotenv_values):
        """Test prompt construction with multiple market data timeframes."""
        module = _LLMModule(
            market_data_keys=['market_data_1Minute_12', 'market_data_5Minute_8'],
            account_state=['cash', 'position']
        )

        market_data_1min = torch.randn(12, 5)
        market_data_5min = torch.randn(8, 5)
        account_state = torch.tensor([1000.0, 0.5])

        prompt = module.construct_prompt(market_data_1min, market_data_5min, account_state)

        assert 'market_data_1Minute_12:' in prompt
        assert 'market_data_5Minute_8:' in prompt

    def test_llm_actor_in_keys_correct(self, mock_openai_client, mock_dotenv_values):
        """Test that LLMActor in_keys includes all market data keys + account_state."""
        actor = LLMActor(
            market_data_keys=['market_data_1Minute_12', 'market_data_5Minute_8'],
            account_state=['cash', 'position']
        )

        assert actor.in_keys == ['market_data_1Minute_12', 'market_data_5Minute_8', 'account_state']

    def test_llm_actor_out_keys_correct(self, mock_openai_client, mock_dotenv_values):
        """Test that LLMActor out_keys includes action and thinking."""
        actor = LLMActor(
            market_data_keys=['market_data_1Minute_12'],
            account_state=['cash', 'position']
        )

        assert actor.out_keys == ['action', 'thinking']


# ============================================================================
# Tests for System Prompt Generation
# ============================================================================


class TestLLMActorSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_includes_available_actions(self, mock_openai_client, mock_dotenv_values):
        """Test that system prompt includes available actions."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert 'all_cash, half_invested, fully_invested' in module.system_prompt

    def test_system_prompt_with_custom_actions(self, mock_openai_client, mock_dotenv_values):
        """Test system prompt with custom action_dict."""
        custom_actions = {"exit": 0, "enter": 1}

        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            action_dict=custom_actions
        )

        assert 'exit, enter' in module.system_prompt

    def test_system_prompt_uses_symbol_and_execute_on(self, mock_openai_client, mock_dotenv_values):
        """Test that system prompt can be formatted with symbol and execute_on."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            symbol="ETH/USD",
            execute_on="15Minute"
        )

        formatted = module.system_prompt.format(symbol="ETH/USD", execute_on="15Minute")

        assert "ETH/USD" in formatted
        assert "15Minute" in formatted


# ============================================================================
# Tests for Debug Mode
# ============================================================================


class TestLLMActorDebugMode:
    """Tests for debug mode output."""

    def test_debug_mode_prints_prompts(self, mock_openai_client, mock_dotenv_values, capsys):
        """Test that debug mode prints system prompt, prompt, and response."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            debug=True
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        module.forward(market_data, account_state)

        captured = capsys.readouterr()

        assert 'SYSTEM PROMPT:' in captured.out
        assert 'PROMPT:' in captured.out
        assert 'RESPONSE:' in captured.out

    def test_debug_mode_false_no_output(self, mock_openai_client, mock_dotenv_values, capsys):
        """Test that debug=False produces no output."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            debug=False
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        module.forward(market_data, account_state)

        captured = capsys.readouterr()

        assert 'SYSTEM PROMPT:' not in captured.out
        assert 'PROMPT:' not in captured.out


# ============================================================================
# Tests for Prompt Output (SFT/Reproducibility)
# ============================================================================


class TestLLMActorPromptOutput:
    """Tests for prompt output feature (for SFT datasets and reproducibility)."""

    def test_forward_returns_four_outputs(self, mock_openai_client, mock_dotenv_values):
        """Test that forward returns action, thinking, system_prompt, and user_prompt."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        outputs = module.forward(market_data, account_state)

        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    def test_system_prompt_is_string(self, mock_openai_client, mock_dotenv_values):
        """Test that system_prompt output is a string."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        _, _, system_prompt, _ = module.forward(market_data, account_state)

        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0

    def test_user_prompt_is_string(self, mock_openai_client, mock_dotenv_values):
        """Test that user_prompt output is a string."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        _, _, _, user_prompt = module.forward(market_data, account_state)

        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0

    def test_system_prompt_contains_formatted_values(self, mock_openai_client, mock_dotenv_values):
        """Test that system_prompt contains symbol and execute_on."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            symbol="ETH/USD",
            execute_on="15Minute"
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        _, _, system_prompt, _ = module.forward(market_data, account_state)

        assert "ETH/USD" in system_prompt
        assert "15Minute" in system_prompt

    def test_user_prompt_contains_market_data_and_account_state(self, mock_openai_client, mock_dotenv_values):
        """Test that user_prompt contains market data and account state."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        _, _, _, user_prompt = module.forward(market_data, account_state)

        assert "Current account state" in user_prompt
        assert "Current market data" in user_prompt
        assert "cash" in user_prompt
        assert "market_data_1Min_10" in user_prompt

    def test_llm_actor_out_keys_includes_prompts(self, mock_openai_client, mock_dotenv_values):
        """Test that LLMActor out_keys includes system_prompt and user_prompt."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert "system_prompt" in actor.out_keys
        assert "user_prompt" in actor.out_keys
        assert actor.out_keys == ['action', 'thinking', 'system_prompt', 'user_prompt']

    def test_llm_actor_spec_includes_prompt_specs(self, mock_openai_client, mock_dotenv_values):
        """Test that LLMActor spec includes system_prompt and user_prompt specs."""
        actor = LLMActor(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        assert "system_prompt" in actor.spec.keys()
        assert "user_prompt" in actor.spec.keys()

        from torchrl.data import Unbounded
        assert isinstance(actor.spec["system_prompt"], Unbounded)
        assert isinstance(actor.spec["user_prompt"], Unbounded)

    def test_prompts_enable_sft_reproducibility(self, mock_openai_client, mock_dotenv_values):
        """Test that captured prompts can be used for SFT dataset construction."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position'],
            symbol="BTC/USD",
            execute_on="5Minute"
        )

        market_data = torch.randn(10, 5)
        account_state = torch.tensor([1000.0, 0.5])

        action_tensor, thinking, system_prompt, user_prompt = module.forward(market_data, account_state)

        # Verify we can construct an SFT training example
        sft_example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": thinking}
            ],
            "action": action_tensor.item()
        }

        # Verify structure
        assert len(sft_example["messages"]) == 3
        assert sft_example["messages"][0]["role"] == "system"
        assert sft_example["messages"][1]["role"] == "user"
        assert sft_example["messages"][2]["role"] == "assistant"
        assert isinstance(sft_example["action"], int)

    def test_prompts_different_across_timesteps(self, mock_openai_client, mock_dotenv_values):
        """Test that user_prompt changes with different market data."""
        module = _LLMModule(
            market_data_keys=['market_data_1Min_10'],
            account_state=['cash', 'position']
        )

        # Timestep 1
        market_data_1 = torch.randn(10, 5)
        account_state_1 = torch.tensor([1000.0, 0.5])
        _, _, _, user_prompt_1 = module.forward(market_data_1, account_state_1)

        # Timestep 2 with different data
        market_data_2 = torch.randn(10, 5)
        account_state_2 = torch.tensor([950.0, 0.3])
        _, _, _, user_prompt_2 = module.forward(market_data_2, account_state_2)

        # Prompts should be different (different market data and account state)
        assert user_prompt_1 != user_prompt_2
