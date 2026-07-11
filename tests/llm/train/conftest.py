"""Shared fixtures for tests/train."""

from unittest.mock import Mock
from types import ModuleType
import sys

import pytest


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
