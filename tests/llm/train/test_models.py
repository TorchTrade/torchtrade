"""Non-GPU tests for build helpers in models.py."""
import warnings
from types import SimpleNamespace

import pytest

from torchtrade.llm.train.models import _warn_if_stochastic


def _model(**config_kwargs):
    return SimpleNamespace(config=SimpleNamespace(**config_kwargs))


@pytest.mark.parametrize("model,should_warn", [
    (_model(hidden_dropout=0.1), True),                                  # flat dropout
    (_model(text_config=SimpleNamespace(attention_dropout=0.2)), True),  # nested (multimodal LM)
    (_model(attention_dropout=0.0, hidden_dropout=0.0), False),          # all zero
    (SimpleNamespace(config=None), False),                              # no config -> no crash
], ids=["flat", "text_config", "zero", "no-config"])
def test_warn_if_stochastic(model, should_warn):
    """Silent-correctness guard: nonzero dropout runs active in train() mode (needed for gradient
    checkpointing) and corrupts GRPO's importance ratio, so build_train_policy must warn. This pins
    the substring/threshold filter AND the nested text_config scan — both otherwise invisible if they
    regress."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_stochastic(model)
    assert any("dropout" in str(w.message) for w in caught) is should_warn
