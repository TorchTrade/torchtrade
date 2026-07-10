import pytest
from torchtrade.train.peft_config import build_peft_config


@pytest.mark.parametrize("method,has_lora,in_4bit", [
    ("full", False, False), ("lora", True, False), ("qlora", True, True),
], ids=["full", "lora", "qlora"])
def test_method_maps_to_config(method, has_lora, in_4bit):
    cfg = build_peft_config(method, lora_r=8, lora_alpha=16)
    assert cfg["load_in_4bit"] is in_4bit
    if has_lora:
        assert cfg["peft_config"] is not None
        assert cfg["peft_config"].r == 8 and cfg["peft_config"].lora_alpha == 16
        assert cfg["peft_config"].lora_dropout == 0.0  # deterministic forward for the GRPO ratio
    else:
        assert cfg["peft_config"] is None


def test_explicit_peft_config_passthrough():
    sentinel = object()
    cfg = build_peft_config("lora", peft_config=sentinel)
    assert cfg["peft_config"] is sentinel


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        build_peft_config("bogus")
