"""Map the user-facing `method` to a PEFT/quantization config."""

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]


def build_peft_config(method, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=None, peft_config=None):
    """method in {"full","lora","qlora"} -> {"peft_config": LoraConfig|None,
    "load_in_4bit": bool}. `peft_config` passthrough overrides the built LoraConfig."""
    if method not in ("full", "lora", "qlora"):
        raise ValueError(f"method must be 'full'|'lora'|'qlora', got {method!r}")
    if method == "full":
        return {"peft_config": None, "load_in_4bit": False}

    if peft_config is None:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules or _DEFAULT_TARGETS,
            task_type="CAUSAL_LM", bias="none",
        )
    return {"peft_config": peft_config, "load_in_4bit": method == "qlora"}
