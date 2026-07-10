"""Model builders + weight sync for the GRPO LLMTrainer (validated on the DGX Spark).

Hybrid stack (torchrl wrappers over a plain vLLM engine, because torchrl's AsyncVLLM +
NCCL sync is broken against the vLLM that runs on GB10/sm_121):
  - inference policy: torchrl vLLMWrapper over a plain `vllm.LLM` (n=K generations),
  - train policy: torchrl TransformersWrapper(generate=False) over a LoRA/QLoRA HF model,
  - weight sync: merge LoRA -> push merged base weights into the vLLM engine via
    `collective_rpc` (disk path over RPC; worker loads real tensors locally).
"""
from __future__ import annotations

import torch

from torchtrade.train.peft_config import build_peft_config


def build_inference_policy(model_name, tokenizer, gpu_memory_utilization=0.3,
                           max_model_len=2048, max_tokens=256):
    """torchrl vLLMWrapper over a plain vllm.LLM (rollout engine)."""
    from vllm import LLM
    from torchrl.modules.llm import vLLMWrapper

    engine = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization,
                 enforce_eager=True, max_model_len=max_model_len)
    policy = vLLMWrapper(
        engine, input_mode="history", chat_template_name="qwen",
        return_log_probs=True, tokenizer=tokenizer, generate=True,
        generate_kwargs={"max_tokens": max_tokens},
    )
    return engine, policy


def build_train_policy(model_name, tokenizer, method="qlora", lora_r=16, lora_alpha=32,
                       peft_config=None, device="cuda"):
    """LoRA/QLoRA HF model wrapped as a torchrl TransformersWrapper(generate=False).

    `input_mode="tokens"` (operate on the rollout's recorded tokens, avoiding the
    re-render token-count mismatch) — the validated recipe.
    """
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    from torchrl.modules.llm import TransformersWrapper

    cfg = build_peft_config(method, lora_r=lora_r, lora_alpha=lora_alpha, peft_config=peft_config)
    load_kwargs = {"dtype": torch.bfloat16}
    if cfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not cfg["load_in_4bit"]:
        hf = hf.to(device)
    if cfg["peft_config"] is not None:
        hf = get_peft_model(hf, cfg["peft_config"])
    policy = TransformersWrapper(
        hf, tokenizer=tokenizer, input_mode="tokens", generate=False,
        return_log_probs=True, pad_output=False, device=device,
    )
    return hf, policy


def _merged_state_dict(hf):
    """Merged (base + LoRA-delta) weights in vLLM/HF naming, WITHOUT mutating `hf`.

    Avoids peft's `merge_adapter()`/`unmerge_adapter()`, which for a 4-bit (QLoRA) base
    dequantize->add->REQUANTIZE the frozen base in place every call — a documented-lossy
    round-trip that, called each training step, drifts the base weights over training.
    Instead we dequantize a *copy* of each LoRA layer's base and add its delta on the copy.
    """
    from peft import PeftModel

    if not isinstance(hf, PeftModel):  # method="full": plain HF model, nothing to merge
        return {k: v.detach().to(torch.bfloat16).cpu() for k, v in hf.state_dict().items()}

    from peft.tuners.lora import LoraLayer

    merged = {}  # module path -> (base + delta), non-mutating
    for name, module in hf.named_modules():
        if isinstance(module, LoraLayer) and module.active_adapters:
            base = module.base_layer.weight
            quant_state = getattr(base, "quant_state", None)
            if quant_state is not None:
                import bitsandbytes.functional as bnb_f
                base = bnb_f.dequantize_4bit(base.data, quant_state)
            delta = module.get_delta_weight(module.active_adapters[0])
            merged[name] = base.to(delta.dtype) + delta

    state = {}
    for k, v in hf.state_dict().items():
        if "lora_" in k:
            continue
        if k.endswith(".base_layer.weight"):
            module_path = k[: -len(".base_layer.weight")]
            weight = merged.get(module_path, v)
            name = module_path.replace("base_model.model.", "") + ".weight"
        elif ".base_layer." in k:
            continue  # bitsandbytes 4-bit quant-state buffers (absmax/quant_map/...) — the
                      # dequantized base is already folded into the merged weight above
        else:
            weight, name = v, k.replace("base_model.model.", "")
        state[name] = weight.detach().to(torch.bfloat16).cpu()
    return state


def sync_weights_to_vllm(engine, hf, path="/tmp/_grpo_merged.pt"):
    """Push the trained weights into the vLLM rollout engine (the validated sync).

    Build the merged base+LoRA state_dict (non-mutating, see `_merged_state_dict`), write it
    to disk, then `collective_rpc` a loader that reads the REAL tensors from disk on the
    worker (only the path crosses the RPC boundary — passing tensors through RPC mangles
    them). Requires env `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Returns the # weights loaded.
    """
    torch.save(_merged_state_dict(hf), path)

    def _load(worker, p):
        import torch as _torch
        weights = _torch.load(p, map_location="cpu")
        worker.model_runner.model.load_weights(list(weights.items()))
        return len(weights)

    return engine.collective_rpc(_load, args=(path,))[0]
