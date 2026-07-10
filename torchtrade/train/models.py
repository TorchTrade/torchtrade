"""Model builders + weight sync for the GRPO LLMTrainer (validated on the DGX Spark).

Hybrid stack (torchrl wrappers over a plain vLLM engine, because torchrl's AsyncVLLM +
NCCL sync is broken against the vLLM that runs on GB10/sm_121):
  - inference policy: torchrl vLLMWrapper over a plain `vllm.LLM` (n=K generations),
  - train policy: torchrl TransformersWrapper(generate=False) over a LoRA/QLoRA HF model,
  - weight sync (LoRA/QLoRA): the vLLM engine loads the frozen 4-bit base ONCE; each step we
    save just the LoRA adapter and hot-swap it via a fresh `LoRARequest` (the base never
    changes, so re-pushing it is waste). "full" method falls back to a merged-weight sync.
"""
from __future__ import annotations

import os

import torch

from torchtrade.train.peft_config import build_peft_config


def _structured_outputs_kwargs(regex):
    """vLLM regex-constrained decoding kwarg (name changed across vLLM versions)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(regex=regex)}
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(regex=regex)}


def _make_vllm_wrapper_cls():
    """vLLMWrapper subclass patching two torchrl gaps the stock wrapper has no pass-through for:
    (1) forwards the structured-output kwarg (else guided decoding looks on but is silently off);
    (2) injects a per-step `LoRARequest` (set on the instance as `_lora_request`) into the
    generate call — `_call_generate` is the raw passthrough all generate paths converge on."""
    from torchrl.modules.llm import vLLMWrapper

    class _TorchTradeVLLMWrapper(vLLMWrapper):
        _lora_request = None  # trainer sets this each step for LoRA hot-swap

        @classmethod
        def _get_wrapper_specific_kwargs(cls, generate_kwargs, wrapper_type):
            out = super()._get_wrapper_specific_kwargs(generate_kwargs, wrapper_type)
            if wrapper_type == "vllm":
                for k in ("structured_outputs", "guided_decoding"):
                    if k in generate_kwargs:
                        out[k] = generate_kwargs[k]
            return out

        def _call_generate(self, *args, **kwargs):
            if self._lora_request is not None:
                kwargs["lora_request"] = self._lora_request
            return super()._call_generate(*args, **kwargs)

    return _TorchTradeVLLMWrapper


def build_inference_policy(model_name, tokenizer, gpu_memory_utilization=0.3,
                           max_model_len=2048, max_tokens=256, action_regex=None,
                           enable_lora=False, max_lora_rank=16):
    """torchrl vLLMWrapper over a plain vllm.LLM (rollout engine).

    `enable_lora` loads the base with vLLM LoRA support so the trainer can hot-swap adapters
    each step (`LoRARequest`); the base may be a bnb-4bit checkpoint (vLLM auto-detects the
    quantization). `action_regex` (e.g. `<answer>(0|..|9)</answer>`) constrains generation so
    every completion is a parseable action index (guided decoding).
    """
    from vllm import LLM

    llm_kwargs = dict(model=model_name, gpu_memory_utilization=gpu_memory_utilization,
                      enforce_eager=True, max_model_len=max_model_len)
    if enable_lora:
        llm_kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    engine = LLM(**llm_kwargs)

    generate_kwargs = {"max_tokens": max_tokens}
    if action_regex is not None:
        generate_kwargs.update(_structured_outputs_kwargs(action_regex))
    wrapper_cls = _make_vllm_wrapper_cls()
    policy = wrapper_cls(
        engine, input_mode="history", chat_template_name="qwen",
        return_log_probs=True, tokenizer=tokenizer, generate=True,
        generate_kwargs=generate_kwargs,
    )
    return engine, policy


def save_lora_adapter(hf, adapter_dir, step):
    """Save the current LoRA adapter and return a fresh `LoRARequest`. A NEW int id every step
    is mandatory: vLLM's adapter cache is keyed on the int id alone, so id reuse serves stale
    weights (verified in vllm/lora/worker_manager.py). Adapter-only save is ~tens of MB."""
    from vllm.lora.request import LoRARequest

    path = os.path.join(adapter_dir, f"step_{step}")
    hf.save_pretrained(path)
    return LoRARequest(f"step_{step}", step + 1, path)


def build_train_policy(model_name, tokenizer, method="qlora", lora_r=16, lora_alpha=32,
                       peft_config=None, device="cuda"):
    """LoRA/QLoRA HF model wrapped as a torchrl TransformersWrapper(generate=False).

    `input_mode="tokens"` (operate on the rollout's recorded tokens, avoiding the
    re-render token-count mismatch) — the validated recipe. QLoRA quantizes the same
    full-precision `model_name` to 4-bit on the fly (the vLLM rollout engine loads it bf16),
    then runs kbit-training prep + gradient checkpointing (needed at 8B depth to fit the
    backward).
    """
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, prepare_model_for_kbit_training
    from torchrl.modules.llm import TransformersWrapper

    cfg = build_peft_config(method, lora_r=lora_r, lora_alpha=lora_alpha, peft_config=peft_config)
    load_kwargs = {"dtype": torch.bfloat16}
    if cfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    hf = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if cfg["load_in_4bit"]:
        hf = prepare_model_for_kbit_training(hf, use_gradient_checkpointing=True)
    else:
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
            # bitsandbytes 4-bit quant-state buffers (absmax/quant_map/...): dropped because the
            # dequantized base is already folded into the merged weight above. Also drops a
            # base_layer.bias if present — correct only because build_peft_config uses bias="none"
            # (frozen bias stays as vLLM loaded it); revisit if a custom bias config is exposed.
            continue
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
