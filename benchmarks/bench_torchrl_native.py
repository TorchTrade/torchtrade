"""TorchRL-native vLLM benchmark — apples-to-apples with our LocalLLMActor path.

Measures TorchRL's native `vLLMWrapper` (and optionally `AsyncVLLM`) processing the
EXACT same elements our `LocalLLMActor` processes: the same system+user prompt,
chat-templated the same way, the same SamplingParams (temperature, max_tokens,
stop=["</answer>"]). The only thing that varies vs `bench_vllm_standalone.py`'s
`ours-batched` engine is the generation front-end (torchrl vLLMWrapper's TensorDict
I/O + token re-decode vs a direct `vllm.LLM.generate`). So comparing this script's
numbers to `ours-batched` isolates torchrl's wrapper overhead.

Runnable only with a torchrl whose vLLM backend matches the installed vLLM. torchrl
0.10.1 (our repo pin) hard-imports `vllm.worker.worker.Worker` (removed in vllm>=0.13)
and cannot load; torchrl >=0.13 works with vllm 0.14. Inside the DGX-Spark vLLM image:

    docker run --rm --gpus all --entrypoint bash \
      -v $PWD:/work -w /work -v ~/.cache/huggingface:/root/.cache/huggingface \
      avarok/vllm-dgx-spark:v11 -c \
      "pip install -q --no-deps torchrl tensordict pyvers orjson && \
       python benchmarks/bench_torchrl_native.py \
         --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct \
         --batch-sizes 1 4 16 32 64 128 --trials 10"

Engines:
  torchrl-sync   — vLLMWrapper(vllm.LLM, input_mode="text")
  torchrl-async  — vLLMWrapper(AsyncVLLM, ...) (needs ray; skipped if unavailable)
"""
from __future__ import annotations

import argparse
import statistics
import time

SYSTEM_PROMPT = (
    "You are a trading agent. Think step by step, then output your action as "
    "<answer>N</answer> where N is 0, 1, or 2."
)


def make_prompts(n):
    base = ("Account state: exposure_pct=0.{i}, position_direction=0.0.\n"
            "Market data (close): {series}\nChoose an action.")
    out = []
    for i in range(n):
        series = " ".join(str(100 + ((i * 7 + t) % 20)) for t in range(48))
        out.append(base.format(i=i % 10, series=series))
    return out


def _percentiles_ms(durations):
    if len(durations) < 2:
        v = durations[0] * 1000 if durations else float("nan")
        return v, v, v
    ms = sorted(d * 1000 for d in durations)
    q = statistics.quantiles(ms, n=100)
    return q[49], q[94], q[98]


def _extract_responses(out, n):
    """Robustly pull N response strings from a vLLMWrapper output tensordict."""
    txt = out["text"]
    resp = getattr(txt, "response", None)
    if resp is None:
        resp = out.get(("text", "response"))
    return [resp[i] for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct"])
    p.add_argument("--engines", nargs="+", default=["torchrl-sync"],
                   choices=["torchrl-sync", "torchrl-async"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--num-replicas", type=int, default=1)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from tensordict import TensorDict
    from torchrl.modules.llm import vLLMWrapper

    gk = {"max_new_tokens": args.max_tokens, "temperature": args.temperature,
          "stop": ["</answer>"]}

    print("model,engine,N,wall_s,decisions_per_s,tokens_per_s,p50_ms,p95_ms,p99_ms", flush=True)

    for model in args.models:
        count_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # Same chat templating LocalLLMActor uses (system + user, generation prompt).
        chat_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # torchrl's vLLMWrapper re-decodes tokens INCLUDING special tokens (e.g.
        # <|im_end|>), whereas raw vllm .text strips them. Strip them here so the
        # generated-token count matches the ours-* path exactly (apples-to-apples).
        _special = list(count_tok.all_special_tokens)

        def gen_token_count(resp):
            for s in _special:
                resp = resp.replace(s, "")
            return len(count_tok(resp, add_special_tokens=False).input_ids)

        def templ(user):
            return chat_tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)

        for engine in args.engines:
            if engine == "torchrl-sync":
                import vllm
                llm = vllm.LLM(model=model, dtype="bfloat16",
                               gpu_memory_utilization=args.gpu_memory_utilization,
                               max_model_len=args.max_model_len)
                wrapper = vLLMWrapper(llm, input_mode="text", generate=True,
                                      return_log_probs=False, generate_kwargs=gk)
                handle = llm
            else:
                import ray
                from torchrl.modules.llm.backends.vllm import AsyncVLLM
                if not ray.is_initialized():
                    ray.init()
                engine_obj = AsyncVLLM.from_pretrained(
                    model, num_devices=1, num_replicas=args.num_replicas)
                wrapper = vLLMWrapper(engine_obj, input_mode="text", generate=True,
                                      return_log_probs=False, generate_kwargs=gk)
                handle = engine_obj

            def run(prompts):
                td = TensorDict({("text", "prompt"): [templ(u) for u in prompts]},
                                batch_size=[len(prompts)])
                out = wrapper(td)
                return _extract_responses(out, len(prompts))

            for n in args.batch_sizes:
                prompts = make_prompts(n)
                for _ in range(args.warmup):
                    run(prompts)
                durations, total_tokens = [], 0
                for _ in range(args.trials):
                    t0 = time.perf_counter()
                    responses = run(prompts)
                    durations.append(time.perf_counter() - t0)
                    total_tokens += sum(gen_token_count(r) for r in responses)
                med = statistics.median(durations)
                total_wall = sum(durations)
                p50, p95, p99 = _percentiles_ms(durations)
                dps = n / med if med > 0 else float("inf")
                tps = total_tokens / total_wall if total_wall > 0 else float("inf")
                print(f"{model},{engine},{n},{med:.6f},{dps:.3f},{tps:.3f},"
                      f"{p50:.2f},{p95:.2f},{p99:.2f}", flush=True)

            try:
                if hasattr(handle, "shutdown"):
                    handle.shutdown()
            except Exception:
                pass
            del wrapper, handle
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
