"""Standalone (torchtrade-free) vLLM throughput benchmark for the LLM-actor path.

Measures exactly what `LocalLLMActor` delegates to — `vllm.LLM.generate(...)` with
manual chat templating and `stop=["</answer>"]` — WITHOUT importing torchtrade
(whose package __init__ eagerly pulls in the full env/broker stack: torchrl,
alpaca, ccxt, ...). That heavy stack won't install cleanly inside a DGX-Spark
vLLM container, so this script reproduces the `ours-current` / `ours-batched`
engines from `bench_llm_actor.py` using raw vLLM. The numbers are equivalent:
`LocalLLMActor.generate_batch` adds only prompt-string building + a regex, which
is microseconds next to generation.

Engines:
  ours-current  — N sequential single-prompt generate() calls (today's behavior)
  ours-batched  — one vllm.LLM.generate([...N]) call (native continuous batching)

Reports per (model, engine, N): median wall, decisions/s, real tokens/s (HF
tokenizer), batch-completion latency p50/p95/p99, and a best-effort prefill/
decode split from vLLM RequestMetrics.

DGX Spark (GB10, aarch64, sm_121) usage — run inside the prebuilt vLLM image
(pip install vllm does NOT work on Spark):

    docker run --rm --gpus all --entrypoint python \
        -v $PWD:/work -w /work -v ~/.cache/huggingface:/root/.cache/huggingface \
        avarok/vllm-dgx-spark:v11 benchmarks/bench_vllm_standalone.py \
        --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --batch-sizes 1 4 16 32 64 128 --trials 20

Set VLLM_USE_V1=0 to try to recover the prefill/decode split (the v1 engine
leaves RequestMetrics timestamps unset).
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
    """N realistic-length trading user prompts (deterministic, content varied)."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct"])
    p.add_argument("--engines", nargs="+", default=["ours-current", "ours-batched"],
                   choices=["ours-current", "ours-batched"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=2048)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("model,engine,N,wall_s,decisions_per_s,tokens_per_s,p50_ms,p95_ms,p99_ms", flush=True)
    pd_rows = []  # (model, N, prefill_ms, decode_ms)

    for model in args.models:
        llm = LLM(model=model, dtype="bfloat16",
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=args.max_model_len)
        chat_tok = llm.get_tokenizer()
        count_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                            stop=["</answer>"])

        def templ(user):
            return chat_tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)

        def run_batched(prompts):
            outs = llm.generate([templ(u) for u in prompts], sp)
            return [o.outputs[0].text for o in outs]

        def run_current(prompts):
            texts = []
            for u in prompts:
                outs = llm.generate([templ(u)], sp)
                texts.append(outs[0].outputs[0].text)
            return texts

        runners = {"ours-current": run_current, "ours-batched": run_batched}

        for engine in args.engines:
            run = runners[engine]
            for n in args.batch_sizes:
                prompts = make_prompts(n)
                for _ in range(args.warmup):
                    run(prompts)
                durations, total_tokens = [], 0
                for _ in range(args.trials):
                    t0 = time.perf_counter()
                    responses = run(prompts)
                    durations.append(time.perf_counter() - t0)
                    total_tokens += sum(len(count_tok(r, add_special_tokens=False).input_ids)
                                        for r in responses)
                med = statistics.median(durations)
                total_wall = sum(durations)
                p50, p95, p99 = _percentiles_ms(durations)
                dps = n / med if med > 0 else float("inf")
                tps = total_tokens / total_wall if total_wall > 0 else float("inf")
                print(f"{model},{engine},{n},{med:.6f},{dps:.3f},{tps:.3f},"
                      f"{p50:.2f},{p95:.2f},{p99:.2f}", flush=True)

        # best-effort prefill/decode split (batched call, raw RequestMetrics)
        for n in args.batch_sizes:
            outs = llm.generate([templ(u) for u in make_prompts(n)], sp)
            pre, dec = [], []
            ok = True
            for o in outs:
                m = getattr(o, "metrics", None)
                if (m is None or getattr(m, "arrival_time", None) is None
                        or getattr(m, "first_token_time", None) is None
                        or getattr(m, "finished_time", None) is None):
                    ok = False
                    break
                pre.append(m.first_token_time - m.arrival_time)
                dec.append(m.finished_time - m.first_token_time)
            if ok and pre:
                pd_rows.append((model, n, statistics.mean(pre) * 1000, statistics.mean(dec) * 1000))
            else:
                pd_rows.append((model, n, None, None))

        del llm
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n# prefill/decode split (ours-batched)", flush=True)
    print("model,N,prefill_ms,decode_ms", flush=True)
    for model, n, pre, dec in pd_rows:
        if pre is None:
            print(f"{model},{n},n/a,n/a", flush=True)
        else:
            print(f"{model},{n},{pre:.2f},{dec:.2f}", flush=True)


if __name__ == "__main__":
    main()
