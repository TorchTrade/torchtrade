"""Benchmark: TorchTrade LocalLLMActor vs TorchRL-native vLLM machinery.

Holds prompts and <answer>N</answer> parsing constant; varies ONLY the
generation engine. Requires a GPU + `vllm` (pip install -e '.[llm]'); the
async engine additionally needs `ray` (pip install -e '.[bench]') and is
skipped with a warning if ray is unavailable.

Reports, per (model, engine, N): median wall time, decisions/s, real
tokens/s (via the model's HF tokenizer), and per-batch-completion latency
p50/p95/p99. For the ours-batched engine it also reports a best-effort
prefill/decode split read from raw vLLM RequestMetrics.

Usage:
    python benchmarks/bench_llm_actor.py --models Qwen/Qwen2.5-0.5B-Instruct \
        --batch-sizes 1 4 16 32 --trials 20 --max-tokens 128
"""
from __future__ import annotations

import argparse
import statistics
import time
import warnings


SYSTEM_PROMPT = (
    "You are a trading agent. Think step by step, then output your action as "
    "<answer>N</answer> where N is 0, 1, or 2."
)


def make_prompts(n: int) -> list:
    """N realistic-length trading user prompts (deterministic, content varied)."""
    base = (
        "Account state: exposure_pct=0.{i}, position_direction=0.0.\n"
        "Market data (close): {series}\n"
        "Choose an action."
    )
    prompts = []
    for i in range(n):
        series = " ".join(str(100 + ((i * 7 + t) % 20)) for t in range(48))
        prompts.append(base.format(i=i % 10, series=series))
    return prompts


def _percentiles_ms(durations):
    """p50/p95/p99 (in ms) of a list of second-durations. Needs >=2 samples."""
    if len(durations) < 2:
        v = durations[0] * 1000 if durations else float("nan")
        return v, v, v
    ms = sorted(d * 1000 for d in durations)
    q = statistics.quantiles(ms, n=100)  # 99 cut points splitting data into 100 groups
    return q[49], q[94], q[98]


def _make_token_counter(model):
    """Load the HF tokenizer ONCE per model and return a resp -> generated-token-count callable."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return lambda resp: len(tok(resp, add_special_tokens=False).input_ids)


def _median_stats(fn, prompts, trials, warmup, count_tokens):
    """Run fn(prompts) warmup times (untimed), then `trials` timed; return dict.

    Per-trial batch wall-time is BATCH-COMPLETION latency: time from submit
    until all N decisions in that trial are ready (for N=1, single-decision
    latency). p50/p95/p99 are computed over these per-trial durations. Token
    counting happens AFTER each trial's wall time is recorded, so tokenizing
    never leaks into the timed region.
    """
    for _ in range(warmup):
        fn(prompts)
    durations = []
    total_tokens = 0
    for _ in range(trials):
        t0 = time.perf_counter()
        responses = fn(prompts)
        durations.append(time.perf_counter() - t0)
        total_tokens += sum(count_tokens(r) for r in responses)
    total_wall = sum(durations)
    med = statistics.median(durations)
    n = len(prompts)
    p50, p95, p99 = _percentiles_ms(durations)
    return {
        "wall_s": med,
        "decisions_per_s": n / med if med > 0 else float("inf"),
        "tokens_per_s": total_tokens / total_wall if total_wall > 0 else float("inf"),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def _prefill_decode_stats(actor, prompts):
    """Best-effort prefill/decode split for the ours-batched engine (raw vLLM call).

    Reads per-request RequestMetrics (arrival_time, first_token_time,
    finished_time) from a direct actor.llm.generate() call. vLLM's v1 engine
    may leave `o.metrics` (or its fields) unset; in that case return
    (None, None) and warn once, letting the caller report n/a.
    """
    formatted = [actor._format_chat_prompt(SYSTEM_PROMPT, p) for p in prompts]
    outputs = actor.llm.generate(formatted, actor.sampling_params)
    prefill_s, decode_s = [], []
    for o in outputs:
        m = o.metrics
        if m is None or m.arrival_time is None or m.first_token_time is None or m.finished_time is None:
            if not _prefill_decode_stats.warned:
                warnings.warn("vLLM RequestMetrics unavailable (arrival_time/first_token_time/"
                              "finished_time missing) - prefill/decode split reported as n/a.")
                _prefill_decode_stats.warned = True
            return None, None
        prefill_s.append(m.first_token_time - m.arrival_time)
        decode_s.append(m.finished_time - m.first_token_time)
    return statistics.mean(prefill_s) * 1000, statistics.mean(decode_s) * 1000


_prefill_decode_stats.warned = False


def _build_local_actor(model, max_tokens, temperature, gpu_memory_utilization=0.9):
    """Construct the LocalLLMActor shared by the ours-current/ours-batched engines."""
    from torchtrade.actor import LocalLLMActor
    return LocalLLMActor(
        model=model, backend="vllm", max_tokens=max_tokens, temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        market_data_keys=["market_data_1Hour_48"], account_state_labels=["exposure_pct"],
        action_levels=[-1.0, 0.0, 1.0],
    )


def engine_ours_current(model, max_tokens, temperature, gpu_memory_utilization):
    """N sequential single-prompt generate() calls (today's behavior at N>1)."""
    actor = _build_local_actor(model, max_tokens, temperature, gpu_memory_utilization)
    def run(prompts):
        return [actor.generate(SYSTEM_PROMPT, p) for p in prompts]
    return run, actor


def engine_ours_batched(model, max_tokens, temperature, gpu_memory_utilization):
    """One native-vLLM batched generate_batch call."""
    actor = _build_local_actor(model, max_tokens, temperature, gpu_memory_utilization)
    def run(prompts):
        return actor.generate_batch(SYSTEM_PROMPT, prompts)
    return run, actor


def _wrapped_engine_run(wrapper):
    """Build a run(prompts) closure for a torchrl vLLMWrapper (text mode)."""
    from tensordict import TensorDict

    def run(prompts):
        td = TensorDict({("text", "prompt"): list(prompts)}, batch_size=[len(prompts)])
        out = wrapper(td)
        return [out["text"].response[i] for i in range(len(prompts))]
    return run


def engine_torchrl_sync(model, max_tokens, temperature, gpu_memory_utilization):
    """vLLMWrapper over a sync vllm.LLM (text mode, no log-probs)."""
    import vllm
    from torchrl.modules.llm import vLLMWrapper
    llm = vllm.LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
    generate_kwargs = {"max_new_tokens": max_tokens, "temperature": temperature,
                       "stop": ["</answer>"]}
    wrapper = vLLMWrapper(
        llm, input_mode="text", generate=True, return_log_probs=False,
        generate_kwargs=generate_kwargs,
    )
    return _wrapped_engine_run(wrapper), llm


def engine_torchrl_async(model, max_tokens, temperature, num_replicas, gpu_memory_utilization):
    """vLLMWrapper over an AsyncVLLM (Ray) engine. Caller guards for ray import."""
    import ray
    from torchrl.modules.llm import vLLMWrapper
    from torchrl.modules.llm.backends.vllm import AsyncVLLM
    if not ray.is_initialized():
        ray.init()
    engine = AsyncVLLM.from_pretrained(model, num_devices=1, num_replicas=num_replicas)
    generate_kwargs = {"max_new_tokens": max_tokens, "temperature": temperature,
                       "stop": ["</answer>"]}
    wrapper = vLLMWrapper(
        engine, input_mode="text", generate=True, return_log_probs=False,
        generate_kwargs=generate_kwargs,
    )
    return _wrapped_engine_run(wrapper), engine


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Single-model alias for --models (ignored if --models is passed).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Sweep the full engine x batch-size grid once per model. "
                        "Overrides --model if both are given.")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-replicas", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                   help="Fraction of GPU memory per engine (lower it on small GPUs).")
    p.add_argument("--engines", nargs="+", default=None,
                   choices=["ours-current", "ours-batched", "torchrl-sync", "torchrl-async"],
                   help="Subset of engines to run (default: all available). Run one per "
                        "process on small GPUs to avoid sequential-load OOM.")
    return p.parse_args()


def main():
    args = parse_args()
    models = args.models or [args.model]
    rows = []  # (model, engine, N, wall_s, decisions_per_s, tokens_per_s, p50_ms, p95_ms, p99_ms)
    pd_rows = []  # (model, N, prefill_ms|None, decode_ms|None) - ours-batched only

    gmu = args.gpu_memory_utilization

    try:
        import ray  # noqa: F401
        ray_available = True
    except ImportError:
        ray_available = False
        warnings.warn("ray not installed; skipping the torchrl-async (AsyncVLLM) engine. "
                      "Install with: pip install -e '.[bench]'")

    for model in models:
        count_tokens = _make_token_counter(model)

        registry = [
            ("ours-current", lambda: engine_ours_current(model, args.max_tokens, args.temperature, gmu)),
            ("ours-batched", lambda: engine_ours_batched(model, args.max_tokens, args.temperature, gmu)),
            ("torchrl-sync", lambda: engine_torchrl_sync(model, args.max_tokens, args.temperature, gmu)),
        ]
        if ray_available:
            registry.append(("torchrl-async",
                             lambda: engine_torchrl_async(model, args.max_tokens,
                                                          args.temperature, args.num_replicas, gmu)))

        engines = [(n, b) for (n, b) in registry if args.engines is None or n in args.engines]

        for name, builder in engines:
            try:
                run, handle = builder()
            except Exception as e:  # noqa: BLE001 - skip an engine we can't construct (missing dep, version mismatch, OOM)
                warnings.warn(f"skipping engine '{name}' for model '{model}': failed to construct "
                              f"({type(e).__name__}: {e})")
                continue
            for n in args.batch_sizes:
                prompts = make_prompts(n)
                stats = _median_stats(run, prompts, args.trials, args.warmup, count_tokens)
                rows.append((model, name, n, stats["wall_s"], stats["decisions_per_s"],
                            stats["tokens_per_s"], stats["p50_ms"], stats["p95_ms"], stats["p99_ms"]))
                print(f"{model:28s} {name:14s} N={n:<3d} wall={stats['wall_s']*1000:8.1f}ms "
                      f"decisions/s={stats['decisions_per_s']:8.2f} "
                      f"tokens/s={stats['tokens_per_s']:8.1f} "
                      f"p50={stats['p50_ms']:7.1f}ms p95={stats['p95_ms']:7.1f}ms p99={stats['p99_ms']:7.1f}ms")

                if name == "ours-batched":
                    try:
                        prefill_ms, decode_ms = _prefill_decode_stats(handle, prompts)
                    except Exception as e:  # noqa: BLE001 - best-effort, never crash the sweep
                        warnings.warn(f"prefill/decode measurement failed for model '{model}' N={n}: "
                                      f"({type(e).__name__}: {e})")
                        prefill_ms, decode_ms = None, None
                    pd_rows.append((model, n, prefill_ms, decode_ms))
            # Free the engine before building the next: vLLM/Ray hold GPU memory
            # and Ray workers that `del` alone does not reliably release.
            try:
                if hasattr(handle, "shutdown"):
                    handle.shutdown()
            except Exception:  # noqa: BLE001 - best-effort teardown for a bench script
                pass
            del run, handle
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    print("\nmodel,engine,N,wall_s,decisions_per_s,tokens_per_s,p50_ms,p95_ms,p99_ms")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.3f},{r[5]:.3f},{r[6]:.2f},{r[7]:.2f},{r[8]:.2f}")

    print("\n# prefill/decode split (ours-batched)")
    print("model,N,prefill_ms,decode_ms")
    for model_name, n, prefill_ms, decode_ms in pd_rows:
        pf = f"{prefill_ms:.2f}" if prefill_ms is not None else "n/a"
        dc = f"{decode_ms:.2f}" if decode_ms is not None else "n/a"
        print(f"{model_name},{n},{pf},{dc}")

    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
