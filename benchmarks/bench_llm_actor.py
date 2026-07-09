"""Benchmark: TorchTrade LocalLLMActor vs TorchRL-native vLLM machinery.

Holds prompts and <answer>N</answer> parsing constant; varies ONLY the
generation engine. Requires a GPU + `vllm` (pip install -e '.[llm]'); the
async engine additionally needs `ray` (pip install -e '.[bench]') and is
skipped with a warning if ray is unavailable.

Usage:
    python benchmarks/bench_llm_actor.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --batch-sizes 1 4 16 32 --trials 5 --max-tokens 128
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


def _median_stats(fn, prompts, trials, warmup):
    """Run fn(prompts) warmup times (untimed), then `trials` timed; return dict."""
    for _ in range(warmup):
        fn(prompts)
    durations = []
    total_tokens = 0
    for _ in range(trials):
        t0 = time.perf_counter()
        responses = fn(prompts)
        durations.append(time.perf_counter() - t0)
        total_tokens += sum(len(r.split()) for r in responses)  # proxy token count
    med = statistics.median(durations)
    n = len(prompts)
    return {
        "wall_s": med,
        "decisions_per_s": n / med if med > 0 else float("inf"),
        "tokens_per_s": (total_tokens / trials) / med if med > 0 else float("inf"),
    }


def engine_ours_current(model, max_tokens, temperature):
    """N sequential single-prompt generate() calls (today's behavior at N>1)."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model=model, backend="vllm", max_tokens=max_tokens, temperature=temperature,
        market_data_keys=["market_data_1Hour_48"], account_state_labels=["exposure_pct"],
        action_levels=[-1.0, 0.0, 1.0],
    )
    def run(prompts):
        return [actor.generate(SYSTEM_PROMPT, p) for p in prompts]
    return run, actor


def engine_ours_batched(model, max_tokens, temperature):
    """One native-vLLM batched generate_batch call."""
    from torchtrade.actor import LocalLLMActor
    actor = LocalLLMActor(
        model=model, backend="vllm", max_tokens=max_tokens, temperature=temperature,
        market_data_keys=["market_data_1Hour_48"], account_state_labels=["exposure_pct"],
        action_levels=[-1.0, 0.0, 1.0],
    )
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


def engine_torchrl_sync(model, max_tokens, temperature):
    """vLLMWrapper over a sync vllm.LLM (text mode, no log-probs)."""
    import vllm
    from torchrl.modules.llm import vLLMWrapper
    llm = vllm.LLM(model=model)
    wrapper = vLLMWrapper(
        llm, input_mode="text", generate=True, return_log_probs=False,
        generate_kwargs={"max_new_tokens": max_tokens, "temperature": temperature,
                         "stop": ["</answer>"]},
    )
    return _wrapped_engine_run(wrapper), llm


def engine_torchrl_async(model, max_tokens, temperature, num_replicas):
    """vLLMWrapper over an AsyncVLLM (Ray) engine. Caller guards for ray import."""
    import ray
    from torchrl.modules.llm import vLLMWrapper
    from torchrl.modules.llm.backends.vllm import AsyncVLLM
    if not ray.is_initialized():
        ray.init()
    engine = AsyncVLLM.from_pretrained(model, num_devices=1, num_replicas=num_replicas)
    wrapper = vLLMWrapper(
        engine, input_mode="text", generate=True, return_log_probs=False,
        generate_kwargs={"max_new_tokens": max_tokens, "temperature": temperature,
                         "stop": ["</answer>"]},
    )
    return _wrapped_engine_run(wrapper), engine


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-replicas", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []  # (engine, N, wall_s, decisions_per_s, tokens_per_s)

    engines = [
        ("ours-current", lambda: engine_ours_current(args.model, args.max_tokens, args.temperature)),
        ("ours-batched", lambda: engine_ours_batched(args.model, args.max_tokens, args.temperature)),
        ("torchrl-sync", lambda: engine_torchrl_sync(args.model, args.max_tokens, args.temperature)),
    ]
    try:
        import ray  # noqa: F401
        engines.append(("torchrl-async",
                        lambda: engine_torchrl_async(args.model, args.max_tokens,
                                                     args.temperature, args.num_replicas)))
    except ImportError:
        warnings.warn("ray not installed; skipping the torchrl-async (AsyncVLLM) engine. "
                      "Install with: pip install -e '.[bench]'")

    for name, builder in engines:
        run, handle = builder()
        for n in args.batch_sizes:
            prompts = make_prompts(n)
            stats = _median_stats(run, prompts, args.trials, args.warmup)
            rows.append((name, n, stats["wall_s"], stats["decisions_per_s"], stats["tokens_per_s"]))
            print(f"{name:14s} N={n:<3d} wall={stats['wall_s']*1000:8.1f}ms "
                  f"decisions/s={stats['decisions_per_s']:8.2f} "
                  f"tokens/s={stats['tokens_per_s']:8.1f}")
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

    print("\nengine,N,wall_s,decisions_per_s,tokens_per_s")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.3f},{r[4]:.3f}")

    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
