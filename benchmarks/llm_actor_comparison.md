# LLM-Actor Inference: TorchTrade vs TorchRL-Native vLLM

Comparison of TorchTrade's `LocalLLMActor` against TorchRL's native vLLM
machinery (`vLLMWrapper`, `AsyncVLLM`) for single-decision trading inference.

Reproduce with:

```bash
pip install -e ".[llm,bench]"   # vllm + ray; needs a CUDA GPU

# One engine per process keeps VRAM clean on small GPUs; VLLM_PLUGINS="" avoids
# torchrl's (incompatible) vLLM plugin poisoning engine init on vllm >= 0.13.
VLLM_PLUGINS="" python benchmarks/bench_llm_actor.py --engines ours-current \
    --models Qwen/Qwen2.5-0.5B-Instruct --batch-sizes 1 4 16 32 --trials 20 --gpu-memory-utilization 0.7
VLLM_PLUGINS="" python benchmarks/bench_llm_actor.py --engines ours-batched \
    --models Qwen/Qwen2.5-0.5B-Instruct --batch-sizes 1 4 16 32 --trials 20 --gpu-memory-utilization 0.7
```

## Method

- Same model, same sampling params (`temperature`, `max_tokens`, `stop=["</answer>"]`),
  same prompts, same `<answer>N</answer>` parsing. **Only the generation engine varies.**
- Warmup runs excluded from timing (vLLM prefix-cache / CUDA-graph); median of >=5 trials.
- Metrics: wall-time per batch, decisions/sec, tokens/sec. Batch sweep N ∈ {1, 4, 16, 32}.
- `tokens/sec` is a **proxy** (whitespace word count of the responses), not an exact tokenizer count — useful for relative comparison across engines, not as an absolute figure.

## Engines

| Engine | What it does |
|--------|--------------|
| `ours-current` | N sequential single-prompt `generate()` calls (today's behavior at N>1). |
| `ours-batched` | One native `vllm.LLM.generate([...N])` call — vLLM continuous batching, no Ray. |
| `torchrl-sync` | `vLLMWrapper(vllm.LLM, input_mode="text")` — one `model.generate(list)` call internally. |
| `torchrl-async` | `vLLMWrapper(AsyncVLLM, ...)` — Ray; **loops per prompt** issuing concurrent remote calls. |

## Results

Measured 2026-07-09 on **NVIDIA RTX 4060 Laptop GPU (8 GB)**, model `Qwen/Qwen2.5-0.5B-Instruct`,
`--trials 5 --gpu-memory-utilization 0.7`, one engine per process. Stack:
torch 2.9.0+cu128, vllm 0.13.0, ray 2.56.0, torchrl 0.10.1, tensordict 0.10.0.

`--trials 20`. `tokens/s` is **real generated tokens** (counted with the model's HF tokenizer),
`p50/p95/p99` are batch-completion latency (time until all N decisions are ready).

| engine | N | wall (ms) | decisions/s | tokens/s | p50/p95/p99 (ms) | speedup |
|--------|---|-----------|-------------|----------|------------------|---------|
| ours-current | 1 | 13.1 | 76.2 | 76.0 | 13.1 / 13.8 / 13.9 | 1.0× |
| ours-current | 4 | 52.3 | 76.4 | 76.4 | 52.3 / 54.0 / 54.6 | 1.0× |
| ours-current | 16 | 212.4 | 75.3 | 75.0 | 212 / 235 / 242 | 1.0× |
| ours-current | 32 | 421.1 | 76.0 | 75.7 | 421 / 446 / 447 | 1.0× |
| **ours-batched** | 1 | 14.2 | 70.7 | 70.1 | 14.2 / 15.1 / 15.4 | 0.9× |
| **ours-batched** | 4 | 21.1 | 189.5 | 188.4 | 21.1 / 23.0 / 23.4 | **2.5×** |
| **ours-batched** | 16 | 30.2 | 529.2 | 528.4 | 30.2 / 42.5 / 47.6 | **7.0×** |
| **ours-batched** | 32 | 47.6 | 672.3 | 669.3 | 47.6 / 52.3 / 54.7 | **8.8×** |
| torchrl-sync | — | — | — | — | — | **could not run** (see below) |
| torchrl-async | — | — | — | — | — | **could not run** (see below) |

**Headline:** batched generation scales throughput ~linearly with N up to ~9× at N=32
(421 ms → 48 ms), while the sequential path is flat at ~76 decisions/s. At N=1 the two are
within noise (batching adds a tiny fixed overhead, no benefit with nothing to batch) — so the
improvement is a pure win for parallel-env / multi-symbol use with no regression for single-obs.

**On tokens/s:** here `tokens/s ≈ decisions/s` because Qwen-0.5B is too small to actually reason —
with `stop=["</answer>"]` it emits ~1 token per decision. tokens/s only becomes a distinct,
meaningful throughput metric with a model that produces real `<think>` reasoning traces; see the
model-size sweep (pending, to be run on larger hardware) for 4B/7B numbers where it diverges.

**Prefill/decode split:** reported `n/a` on this stack — vLLM 0.13's v1 engine does not populate
per-request `RequestMetrics` timestamps. Forcing the legacy engine (`VLLM_USE_V1=0`) is the likely
way to recover the split on the larger-hardware run.

### The torchrl-native legs could not run (a finding in itself)

`torchrl-sync` and `torchrl-async` both failed to construct:

```
ModuleNotFoundError: No module named 'vllm.worker'
```

`torchrl 0.10.1`'s vLLM backend (`torchrl/modules/llm/backends/vllm/vllm_async.py`) hard-imports
`from vllm.worker.worker import Worker` at import time, but **vllm 0.13.0 removed that module path**.
vllm 0.13.0 is the newest vllm that pairs with our pinned `torch 2.9.0` (older vllm needs older
torch, which would break `torchrl 0.10.1`/`tensordict 0.10.0`). So torchrl's native `vLLMWrapper`/
`AsyncVLLM` are **unusable with the vllm version that fits our current stack** — worse, torchrl
registers a vLLM plugin that vLLM 0.13 auto-loads on every engine init, so the mismatch poisons
`vllm.LLM` construction entirely unless launched with `VLLM_PLUGINS=""` (which the ours-* runs use).

## Semantics & fairness caveats (important)

- **`ours-current` at N>1 is N sequential calls** — the real-world contrast, not a rigged one.
- **`torchrl-sync` and `ours-batched` both do one batched `model.generate(list)`** — expect them to be close; any gap is wrapper overhead.
- **`torchrl-async` (AsyncVLLM) does NOT do a single batched call** — it issues one Ray remote generate per prompt and gathers them. Its win is concurrency + multi-replica scaling, plus weight-sync hooks for RL training, not vLLM continuous batching per se.
- **`vLLMWrapper` re-decodes tokens** via `tokenizer.batch_decode` rather than using vLLM's `.outputs[0].text`; this is extra CPU work the raw `ours-*` path doesn't pay. Reported numbers include it (surfaced, not hidden).

## DGX Spark (GB10) — model-size sweep

Measured 2026-07-09 on an **NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, sm_121,
unified 128 GB)**. `pip install vllm` does not work on sm_121, so this ran inside the
prebuilt `avarok/vllm-dgx-spark:v11` container (vLLM 0.14, torch 2.9.1+cu130, FlashInfer
sm_121). Because `import torchtrade` eagerly pulls the full env/broker stack (torchrl,
alpaca, ccxt, …) which won't install cleanly in that container, the Spark run uses
`benchmarks/bench_vllm_standalone.py` — a torchtrade-free script that measures the exact
`vllm.LLM.generate(...)` path `LocalLLMActor` wraps (same chat template, same
`stop=["</answer>"]`, same metrics). `tokens/s` is real generated tokens; `--trials 20`
(10 for 7B). The GB10 allocates a **99.5 GiB KV cache** (≈4200× concurrency), so batching
never saturates.

**Qwen2.5-0.5B** (trivial ~1-token output → tokens/s ≈ decisions/s):

| engine | N | decisions/s | tokens/s | speedup |
|--------|---|-------------|----------|---------|
| ours-current | any | ~63–65 | ~63–65 | 1.0× |
| ours-batched | 32 | 1008 | 975 | ~16× |
| ours-batched | 64 | 1373 | 1304 | ~21× |
| ours-batched | 128 | **1768** | 1737 | **~27×** |

**Qwen2.5-3B** (real reasoning → tokens/s diverges from decisions/s):

| engine | N | decisions/s | tokens/s | p50/p95 (ms) |
|--------|---|-------------|----------|--------------|
| ours-current | 16–64 | ~14.7 (flat) | ~17–19 | high variance |
| ours-batched | 4 | 41 | 41 | 97 / 100 |
| ours-batched | 32 | 259 | 83 | 124 / 4266 |
| ours-batched | 64 | 330 | 133 | 194 / 4236 |

**Qwen2.5-7B** (tokens/s is the meaningful metric; batch wall ≈ constant ~8–9 s, bounded by
the longest 128-token reasoning trace):

| engine | N | decisions/s | tokens/s | wall (ms) | p50/p99 (ms) |
|--------|---|-------------|----------|-----------|--------------|
| ours-current | 1 | 1.69 | 9.7 | 594 | 594 / 16948 |
| ours-batched | 1 | 1.67 | 6.7 | 597 | 597 / 600 |
| ours-batched | 32 | 3.9 | 52 | 8263 | 8263 / 9828 |
| ours-batched | 64 | 7.6 | 98 | 8377 | 8377 / 8476 |
| ours-batched | 128 | 14.5 | **200** | 8827 | 8827 / 8862 |

### What the Spark run shows

1. **Batching decouples throughput from latency.** With continuous batching, N decisions
   complete in ~the time of the slowest single generation, so throughput scales ~linearly
   with N while wall-time stays roughly flat. At 7B, N=128 yields **200 tokens/s vs 6.7 at
   N=1 (~30×)** and 128 decisions in ~8.8 s (≈ one long generation), while sequential is flat
   at ~1.7 decisions/s.
2. **tokens/s is the robust throughput metric; decisions/s is noisy at variable generation
   length.** For 3B/7B the median-trial wall depends on whether that trial hit a full-length
   reasoning trace, so decisions/s bounces around (7B reads 1.6 at N=16 but 14.5 at N=128);
   tokens/s trends cleanly. This confirms measuring real tokens/s (not the earlier word
   proxy) was the right call.
3. **Latency tail grows sharply with model size.** 7B p50 ≈ 600 ms but p99 up to ~17 s when a
   decision emits a full 128-token trace — a real consideration for a live bar deadline;
   production should cap `max_tokens` tightly.
4. **The GB10 trades single-stream clock for parallelism.** At small batch / small model it is
   slightly *slower* per decision than the RTX 4060 laptop (~63 vs ~76 dec/s at 0.5B N=1); its
   advantage is large-batch, large-model throughput (the 99 GB KV cache).

### Caveats

- The main sweep was interrupted (to skip the pathologically slow 7B sequential-large-N cells,
  which add no insight beyond predictable N× scaling), so a few 0.5B/3B cells are absent; the
  trends are unaffected. Reproduce a full clean run with the commands in
  `bench_vllm_standalone.py`'s header.
- **Prefill/decode split is `n/a`** everywhere — vLLM 0.13/0.14's v1 engine does not populate
  per-request `RequestMetrics` timestamps. Retry with `VLLM_USE_V1=0` to recover it.

## TorchRL-native head-to-head (measured on DGX Spark)

The earlier "could not run" for the native legs was our repo's `torchrl==0.10.1` (hard-imports
`vllm.worker.worker.Worker`, gone in vllm>=0.13). Inside the Spark container we installed
**torchrl 0.13.2** (`pip install --no-deps torchrl tensordict pyvers orjson`), whose
`vLLMWrapper` DOES load against vllm 0.14, and benchmarked it **apples-to-apples**: same
chat template, same `SamplingParams` (temperature, max_tokens, `stop=["</answer>"]`). A greedy
side-by-side confirmed both paths generate identical content (the wrapper only appends the
`<|im_end|>` EOS token in its re-decode, which we strip before counting). Script:
`benchmarks/bench_torchrl_native.py`.

**`torchrl-sync` (`vLLMWrapper` over `vllm.LLM`) is ~25–30% slower than our direct batched
path** — clean at 0.5B (trivial 1-token output → pure speed, no length confound):

| N | ours-batched (dec/s) | torchrl-sync (dec/s) | torchrl overhead |
|---|----------------------|----------------------|------------------|
| 16 | 632 | 433 | −31% |
| 32 | 1008 | 782 | −22% |
| 64 | 1372 | 1025 | −25% |
| 128 | 1767 | 1254 | −29% |

Both wrap the **same vLLM engine**, so this gap is pure wrapper cost (TensorDict I/O + token
re-decode). At 3B/7B the two track within the (large) sampling-length variance — `tokens/s`
is *not* a reliable head-to-head there because temperature=0.7 makes the two sample
different-length reasoning; `decisions/s` at 0.5B is the unconfounded comparison. **Takeaway:
TorchRL-native is not faster for inference — it is measurably slower.**

**`torchrl-async` (`AsyncVLLM`, the RL-training / multi-replica path) does not load** on this
stack: `ImportError: cannot import name 'WeightTransferConfig' from 'vllm.config'`
(torchrl 0.13.2's async backend vs vllm 0.14). So the one capability that would justify
adopting native — async generation + in-place weight-sync for on-policy RL — could not even be
measured, and is tightly coupled to a specific torchrl↔vllm pairing.

### Capability matrix

| Capability | TorchTrade batched actor (this PR) | TorchRL-native |
|---|---|---|
| Batched inference throughput | ✅ native vLLM, fastest measured | ✅ same vLLM, **~25–30% slower** |
| Extra dependencies | **none** | torchrl 0.11+ (backlog E1) + `ray` |
| Native token log-probs (for RL) | ❌ | ✅ (`LogProbs` tensorclass) |
| `LLMCollector` / `GRPOLoss` / replay-buffer integration | ❌ | ✅ (the real draw) |
| In-place weight-sync for RL rollout loops | ❌ (full engine reload, ~20–30 s) | ⚠️ in principle — **blocked on vllm 0.14 here** |
| Multi-GPU serving replicas | ❌ | ✅ (multi-GPU; not testable on single-GPU GB10) |

See `benchmarks/plots/` for the rendered figures (batching-win bar + throughput-scaling per model).

## What do we actually need from TorchRL natively? (recommendation)

Now backed by a measured head-to-head, not conjecture:

1. **Ship our batched actor (this PR) — it is the throughput win, today, with no new dependency.**
   Native vLLM continuous batching, up to ~27× (0.5B) / ~30× tokens/s (7B) over the sequential
   path. Nothing to adopt from TorchRL to get it.
2. **Do NOT adopt TorchRL-native for inference speed — it is measurably *slower*.** `vLLMWrapper`
   wraps the same vLLM engine and adds ~25–30% overhead (TensorDict I/O + token re-decode), and it
   requires the `torchrl 0.11+` upgrade (backlog **E1**) plus `ray`. There is no throughput reason
   to take that on.
3. **TorchRL-native's only real draw is RL-training integration** — native token log-probs,
   `LLMCollector`/`GRPOLoss`/replay-buffer plumbing, and `AsyncVLLM` in-place weight-sync for
   on-policy rollout loops. That is the path to *training* an LLM trading policy (backlog **B1**),
   not serving one. But it is doubly gated: behind E1, **and** behind fragile torchrl↔vllm version
   coupling — the `AsyncVLLM` path did not even import against vllm 0.14 here
   (`WeightTransferConfig` ImportError).

**Bottom line / the statement for the PR:** this PR delivers the inference win outright, with zero
new dependencies. Adopting TorchRL-native buys us *nothing* for inference (it's slower) — its value
is exclusively for a *future* RL-trained LLM policy, and even that is not turnkey (needs E1 + a
working torchrl+vllm pin). Recommendation: **merge the batched actor now; defer TorchRL-native
until we actually build the RL-training loop (B1), and treat getting a working `AsyncVLLM`+vllm
combination as its own spike.**

---

## For the announcement (X post draft)

> ⚡ New in TorchTrade: **batched LLM inference** for the trading actor.
>
> One vLLM call decides for N symbols at once instead of N sequential calls. On an NVIDIA DGX
> Spark (GB10): **up to 27× throughput** (Qwen2.5-0.5B) and **~30× tokens/sec** on a 7B model —
> because a batch of N decisions completes in ~the time of a single one. Run more symbols live,
> or backtest an LLM policy far faster, on the same GPU. No new dependencies.
>
> We also benchmarked TorchRL's native vLLM path head-to-head (apples-to-apples, same prompts &
> sampling): it wraps the same engine and runs ~25–30% slower. Its value isn't speed — it's the
> plumbing to *train* the LLM policy with RL (log-probs, collectors, weight-sync). That's the next
> chapter.
>
> [attach `benchmarks/plots/fig_batching_win_tokens_per_s.png` + `fig_scaling_decisions_per_s.png`]

Plots (`benchmarks/plots/`): `fig_batching_win_tokens_per_s.png` (hero bar), plus
`fig_scaling_{tokens,decisions}_per_s.png` (throughput vs batch size per model). Regenerate with
`python benchmarks/plot_bench.py benchmarks/data/spark_combined.csv --out benchmarks/plots`.
