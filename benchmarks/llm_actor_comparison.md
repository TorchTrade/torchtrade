# LLM-Actor Inference: TorchTrade vs TorchRL-Native vLLM

Comparison of TorchTrade's `LocalLLMActor` against TorchRL's native vLLM
machinery (`vLLMWrapper`, `AsyncVLLM`) for single-decision trading inference.

Reproduce with:

```bash
pip install -e ".[llm,bench]"   # vllm + ray; needs a CUDA GPU

# One engine per process keeps VRAM clean on small GPUs; VLLM_PLUGINS="" avoids
# torchrl's (incompatible) vLLM plugin poisoning engine init on vllm >= 0.13.
VLLM_PLUGINS="" python benchmarks/bench_llm_actor.py --engines ours-current \
    --batch-sizes 1 4 16 32 --trials 5 --gpu-memory-utilization 0.7
VLLM_PLUGINS="" python benchmarks/bench_llm_actor.py --engines ours-batched \
    --batch-sizes 1 4 16 32 --trials 5 --gpu-memory-utilization 0.7
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

| engine | N | wall (ms) | decisions/s | speedup vs current |
|--------|---|-----------|-------------|--------------------|
| ours-current | 1 | 13.0 | 77.0 | 1.0× |
| ours-current | 4 | 51.1 | 78.3 | 1.0× |
| ours-current | 16 | 204.6 | 78.2 | 1.0× |
| ours-current | 32 | 417.2 | 76.7 | 1.0× |
| **ours-batched** | 1 | 13.0 | 77.0 | **1.0×** |
| **ours-batched** | 4 | 20.2 | 197.7 | **2.5×** |
| **ours-batched** | 16 | 28.4 | 562.4 | **7.2×** |
| **ours-batched** | 32 | 46.3 | 691.5 | **9.0×** |
| torchrl-sync | — | — | — | **could not run** (see below) |
| torchrl-async | — | — | — | **could not run** (see below) |

**Headline:** batched generation scales throughput ~linearly with N up to ~9× at N=32
(417 ms → 46 ms), while the sequential path is flat at ~77 decisions/s. At N=1 the two
are identical — batching adds nothing and costs nothing for single-symbol/sequential use,
so the improvement is a pure win with no regression.

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

## What do we actually need from TorchRL natively? (recommendation)

Backed by the measurements above:

1. **Keep our own batched actor — it is the win, today, with no extra dependency.**
   `ours-batched` delivers up to ~9× throughput via native vLLM continuous batching and needs
   no Ray and no torchrl LLM stack. This is the change to ship now.
2. **Do NOT adopt torchrl's native `vLLMWrapper`/`AsyncVLLM` on the current stack — it does not
   even import.** torchrl 0.10.1's vLLM integration is pinned (by a hard `vllm.worker.worker`
   import) to an older vLLM than the one that pairs with our `torch 2.9.0`. Using the native path
   requires first upgrading to **torchrl 0.11+** (backlog item **E1** — which we already deferred
   because 0.11 changed `torchrl.collectors.SyncDataCollector`). So E1 is a hard prerequisite for
   any native-vLLM adoption, and this benchmark could not measure whether native would out-perform
   our own batched path until that upgrade lands.
3. **The `AsyncVLLM`/Ray path's value remains future multi-replica scaling + weight-sync during RL
   training** (backlog B1 / RL-trainable LLM), not single-symbol sequential inference — and it is
   gated behind E1 as well. Revisit only when that training use case is concrete.

**Bottom line:** the batched-actor improvement stands on its own (measured ~9× at N=32). Native
TorchRL vLLM integration is not adoptable without the torchrl 0.11 upgrade (E1); re-run this
benchmark's `torchrl-sync`/`torchrl-async` legs after E1 to decide whether native beats our own
batched path.
