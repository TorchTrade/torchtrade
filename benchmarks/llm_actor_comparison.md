# LLM-Actor Inference: TorchTrade vs TorchRL-Native vLLM

Comparison of TorchTrade's `LocalLLMActor` against TorchRL's native vLLM
machinery (`vLLMWrapper`, `AsyncVLLM`) for single-decision trading inference.

Reproduce with:

```bash
pip install -e ".[llm,bench]"   # vllm + ray; needs a CUDA GPU
python benchmarks/bench_llm_actor.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-sizes 1 4 16 32 --trials 5
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

## Results (fill in from a GPU run)

| engine | N | wall (ms) | decisions/s | tokens/s |
|--------|---|-----------|-------------|----------|
| _to be measured_ | | | | |

## Semantics & fairness caveats (important)

- **`ours-current` at N>1 is N sequential calls** — the real-world contrast, not a rigged one.
- **`torchrl-sync` and `ours-batched` both do one batched `model.generate(list)`** — expect them to be close; any gap is wrapper overhead.
- **`torchrl-async` (AsyncVLLM) does NOT do a single batched call** — it issues one Ray remote generate per prompt and gathers them. Its win is concurrency + multi-replica scaling, plus weight-sync hooks for RL training, not vLLM continuous batching per se.
- **`vLLMWrapper` re-decodes tokens** via `tokenizer.batch_decode` rather than using vLLM's `.outputs[0].text`; this is extra CPU work the raw `ours-*` path doesn't pay. Reported numbers include it (surfaced, not hidden).

## What do we actually need from TorchRL natively? (recommendation)

_Filled in after measurement. Decision axes:_
- If `torchrl-sync` ≈ `ours-batched`: our own batched path already captures the
  inference throughput win with **no Ray dependency** → keep our actor, defer native adoption.
- The **`AsyncVLLM`/Ray** path's value is future **multi-replica scaling and weight-sync
  during RL training** (backlog B1 / RL-trainable LLM), not single-symbol sequential
  inference. Adopt it only when that training use case is concrete.
