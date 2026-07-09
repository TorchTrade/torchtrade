"""Smoke test: the benchmark script imports and exposes a CLI without a model/GPU."""

import subprocess
import sys


def test_bench_help_runs_without_model():
    """`bench_llm_actor.py --help` must work with no vllm/ray/GPU (imports deferred)."""
    result = subprocess.run(
        [sys.executable, "benchmarks/bench_llm_actor.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--batch-sizes" in result.stdout
    assert "--model" in result.stdout
