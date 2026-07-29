"""Contract tests for observation/reward spec declarations.

`Bounded(low=-inf, high=inf)` is not a harmless spelling of "unconstrained": TorchRL
samples a Bounded spec as `uniform() * (high - low) + low`, so infinite bounds make
every `.rand()` draw NaN and the spec rejects its own sample. Nothing in the normal
test path catches it -- `check_env_specs()` builds its dummy batch from `spec.zero()`,
never `.rand()` -- so it survived across every environment until an example that used
the standard `spec.rand()` lazy-init idiom silently initialised its nets on NaN.

The structural guard is source-level on purpose. Two attempts to sweep this by hand
missed sites: once because the sweep was driven by a file list, and once because a file
spelled infinity `float("inf")` instead of `torch.inf`. A guard that parses every file
in the package cannot be escaped by either.
"""

import ast
import pathlib

import pytest
import torch

import torchtrade

PACKAGE_ROOT = pathlib.Path(torchtrade.__file__).parent


def _is_infinite(node):
    """True for `torch.inf`, `np.inf`, `float("inf")` and their negations."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_infinite(node.operand)
    if isinstance(node, ast.Attribute) and node.attr == "inf":
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and str(node.args[0].value).lower().lstrip("-") in {"inf", "infinity"}
    ):
        return True
    return isinstance(node, ast.Constant) and node.value in (float("inf"), float("-inf"))


def _infinite_bounded_calls():
    """Every `Bounded(...)` in the package whose low AND high are both infinite."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "Bounded":
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if "low" in kw and "high" in kw and _is_infinite(kw["low"]) and _is_infinite(kw["high"]):
                yield f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}"


def test_no_spec_is_bounded_by_infinity():
    """Use Unbounded instead: a doubly-infinite Bounded samples NaN from .rand().

    Half-infinite bounds are deliberately allowed -- e.g. polymarket's market_state is
    Bounded(low=0.0, high=inf), where the lower bound is a real constraint (prices and
    sizes are non-negative) and .rand() yields inf rather than NaN.
    """
    offenders = list(_infinite_bounded_calls())
    assert not offenders, (
        "Bounded(low=-inf, high=inf) samples NaN; use Unbounded(shape=..., dtype=...):\n  "
        + "\n  ".join(offenders)
    )


def test_offline_env_specs_sample_finite_values(sample_ohlcv_df):
    """The behaviour the structural guard exists to protect: a spec must be able to
    produce a sample it would itself accept. check_env_specs() cannot catch this."""
    from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

    env = SequentialTradingEnv(
        sample_ohlcv_df,
        SequentialTradingEnvConfig(
            time_frames=["5Min", "15Min"], window_sizes=[10, 10],
            execute_on="5Min", initial_cash=1000, action_levels=[0, 1],
        ),
    )

    sample = env.observation_spec.rand()
    for key in sample.keys():
        value = sample[key]
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"{key} sampled non-finite values"
    assert env.observation_spec.is_in(sample), "spec rejects its own sample"

    reward = env.reward_spec.rand()
    assert torch.isfinite(reward).all(), "reward_spec sampled non-finite values"
