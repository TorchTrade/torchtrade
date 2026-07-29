"""Contract tests for observation/reward spec declarations.

TorchRL samples a Bounded spec as `uniform() * (high - low) + low`, so an infinite
bound makes `.rand()` produce NaN (both bounds infinite) or inf (one bound infinite),
and inf poisons the first forward pass just as thoroughly. `check_env_specs()` cannot
catch either: it builds its dummy batch from `spec.zero()` and a real rollout, never
`.rand()`. So every environment shipped a spec that could not be sampled, and the
standard `spec.rand()` lazy-init idiom silently initialised networks on garbage.

Use `Unbounded(shape=..., dtype=...)` for unbounded quantities, and finite numbers
where a bound is real.
"""

import ast
import pathlib

import pytest
import torch
from tensordict import TensorDictBase

import torchtrade

# examples/ is scanned too: the bug was found there, in an example whose lazy init used
# the spec.rand() idiom. Guarding only the package would leave its discovery site open.
REPO_ROOT = pathlib.Path(torchtrade.__file__).parent.parent
SCAN_ROOTS = [REPO_ROOT / "torchtrade", REPO_ROOT / "examples"]


def _infinite(node, consts):
    """True for `torch.inf`, `np.inf`, `float("inf")`, their negations, and names bound
    to any of those at module level."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _infinite(node.operand, consts)
    if isinstance(node, ast.Attribute):
        return node.attr == "inf"
    if isinstance(node, ast.Name):
        return consts.get(node.id, False)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
        return bool(node.args) and isinstance(node.args[0], ast.Constant) and str(
            node.args[0].value
        ).lstrip("-").lower().startswith("inf")
    return False


def _bounded_names(tree):
    """`Bounded`, plus any alias it was imported under."""
    names = {"Bounded"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names if a.name == "Bounded")
    return names


def _infinitely_bounded_calls():
    """Every `Bounded(...)` under the scan roots with an infinite low or high."""
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            names = _bounded_names(tree)
            # Module-level `INF = torch.inf` style indirection.
            consts = {
                t.id: _infinite(n.value, {})
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_bounded = (
                    (isinstance(func, ast.Name) and func.id in names)
                    or (isinstance(func, ast.Attribute) and func.attr == "Bounded")
                )
                if not is_bounded:
                    continue
                kw = {k.arg: k.value for k in node.keywords}
                # Bounded(low, high, ...) may pass either positionally.
                low = kw.get("low", node.args[0] if len(node.args) > 0 else None)
                high = kw.get("high", node.args[1] if len(node.args) > 1 else None)
                if (low is not None and _infinite(low, consts)) or (
                    high is not None and _infinite(high, consts)
                ):
                    yield f"{path.relative_to(REPO_ROOT)}:{node.lineno}"


def test_no_spec_is_bounded_by_infinity():
    """An infinite bound on either side makes .rand() unsamplable (NaN, or inf that
    becomes NaN downstream). Catches aliased, qualified and positional spellings, so
    it cannot be sidestepped by writing the same thing a different way."""
    offenders = list(_infinitely_bounded_calls())
    assert not offenders, (
        "Bounded with an infinite bound cannot be sampled; use Unbounded(shape=..., "
        "dtype=...) or a finite bound:\n  " + "\n  ".join(offenders)
    )


def _offline_envs(df):
    """One instance per offline env family. Built here rather than fixtured so a new
    family shows up as an import error instead of silently going uncovered."""
    from torchtrade.envs.offline import (
        OneStepTradingEnv, SequentialTradingEnv, SequentialTradingEnvSLTP,
        VectorizedSequentialTradingEnv,
    )

    common = dict(
        time_frames=["5Min", "15Min"], window_sizes=[10, 10],
        execute_on="5Min", initial_cash=1000,
    )
    from torchtrade.envs.offline import (
        OneStepTradingEnvConfig, SequentialTradingEnvConfig,
        SequentialTradingEnvSLTPConfig, VectorizedSequentialTradingEnvConfig,
    )
    return {
        "sequential": SequentialTradingEnv(df, SequentialTradingEnvConfig(**common)),
        "sequential_sltp": SequentialTradingEnvSLTP(df, SequentialTradingEnvSLTPConfig(**common)),
        "onestep": OneStepTradingEnv(df, OneStepTradingEnvConfig(**common)),
        "vectorized": VectorizedSequentialTradingEnv(
            df, VectorizedSequentialTradingEnvConfig(**common, num_envs=2)
        ),
    }


@pytest.fixture
def offline_envs(sample_ohlcv_df):
    return _offline_envs(sample_ohlcv_df)


@pytest.mark.parametrize(
    "family", ["sequential", "sequential_sltp", "onestep", "vectorized"]
)
def test_env_specs_sample_finite_values(offline_envs, family):
    """The behaviour the structural guard protects, checked per env family: a spec must
    produce a sample it would itself accept. The guard proves no infinite Bounded is
    written down; this proves the specs are actually samplable."""
    env = offline_envs[family]

    for name in ("observation_spec", "reward_spec", "action_spec", "full_done_spec"):
        spec = getattr(env, name, None)
        if spec is None:
            continue
        sample = spec.rand()
        values = (
            sample.values(include_nested=True, leaves_only=True)
            if isinstance(sample, TensorDictBase)
            else [sample]
        )
        for value in values:
            if value.is_floating_point():
                assert torch.isfinite(value).all(), f"{family}.{name} sampled non-finite"
        assert spec.is_in(sample), f"{family}.{name} rejects its own sample"
