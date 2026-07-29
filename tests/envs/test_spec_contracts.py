"""Contract tests for observation/reward spec declarations.

TorchRL samples a Bounded spec as `uniform() * (high - low) + low`, so an infinite
bound makes `.rand()` produce NaN (both bounds infinite) or inf (one bound infinite),
and inf poisons a lazily-built network just as thoroughly. `check_env_specs()` catches
neither: it builds its dummy batch from `spec.zero()` and a real rollout, never
`.rand()`. Use `Unbounded(shape=..., dtype=...)`, or finite numbers where a bound is
real.
"""

import ast
import math
import pathlib
import types

import numpy as np
import pytest
import torch
from tensordict import TensorDictBase

import torchtrade
import torchtrade.envs  # noqa: F401 -- registers every live env as a subclass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.offline import (
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
    VectorizedSequentialTradingEnvSLTP,
    VectorizedSequentialTradingEnvSLTPConfig,
)

# examples/ too: the .rand() lazy-init idiom lives there, and that is where this bug
# was found.
REPO_ROOT = pathlib.Path(torchtrade.__file__).parent.parent
SCAN_ROOTS = [REPO_ROOT / "torchtrade", REPO_ROOT / "examples"]


def _infinite(node):
    """True for `torch.inf`, `np.inf`, `float("inf")`, an overflowing literal like
    `1e999`, and negations of any of those."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _infinite(node.operand)
    if isinstance(node, ast.Attribute):
        return node.attr == "inf"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
        return bool(node.args) and _infinite(node.args[0])
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and math.isinf(node.value) or (
            isinstance(node.value, str) and node.value.lstrip("-").lower().startswith("inf")
        )
    return False


def _bounded_aliases(tree):
    """`Bounded` plus any name it was imported under."""
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
            aliases = _bounded_aliases(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    (isinstance(func, ast.Name) and func.id in aliases)
                    or (isinstance(func, ast.Attribute) and func.attr == "Bounded")
                ):
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                low = kw.get("low", node.args[0] if node.args else None)
                high = kw.get("high", node.args[1] if len(node.args) > 1 else None)
                if (low is not None and _infinite(low)) or (high is not None and _infinite(high)):
                    yield f"{path.relative_to(REPO_ROOT)}:{node.lineno}"


def test_no_spec_is_bounded_by_infinity():
    """An infinite bound on either side makes .rand() unsamplable.

    Catches the spellings a person plausibly writes: keyword or positional, aliased
    import, qualified `module.Bounded`, `torch.inf`/`np.inf`/`float("inf")`/`1e999`.
    It does NOT resolve indirection -- `partial(Bounded, ...)`, a Bounded subclass,
    a name assigned from Bounded, or `**kwargs` unpacking all escape it, and only
    `.py` files under torchtrade/ and examples/ are scanned. It is a guard against
    re-introducing the pattern, not an adversarial sandbox.
    """
    offenders = list(_infinitely_bounded_calls())
    assert not offenders, (
        "Bounded with an infinite bound cannot be sampled; use Unbounded(shape=..., "
        "dtype=...) or a finite bound:\n  " + "\n  ".join(offenders)
    )


def _assert_specs_sample_finite(env, label):
    """A spec must produce a sample it would itself accept.

    isfinite is the load-bearing assertion; is_in is near-tautological for Unbounded
    and Categorical (it checks shape/dtype only) but would catch a shape or dtype
    declaration that disagrees with what the spec generates.
    """
    for name in ("observation_spec", "reward_spec", "action_spec", "full_done_spec"):
        spec = getattr(env, name)
        sample = spec.rand()
        values = (
            list(sample.values(include_nested=True, leaves_only=True))
            if isinstance(sample, TensorDictBase)
            else [sample]
        )
        for value in values:
            if value.is_floating_point():
                assert torch.isfinite(value).all(), f"{label}.{name} sampled non-finite"
        assert spec.is_in(sample), f"{label}.{name} rejects its own sample"


OFFLINE_ENVS = [
    (SequentialTradingEnv, SequentialTradingEnvConfig, {}),
    (SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig, {}),
    (OneStepTradingEnv, OneStepTradingEnvConfig, {}),
    (VectorizedSequentialTradingEnv, VectorizedSequentialTradingEnvConfig, {"num_envs": 2}),
    (VectorizedSequentialTradingEnvSLTP, VectorizedSequentialTradingEnvSLTPConfig, {"num_envs": 2}),
]


@pytest.mark.parametrize(
    "env_cls,config_cls,extra", OFFLINE_ENVS, ids=[e[0].__name__ for e in OFFLINE_ENVS]
)
def test_offline_env_specs_sample_finite(sample_ohlcv_df, env_cls, config_cls, extra):
    env = env_cls(
        sample_ohlcv_df,
        config_cls(
            time_frames=["5Min", "15Min"], window_sizes=[10, 10],
            execute_on="5Min", initial_cash=1000, **extra,
        ),
    )
    _assert_specs_sample_finite(env, env_cls.__name__)


class _StubObserver:
    """Satisfies every live observer interface the spec builders read -- alpaca derives
    the feature width from get_observations(), the futures bases from get_features()."""

    def get_keys(self):
        return ["1Minute_10", "5Minute_10"]

    def get_features(self):
        return {"observation_features": ["a", "b", "c", "d", "e"]}

    def get_observations(self):
        return {k: np.zeros((10, 5)) for k in self.get_keys()}


def _concrete_live_envs():
    """Discovered, not hand-listed, so exchange #6 cannot skip this by being forgotten."""
    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    return sorted(
        (c for c in set(walk(TorchTradeLiveEnv)) if not getattr(c, "__abstractmethods__", None)),
        key=lambda c: c.__name__,
    )


LIVE_ENVS = _concrete_live_envs()


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=[c.__name__ for c in LIVE_ENVS])
def test_live_env_specs_sample_finite(env_cls):
    """Built via __new__ + the unbound spec builder rather than a full env: every
    exchange needs different broker mocks to construct, and the specs are all this
    test is about."""
    env = env_cls.__new__(env_cls)
    object.__setattr__(env, "observer", _StubObserver())
    object.__setattr__(
        env, "config", types.SimpleNamespace(window_sizes=[10, 10], include_base_features=False)
    )
    env_cls._build_observation_specs(env)

    sample = env.observation_spec.rand()
    for key in sample.keys():
        value = sample[key]
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"{env_cls.__name__}.{key} sampled non-finite"
    assert env.observation_spec.is_in(sample)
