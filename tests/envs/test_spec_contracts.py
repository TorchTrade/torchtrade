"""Contract tests for spec declarations. Two unrelated contracts live here.

**Specs must be samplable.** TorchRL samples a Bounded spec as
`uniform() * (high - low) + low`, so an infinite bound makes `.rand()` produce NaN (both
bounds infinite) or inf (one bound infinite), and inf poisons a lazily-built network just
as thoroughly. `check_env_specs()` catches neither: it builds its dummy batch from
`spec.zero()` and a real rollout, never `.rand()`. Use `Unbounded(shape=..., dtype=...)`,
or finite numbers where a bound is real.

**The done spec is declared exactly once** (#272). TorchRL's default done spec carries
only `done` and `terminated`, so anything pre-allocating from the spec drops `truncated`
with no error at all. `TorchTradeBaseEnv` declares all three for live and offline alike;
the tests at the end of this file guard that one declaration against both drift and
silent loss.

Since #313 the LIVE `_step` methods no longer write `truncated` by hand, so for them the
declaration is its only source. One consequence is worth knowing: their `check_env_specs`
can no longer catch a *narrowed* done spec, because it compares a real rollout against a
fake one and both would lack the key. `assert_the_step_emits_the_whole_done_family`, run
against all ten live envs, is what catches it there. The offline envs still write
`truncated` themselves -- they genuinely truncate -- so their `check_env_specs` does still
catch it, and so does `test_a_collector_batch_carries_truncated` below. Dropping
`truncated` from the shared spec fails 21 tests: 17 across those three guards -- 10 live
done-family cells, 6 offline `check_env_specs` cases and the collector test -- plus 4
offline `test_action_types_recorded` cells that witness it incidentally.
"""

import ast
import inspect
import math
import pathlib
import types

import pytest
import torch
from tensordict import TensorDictBase

import torchtrade
import torchtrade.envs  # noqa: F401 -- registers every live env as a subclass
from tests.envs.test_live_env_base import _subclasses
from torchtrade.envs.core.base import TorchTradeBaseEnv
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

# examples/ is scanned too: that is where this bug was found, and example code is where
# the spec.rand() lazy-init idiom tends to reappear.
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
        if isinstance(node.value, str):
            return node.value.lstrip("-").lower().startswith("inf")
        return isinstance(node.value, float) and math.isinf(node.value)
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
    `.py` files under torchtrade/ and examples/ are scanned. Those are left alone
    deliberately: each is caught by the behavioural tests below (verified), and an
    earlier attempt to resolve name indirection here introduced both false positives
    and false negatives. This guards against re-introducing the pattern; it is not an
    adversarial sandbox.
    """
    offenders = list(_infinitely_bounded_calls())
    assert not offenders, (
        "Bounded with an infinite bound cannot be sampled; use Unbounded(shape=..., "
        "dtype=...) or a finite bound:\n  " + "\n  ".join(offenders)
    )


def _assert_specs_sample_finite(env, label):
    """A spec must produce a sample it would itself accept.

    isfinite is the load-bearing assertion. is_in is a strict subset of it -- it rejects
    the NaN a two-sided-infinite Bounded produces, but accepts the inf a one-sided one
    does -- kept as a second net on the NaN case, not as independent coverage.
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


# One per family, though 3 of the 5 currently share TorchTradeOfflineEnv's builder, so a
# defect there fails 3 cases. Kept per the all-environments rule: a family that later
# overrides spec construction is then already covered.
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
    """Satisfies every live observer interface the spec builder reads.

    All five venues take the feature width from get_features() and read get_keys(); the
    spec path never calls get_observations(), so this stub does not define it (#288)."""

    def get_keys(self):
        return ["1Minute_10", "5Minute_10"]

    def get_features(self):
        return {"observation_features": ["a", "b", "c", "d", "e"]}



def _concrete_live_envs():
    """Discovered, not hand-listed, so exchange #6 cannot skip this by being forgotten.

    Abstract bases are dropped -- EnvBase.__new__ rejects them -- and so is anything
    defined outside the package: a test harness that subclasses a live base to drive one
    method would otherwise be discovered as an eleventh exchange and asked for specs it
    was never built to have.
    """
    return sorted(
        (c for c in set(_subclasses(TorchTradeLiveEnv))
         if not getattr(c, "__abstractmethods__", None)
         and c.__module__.startswith("torchtrade.")),
        key=lambda c: c.__name__,
    )


LIVE_ENVS = _concrete_live_envs()


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=[c.__name__ for c in LIVE_ENVS])
def test_live_env_specs_sample_finite(env_cls):
    """Built via __new__ + the unbound spec builder rather than a full env: every
    exchange needs different broker mocks to construct, and the specs are all this
    test is about.

    The 10 cases resolve to 5 builders (no SLTP variant overrides one), so a defect
    fails 2 cases. Deduplicating would need a hand-maintained MRO map, which is exactly
    what the __subclasses__ discovery exists to avoid.
    """
    env = env_cls.__new__(env_cls)
    env.observer = _StubObserver()
    # include_base_features=True so the shared builder declares base_features: that spec
    # has a documented history of drifting per-exchange (#61) and is otherwise unexercised.
    env.config = types.SimpleNamespace(window_sizes=[10, 10], include_base_features=True)
    env_cls._build_observation_specs(env)

    sample = env.observation_spec.rand()
    for key in sample.keys():
        value = sample[key]
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"{env_cls.__name__}.{key} sampled non-finite"
    assert env.observation_spec.is_in(sample)


def test_live_env_discovery_covers_every_exchange():
    """The parametrize above is only unskippable if discovery actually finds everything:
    an exchange dropping out would silently shrink it and stay green."""
    exchanges = {c.__module__.split(".")[3] for c in LIVE_ENVS}
    assert exchanges == {"alpaca", "binance", "bitget", "bybit", "okx"}, exchanges
    assert len(LIVE_ENVS) == 10, [c.__name__ for c in LIVE_ENVS]


def test_transform_observation_spec_samples_finite():
    """The transform is not an env, so the live parametrize cannot reach it -- without
    this its spec is guarded only by the source scan."""
    from torchrl.data import Composite
    from torchtrade.envs.transforms.binance_ohlcv import BinanceOHLCVTransform
    from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

    transform = BinanceOHLCVTransform.__new__(BinanceOHLCVTransform)
    transform.observer = types.SimpleNamespace(
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
        window_sizes=[60, 30],
    )
    transform._key_prefix = "ohlcv"
    transform._n_features = 5

    spec = transform.transform_observation_spec(Composite(shape=()))
    sample = spec.rand()
    for key in sample.keys():
        assert torch.isfinite(sample[key]).all(), f"transform.{key} sampled non-finite"


def test_polymarket_env_specs_sample_finite():
    """PolymarketBetEnv subclasses EnvBase directly, not TorchTradeLiveEnv, so it is
    invisible to the __subclasses__ discovery above."""
    from unittest.mock import MagicMock

    from torchtrade.envs.live.polymarket import PolymarketBetEnv, PolymarketBetEnvConfig

    env = PolymarketBetEnv(
        PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", dry_run=True),
        scanner=MagicMock(),
        trader=MagicMock(),
    )
    _assert_specs_sample_finite(env, "PolymarketBetEnv")


# ============================================================================
# DONE SPEC (#272)
# ============================================================================


@pytest.mark.parametrize(
    "env_cls",
    # Exactly the envs that inherit the shared declaration. The vectorized ones subclass
    # EnvBase directly, so they never reach it and their batched spec is an override.
    LIVE_ENVS + [
        cls for cls, _cfg, _kw in OFFLINE_ENVS if issubclass(cls, TorchTradeBaseEnv)
    ],
    ids=lambda c: c.__name__,
)
def test_no_env_declares_its_own_done_spec(env_cls):
    """A second, per-env copy of the done spec is free to drift from the shared one.

    TorchTradeBaseEnv.__init__ declares it once, for live and offline alike. This PR
    deleted the one surviving duplicate, in core/offline_base.py, and nothing else stops
    it coming back -- in that class or any other on the way down.

    AST, not source text: assigning the NARROWER done_spec reproduces #272 exactly, since
    it drops truncated, while never containing the string "full_done_spec".

    What this does NOT catch: it matches attribute writes, so reaching the spec any other
    way -- `del self.full_done_spec["truncated"]`, `.set(...)`, an `output_spec[...]`
    write, `setattr(self, "full_done_spec", ...)`, or a helper outside the MRO -- passes.

    What happens to those next turns on the payload, not the form. Any of them that
    removes or narrows truncated is still caught by the ten
    assert_the_step_emits_the_whole_done_family cells on the live side, by the offline
    check_env_specs tests, and by test_a_collector_batch_carries_truncated. Any of them
    that installs an IDENTICAL copy is caught by neither -- and that is the same blind
    spot this test exists to close for plain assignment: a copy nothing can see until the
    day it drifts, which is the failure mode that left this repo with three diverging
    SLTP action maps.
    """
    # The MRO, not just the leaf: the duplicate this PR removed lived in a BASE class,
    # which inspect.getsource(leaf) never shows. Stop at the one legitimate owner, which
    # also keeps torchrl's own source out.
    assigned = set()
    for klass in env_cls.__mro__:
        if klass is TorchTradeBaseEnv:
            break
        assigned |= {
            node.attr for node in ast.walk(ast.parse(inspect.getsource(klass)))
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
        }
    clashes = assigned & {"done_spec", "full_done_spec"}
    assert not clashes, (
        f"{env_cls.__name__} assigns {sorted(clashes)}; it inherits a done spec from "
        "TorchTradeBaseEnv, and a second copy is free to drift"
    )


def test_a_collector_batch_carries_truncated(sample_ohlcv_df):
    """The consequence #272 actually had, demonstrated rather than inferred.

    A collector pre-allocates its buffer from fake_tensordict(), which is built from the
    declared spec -- so an undeclared key is simply absent from every batch it yields, and
    nothing raises. check_env_specs catches the same defect, but only because a test calls
    it; a training run gets no signal at all, which is why this went unnoticed.

    The env here is offline, which never had the bug -- its spec already declared
    truncated. The mechanism is env-agnostic and a collector is cheap on this side, where
    the live envs would need broker mocks and a patched clock. So this pins the shared
    declaration going forward; the ten assert_the_step_emits_the_whole_done_family cells
    are what cover the live regression itself, since #313 left the live check_env_specs
    tests unable to see a narrowed done spec.
    """
    from torchrl.collectors import Collector

    env = SequentialTradingEnv(
        sample_ohlcv_df,
        SequentialTradingEnvConfig(
            time_frames=["5Min", "15Min"], window_sizes=[10, 10],
            execute_on="5Min", initial_cash=1000,
        ),
    )
    collector = Collector(env, frames_per_batch=3, total_frames=3)
    try:
        batch = next(iter(collector))
    finally:
        collector.shutdown()

    assert "truncated" in batch["next"].keys(), (
        "the collector yielded a batch without truncated, and did not raise -- "
        "a key absent from the spec is absent from every batch, with no error"
    )
