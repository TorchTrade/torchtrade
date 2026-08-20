"""An offline config field nothing reads is worse than a missing one (#289)."""

import ast
import dataclasses
import pathlib

import pytest

from torchtrade.envs.core.common_types import MarginMode

from torchtrade.envs.offline import (
    OneStepTradingEnvConfig,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnvConfig,
    VectorizedSequentialTradingEnvSLTPConfig,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
# core/live.py is a LIVE module that happens to sit under core/. Sweeping it in was how
# the first version of this guard cleared `include_base_features`: live.py genuinely
# reads it, so the offline field looked consumed.
OFFLINE_SOURCES = [
    *(REPO / "torchtrade" / "envs" / "offline").rglob("*.py"),
    REPO / "torchtrade" / "envs" / "core" / "base.py",
    REPO / "torchtrade" / "envs" / "core" / "offline_base.py",
    REPO / "torchtrade" / "envs" / "core" / "state.py",
]

CONFIGS = [
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTPConfig,
    OneStepTradingEnvConfig,
    VectorizedSequentialTradingEnvConfig,
    VectorizedSequentialTradingEnvSLTPConfig,
]

# Descriptive, not behavioural: `symbol` labels the data the user supplied, and the
# offline envs correctly do not branch on it -- the dataframe is the dataframe.
DESCRIPTIVE_FIELDS = {"symbol"}

# Unread AND kept, because they are public fields an outside caller may pass -- but
# __post_init__ REJECTS any value other than the implemented one, so they cannot be set
# and ignored. `test_a_flag_the_offline_envs_ignore_is_rejected` proves that, which is
# what earns the exemption; being listed here is not enough on its own.
REJECTED_FIELDS = {"include_base_features", "margin_mode"}


def _names_read_offline() -> set:
    """Attribute names actually READ, by AST, excluding validation raises.

    Three things the regex version got wrong:

    - A `getattr(config, "seed", None)` read was invisible, so a field consumed only
      that way looked dead.
    - `self.<field>` inside `__post_init__` counted as a read -- so ADDING a validation
      raise that names a field permanently disarms the guard for it. That is exactly
      backwards: the raise exists because the field is dead.
    - A match anywhere in the concatenated sources cleared the field on every config, so
      `margin_mode` -- copied once in sequential.py and read nowhere -- was exempt on all
      five.
    """
    class _StripPostInit(ast.NodeTransformer):
        """Remove __post_init__ bodies entirely.

        `ast.walk` yields a function's children independently of the function node, so
        `continue`-ing on the FunctionDef skips nothing -- the first version of this did
        exactly that and the disarm mutation sailed through. The subtree has to go.
        """

        def visit_FunctionDef(self, node):
            return None if node.name == "__post_init__" else self.generic_visit(node)

    read = set()
    for path in OFFLINE_SOURCES:
        tree = _StripPostInit().visit(ast.parse(path.read_text()))
        for child in ast.walk(tree):
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                read.add(child.attr)
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == "getattr" and len(child.args) >= 2
                    and isinstance(child.args[1], ast.Constant)):
                read.add(child.args[1].value)
    return read


@pytest.mark.parametrize("config_cls", CONFIGS, ids=lambda c: c.__name__)
def test_every_offline_config_field_is_read_somewhere(config_cls):
    """`include_base_features` and `margin_mode` both sat here with ZERO readers.

    Setting either did nothing, silently: `include_base_features=True` promised an
    observation that was never emitted, and `margin_mode=CROSSED` promised cross-margin
    liquidation while `_liquidation_price` calls `isolated_liquidation_price`
    unconditionally. Both now raise; this keeps the next one from being added quietly.
    """
    read = _names_read_offline()
    unread = sorted(
        f.name for f in dataclasses.fields(config_cls)
        if f.name not in DESCRIPTIVE_FIELDS | REJECTED_FIELDS
        and f.name not in read
    )
    assert not unread, (
        f"{config_cls.__name__} declares fields nothing in the offline envs reads: "
        f"{unread} -- a config field that does nothing is a silent no-op for whoever "
        f"sets it. Reject it in __post_init__ or delete it."
    )


@pytest.mark.parametrize("config_cls", [
    SequentialTradingEnvConfig, SequentialTradingEnvSLTPConfig, OneStepTradingEnvConfig,
], ids=lambda c: c.__name__)
@pytest.mark.parametrize("kwargs,match", [
    ({"include_base_features": True}, "not implemented for the offline"),
    ({"margin_mode": MarginMode.CROSSED}, "not implemented for the offline"),
])
def test_a_flag_the_offline_envs_ignore_is_rejected(config_cls, kwargs, match):
    """Every config that inherits the validation, not just Sequential.

    Both defaulted to the value the envs actually implement, so the wrong setting was
    accepted and ignored -- a policy configured for cross margin trained against
    isolated liquidation prices with nothing saying so.
    """
    with pytest.raises(ValueError, match=match):
        config_cls(**kwargs)
