"""An offline config field nothing reads is worse than a missing one (#289)."""

import dataclasses
import pathlib
import re

import pytest

from torchtrade.envs.offline import (
    OneStepTradingEnvConfig,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnvConfig,
    VectorizedSequentialTradingEnvSLTPConfig,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
OFFLINE = REPO / "torchtrade" / "envs" / "offline"
CORE = REPO / "torchtrade" / "envs" / "core"

CONFIGS = [
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTPConfig,
    OneStepTradingEnvConfig,
    VectorizedSequentialTradingEnvConfig,
    VectorizedSequentialTradingEnvSLTPConfig,
]


def _offline_source() -> str:
    return "\n".join(
        p.read_text() for p in [*OFFLINE.rglob("*.py"), *CORE.rglob("*.py")]
    )


# Descriptive, not behavioural: `symbol` labels the data the user supplied and the
# offline envs correctly do not branch on it -- the dataframe is the dataframe. Listed
# explicitly so the distinction is a decision rather than an oversight.
DESCRIPTIVE_FIELDS = {
    "symbol",
    # Read by no offline env, but hydra configs in examples/ pass it, so deleting it
    # turns working configs into a TypeError. __post_init__ rejects True instead, which
    # is what test_setting_it_raises below pins.
    "include_base_features",
}


@pytest.mark.parametrize("config_cls", CONFIGS, ids=lambda c: c.__name__)
def test_every_offline_config_field_is_read_somewhere(config_cls):
    """`include_base_features` sat on SequentialTradingEnvConfig with ZERO readers.

    #289 lists it as a parity gap -- the vectorized envs lack it -- but a field nothing
    consumes is not a feature to propagate. Setting it did nothing, silently, which is
    worse than the vectorized envs not offering it: a user who set it got no error and no
    behaviour. Removing it turns a silent no-op into a loud TypeError.

    (The LIVE envs have a real `include_base_features` that `core/live.py` reads; this
    covers the offline configs only.)
    """
    source = _offline_source()
    unread = [
        f.name for f in dataclasses.fields(config_cls)
        if f.name not in DESCRIPTIVE_FIELDS
        and not re.search(rf"(?:config|cfg|self)\.{re.escape(f.name)}\b", source)
        and not re.search(rf"\b{re.escape(f.name)}\s*=", source.replace(
            f"{f.name}: ", "@@"))
    ]
    assert not unread, (
        f"{config_cls.__name__} declares fields nothing in offline/ or core/ reads: "
        f"{unread} -- a config field that does nothing is a silent no-op for whoever "
        f"sets it"
    )


def test_setting_include_base_features_raises_instead_of_being_ignored():
    """It defaulted to False and nothing read it, so `True` silently did nothing.

    A policy configured to receive base_features would have trained without them and
    nothing would have said so -- the silent-no-op failure that boundary validation
    exists to prevent. The field stays (hydra configs pass it); the lie does not.
    """
    with pytest.raises(ValueError, match="not implemented for the offline"):
        SequentialTradingEnvConfig(include_base_features=True)

    # False remains valid -- that is what every shipped config sets.
    assert SequentialTradingEnvConfig(include_base_features=False) is not None
