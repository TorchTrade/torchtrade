# Rule-based actors (no optional dependencies)
from torchtrade.actor.rulebased import (
    RuleBasedActor,
    MomentumActor,
    MeanReversionActor,
    BreakoutActor,
    SLTPRuleBasedActor,
    MomentumSLTPActor,
    MeanReversionSLTPActor,
    BreakoutSLTPActor,
    FuturesRuleBasedActor,
    MomentumFuturesActor,
    MeanReversionFuturesActor,
    BreakoutFuturesActor,
    create_expert_ensemble,
)

__all__ = [
    "RuleBasedActor",
    "MomentumActor",
    "MeanReversionActor",
    "BreakoutActor",
    "SLTPRuleBasedActor",
    "MomentumSLTPActor",
    "MeanReversionSLTPActor",
    "BreakoutSLTPActor",
    "FuturesRuleBasedActor",
    "MomentumFuturesActor",
    "MeanReversionFuturesActor",
    "BreakoutFuturesActor",
    "create_expert_ensemble",
]

# Optional actors with external dependencies
try:
    from torchtrade.actor.llm_actor import LLMActor
    __all__.append("LLMActor")
except ImportError:
    pass

try:
    from torchtrade.actor.human import HumanActor
    __all__.append("HumanActor")
except ImportError:
    pass
