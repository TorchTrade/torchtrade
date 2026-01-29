# Rule-based actors (no optional dependencies)
from torchtrade.actor.rulebased import (
    RuleBasedActor,
    MeanReversionActor,
)

__all__ = [
    "RuleBasedActor",
    "MeanReversionActor",
]

# Optional actors with external dependencies
try:
    from torchtrade.actor.frontier_llm_actor import LLMActor
    __all__.append("LLMActor")
except ImportError:
    pass

try:
    from torchtrade.actor.local_llm_actor import LocalLLMActor
    __all__.append("LocalLLMActor")
except ImportError:
    pass

