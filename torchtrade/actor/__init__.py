# Rule-based actors (no optional dependencies)
from torchtrade.actor.rulebased import (
    RuleBasedActor,
    MeanReversionActor,
)

__all__ = [
    "RuleBasedActor",
    "MeanReversionActor",
]

# Optional LLM actors
try:
    from torchtrade.actor.base_llm_actor import BaseLLMActor
    __all__.append("BaseLLMActor")
except ImportError:
    pass

try:
    from torchtrade.actor.frontier_llm_actor import FrontierLLMActor, LLMActor
    __all__.extend(["FrontierLLMActor", "LLMActor"])
except ImportError:
    pass

try:
    from torchtrade.actor.local_llm_actor import LocalLLMActor
    __all__.append("LocalLLMActor")
except ImportError:
    pass
