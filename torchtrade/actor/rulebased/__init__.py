"""
Rule-based trading actors for imitation learning and expert demonstrations.

This package provides deterministic trading strategies that can be used as experts
for behavioral cloning or as baselines for RL policy evaluation.
"""

from torchtrade.actor.rulebased.base import RuleBasedActor
from torchtrade.actor.rulebased.meanreversion.actor import MeanReversionActor

__all__ = [
    "RuleBasedActor",
    "MeanReversionActor",
]
