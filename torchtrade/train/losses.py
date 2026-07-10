"""Pluggable loss resolution for the LLM GRPO trainer.

The seam covers the group-relative-PG family that consumes the OneStep-K-group
rollout (GRPO and cousins). A loss needing a different rollout (PPO/Sequential,
DPO) is a new recipe, not a loss_kwargs change.
"""
from typing import Callable, Optional, Union


def _grpo_loss_cls():
    # Imported lazily: torchrl.objectives.llm pulls heavy LLM deps.
    from torchrl.objectives.llm import GRPOLoss
    return GRPOLoss


# name -> zero-arg callable returning the loss class (lazy import)
_LOSS_REGISTRY = {"grpo": _grpo_loss_cls}


def validate_num_generations(n: int) -> None:
    if n < 2:
        raise ValueError(
            f"num_generations must be >= 2 for GRPO (got {n}); the group-relative "
            "advantage is undefined for a group of size 1."
        )


def resolve_loss(loss: Union[str, Callable], actor_network, loss_kwargs: Optional[dict] = None):
    """Resolve `loss` to a constructed LossModule for `actor_network`."""
    if callable(loss) and not isinstance(loss, str):
        return loss(actor_network)
    if isinstance(loss, str):
        if loss not in _LOSS_REGISTRY:
            raise ValueError(f"unknown loss '{loss}'; known: {sorted(_LOSS_REGISTRY)}")
        cls = _LOSS_REGISTRY[loss]()
        return cls(actor_network, **(loss_kwargs or {}))
    raise ValueError(f"loss must be a registry name or a factory callable, got {type(loss)}")
