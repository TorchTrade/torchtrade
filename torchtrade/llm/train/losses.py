"""Pluggable loss resolution for the LLM GRPO trainer.

The seam covers the group-relative-PG family that consumes the OneStep-K-group
rollout (GRPO and cousins). A loss needing a different rollout (PPO/Sequential,
DPO) is a new recipe, not a loss_kwargs change.
"""


def validate_num_generations(n: int) -> None:
    if n < 2:
        raise ValueError(
            f"num_generations must be >= 2 for GRPO (got {n}); the group-relative "
            "advantage is undefined for a group of size 1."
        )


def resolve_loss(loss, actor_network, loss_kwargs=None):
    """Resolve `loss` to a constructed LossModule for `actor_network`: the name ``"grpo"``, or a
    factory callable ``f(actor) -> LossModule`` for anything else."""
    if callable(loss) and not isinstance(loss, str):
        return loss(actor_network)
    if loss == "grpo":
        from torchrl.objectives.llm import GRPOLoss  # lazy: pulls heavy LLM deps
        return GRPOLoss(actor_network, **(loss_kwargs or {}))
    raise ValueError(f"unknown loss {loss!r}; use 'grpo' or a factory callable f(actor) -> LossModule")
