"""SAO loss — Single-Rollout Asynchronous Optimization for LLM trading actors.

SAO (arXiv:2607.07508) is an LLM RL loss at its core, like TorchRL's ``GRPOLoss``.
It is a thin subclass of TorchRL's LLM ``GRPOLoss`` that reuses the
entire proven pipeline — ``masking_strategy`` → the wrapper's ``_get_rlhf_dist``
hook, per-token log-weights, token/prompt aggregation, ESS, entropy, optional KL
— and overrides ONLY the policy objective, exactly as TorchRL's own ``CISPOLoss``
does. Two swaps vs GRPO:

1. **Clip → DIS mask (Eq. 3).** GRPO clamps the importance ratio to the trust
   region; SAO instead **masks out-of-band tokens to zero** and keeps the ratio
   as the off-policy weight in-band (``f(x)=x`` in-band, ``0`` outside).
2. **Group baseline → critic advantage.** GRPO's advantage is group-relative
   (``MCAdvantage`` over K samples of one prompt); SAO's is ``Â = R − V(s)`` from
   a critic, computed by the trainer and written to the ``advantage`` key. This
   loss consumes ``advantage`` exactly as GRPO does (per-sequence, shape
   ``[B,1,1]``, broadcast across the answer tokens) — it does not care how the
   advantage was produced, so the "single rollout + critic" story lives in the
   trainer, not here.

The per-token DIS mask matches the paper's token-level ``f(r_t)``. There is no
entropy or KL term by default (the paper's objective has neither).
"""

from __future__ import annotations

from torchrl.objectives.llm.grpo import GRPOLoss, GRPOLossOutput


class SAOLossOutput(GRPOLossOutput):
    """SAO loss output. Same fields as GRPO; ``clip_fraction`` carries the DIS
    **masked fraction** (tokens outside the trust region)."""


class SAOLoss(GRPOLoss):
    """SAO objective for LLM actors: DIS-masked single-rollout PG with a critic advantage.

    ``_compute_policy_objective`` is the only override. In-band the gain is the
    differentiable ``ratio · advantage`` (the standard PPO-surrogate form, whose
    gradient equals ``ratio · Â · ∇log π_θ`` — the SAO/importance-weighted PG
    gradient); out-of-band tokens are zeroed, contributing no loss and no
    gradient. The returned second value (stored by the parent under the
    ``clip_fraction`` key) is the fraction of tokens masked out.

    Args:
        actor_network: the LLM policy wrapper (e.g. ``ChunkedTransformersWrapper``).
        epsilon_low: lower trust-region half-width ε_l — tokens with
            ``ratio ≤ 1 − ε_l`` are masked. Must be in [0, 1). Default 0.2.
        epsilon_high: upper trust-region half-width ε_h — tokens with
            ``ratio ≥ 1 + ε_h`` are masked. Default 0.2. The paper uses aggressive
            asymmetric "clip-higher" values (e.g. ε_l=0.3, ε_h=5.0); widen
            ``epsilon_high`` to opt in.
        entropy_bonus: default ``False`` (the paper's objective has no entropy
            term), overriding GRPO's ``True``.
        masking_strategy: default ``"rlhf"`` (score assistant/answer tokens only),
            overriding GRPO's ``"sft"`` — this is the answer-token training the
            trainer already uses.

    All other keyword arguments (``aggregation``, ``kl_to_ref_coeff``,
    ``reduction``, ``device`` …) pass straight through to ``GRPOLoss``.
    """

    output_type = SAOLossOutput

    def __init__(
        self,
        actor_network=None,
        *,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        entropy_bonus: bool = False,
        masking_strategy: str = "rlhf",
        **kwargs,
    ):
        # Reuse GRPO's asymmetric clip machinery: it registers clip_epsilon_low /
        # clip_epsilon_high buffers and exposes _clip_bounds = (log(1-ε_l),
        # log(1+ε_h)), which is exactly the log-space trust region the DIS mask
        # tests against. (GRPO validates ε_l < 1 so 1-ε_l > 0 — fine for SAO.)
        super().__init__(
            actor_network,
            clip_epsilon=(epsilon_low, epsilon_high),
            entropy_bonus=entropy_bonus,
            masking_strategy=masking_strategy,
            **kwargs,
        )

    def _compute_policy_objective(self, log_weight, advantage):
        """SAO Eq. 1+3: in-band gain = ratio·Â (identity f), out-of-band = 0.

        Returns ``(-gain, masked_fraction)``. ``log_weight`` is the per-token
        log importance ratio ``[B, seq, 1]``; ``advantage`` broadcasts from
        ``[B, 1, 1]`` across the token dim.
        """
        low, high = self._clip_bounds  # (log(1 - ε_l), log(1 + ε_h))
        in_band = (log_weight > low) & (log_weight < high)  # per-token, Eq. 3 open interval
        gain = log_weight.exp() * advantage * in_band.to(log_weight.dtype)
        masked_fraction = (~in_band).to(log_weight.dtype).mean()
        return -gain, masked_fraction
