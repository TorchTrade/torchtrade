"""Tests for SAOLoss — the LLM-actor SAO loss (subclass of TorchRL GRPOLoss).

The only behavior that differs from GRPO is `_compute_policy_objective` (the DIS
mask vs GRPO's clamp), so that is what these tests pin. The full LLM pipeline
(masking, aggregation, ESS) is inherited from GRPOLoss and tested by TorchRL; a
model-in-the-loop test belongs in the trainer smoke test, not here.
"""

import pytest
import torch

from torchtrade.losses.sao_loss import SAOLoss


def _loss(epsilon_low=0.2, epsilon_high=0.2):
    # actor_network=None is enough to exercise _compute_policy_objective directly.
    return SAOLoss(actor_network=None, epsilon_low=epsilon_low, epsilon_high=epsilon_high)


def test_in_band_gain_equals_ratio_times_advantage():
    """In-band, f(x)=x: gain = ratio·Â elementwise (identity), NOT a 0/1 gate. A vector of
    distinct in-band ratios pins the exact value, the ratio-scaling, AND the per-sequence→
    per-token advantage broadcast in one shot — a gate-only impl (gain=Â) fails on any ratio≠1."""
    loss = _loss()
    ratios = torch.tensor([0.9, 1.0, 1.1])  # all in-band for eps 0.2 (band 0.8-1.2)
    log_weight = ratios.log().reshape(1, 3, 1)
    advantage = torch.full((1, 1, 1), 2.0)  # broadcasts [1,1,1] -> [1,3,1] across the token dim
    neg_gain, masked_fraction = loss._compute_policy_objective(log_weight, advantage)
    assert torch.allclose(neg_gain, -(ratios.reshape(1, 3, 1) * 2.0), atol=1e-6)
    assert masked_fraction.item() == pytest.approx(0.0)


def test_out_of_band_tokens_masked_to_zero():
    """Out-of-band -> gain exactly 0 (masked, not clamped to a nonzero boundary)."""
    loss = _loss()
    log_weight = torch.full((1, 4, 1), 5.0)  # ratio = e^5, far above 1+eps
    advantage = torch.full((1, 1, 1), 3.0)
    neg_gain, masked_fraction = loss._compute_policy_objective(log_weight, advantage)
    assert (neg_gain == 0.0).all()
    assert masked_fraction.item() == pytest.approx(1.0)


def test_extreme_out_of_band_stays_finite():
    """An extreme out-of-band log-weight (exp overflows fp32) must be masked to a finite
    zero, NOT become inf*0=nan that poisons the whole batch loss. Regression for the
    DIS-mask overflow (PR #261 review): exp() must be clamped before the mask multiply."""
    loss = _loss()
    log_weight = torch.tensor([[[0.0], [100.0]]])  # in-band, then exp(100)=inf without the clamp
    advantage = torch.ones(1, 1, 1)
    neg_gain, masked_fraction = loss._compute_policy_objective(log_weight, advantage)
    assert torch.isfinite(neg_gain).all(), "masked out-of-band token produced a non-finite gain"
    assert neg_gain[0, 1, 0].item() == 0.0          # the extreme token contributes exactly 0
    assert masked_fraction.item() == pytest.approx(0.5)


def test_asymmetric_band_keeps_high_ratio():
    """Clip-higher: with ε_high large, a high ratio stays in-band; ε_low small
    masks a modestly-low ratio. Catches a low/high operand swap."""
    loss = _loss(epsilon_low=0.1, epsilon_high=5.0)  # band (0.9, 6.0)
    import math
    log_weight = torch.tensor([[[math.log(4.0)], [math.log(0.5)]]])  # 4.0 kept, 0.5 masked
    advantage = torch.ones(1, 1, 1)
    neg_gain, masked_fraction = loss._compute_policy_objective(log_weight, advantage)
    assert neg_gain[0, 0, 0].item() != 0.0          # 4.0 in-band (< 6.0)
    assert neg_gain[0, 1, 0].item() == 0.0          # 0.5 masked (< 0.9)
    assert masked_fraction.item() == pytest.approx(0.5)


def test_gradient_flows_through_in_band_ratio_only():
    """Gradient reaches log_weight for in-band tokens (via the differentiable
    ratio) and is exactly zero for masked tokens."""
    loss = _loss()
    log_weight = torch.tensor([[[0.0], [5.0]]], requires_grad=True)  # in-band, out-of-band
    advantage = torch.ones(1, 1, 1)
    neg_gain, _ = loss._compute_policy_objective(log_weight, advantage)
    neg_gain.sum().backward()
    assert log_weight.grad[0, 0, 0].item() != 0.0   # in-band token trained
    assert log_weight.grad[0, 1, 0].item() == 0.0   # masked token: no gradient


def test_defaults_are_paper_faithful():
    """The class defaults match the paper (arXiv:2607.07508 §4.1, math/TIR): no
    entropy, rlhf masking, and the asymmetric clip-higher band ε_l=0.3, ε_h=5.0
    (NOT a tight symmetric band — the wide upper bound is the point of DIS)."""
    loss = SAOLoss(actor_network=None)  # NO epsilon args -> exercise the class defaults
    assert loss.entropy_bonus is False
    assert loss.masking_strategy == "rlhf"
    assert float(loss.clip_epsilon_low) == pytest.approx(0.3)
    assert float(loss.clip_epsilon_high) == pytest.approx(5.0)
