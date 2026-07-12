"""Memory-bounded per-token log-prob: compute log P(target) from hidden states + the lm_head
WITHOUT ever materializing the full [B, T, vocab] logits tensor.

This is what makes num_generations (K) a free config knob for large-vocab models (e.g. Qwen3.5's
248k vocab): standard GRPO builds [K, seq, vocab] logits — O(K·seq·vocab) — which OOMs as K grows.
Here we chunk over the sequence dimension, so peak logit memory is O(chunk·vocab), independent of
K and seq. (The Cut-Cross-Entropy / Liger idea.)
"""
from __future__ import annotations

import torch


def chunked_token_log_probs(hidden_states, lm_head_weight, target_ids, chunk_size=1024,
                            lm_head_bias=None):
    """log P(target_ids[b,t] | hidden_states[b,t]) for every position, chunked over B*T.

    Args:
        hidden_states: [B, T, H] final-layer hidden states (already shifted so position t predicts
            target_ids[b, t]).
        lm_head_weight: [V, H] the lm_head weight (V = vocab size).
        target_ids: [B, T] token ids whose log-prob we want.
        chunk_size: number of (flattened) positions per chunk; caps peak vocab memory at chunk*V.
        lm_head_bias: optional [V].

    Returns:
        log_probs: [B, T] float32.
    """
    B, T, H = hidden_states.shape
    flat_h = hidden_states.reshape(B * T, H)
    flat_tgt = target_ids.reshape(B * T)
    out = torch.empty(B * T, dtype=torch.float32, device=hidden_states.device)
    for i in range(0, B * T, chunk_size):
        h = flat_h[i:i + chunk_size]                      # [c, H]
        logits = torch.nn.functional.linear(h, lm_head_weight, lm_head_bias).float()  # [c, V]
        logsumexp = torch.logsumexp(logits, dim=-1)       # [c]
        chosen = logits.gather(-1, flat_tgt[i:i + chunk_size, None]).squeeze(-1)  # [c]
        out[i:i + chunk_size] = chosen - logsumexp
    return out.reshape(B, T)


def chunked_token_entropy(hidden_states, lm_head_weight, chunk_size=1024, lm_head_bias=None):
    """Per-position categorical entropy from hidden states, chunked over B*T (no full logits).

    Sibling of `chunked_token_log_probs` — same memory bound (O(chunk*vocab)), used for the GRPO
    entropy bonus.
    """
    B, T, H = hidden_states.shape
    flat_h = hidden_states.reshape(B * T, H)
    out = torch.empty(B * T, dtype=torch.float32, device=hidden_states.device)
    for i in range(0, B * T, chunk_size):
        logits = torch.nn.functional.linear(flat_h[i:i + chunk_size], lm_head_weight,
                                            lm_head_bias).float()  # [c, V]
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        out[i:i + chunk_size] = -(logp.exp() * logp).sum(-1)
    return out.reshape(B, T)
