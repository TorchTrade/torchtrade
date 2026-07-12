"""Memory-bounded GRPO training via a LAZY distribution — no GRPOLoss patching.

torchrl's stock `GRPOLoss._get_cur_log_prob` doesn't build the distribution itself: it delegates
to the WRAPPER's `_get_rlhf_dist` / `_get_generic_dist` / `_get_sft_dist` (else `get_dist`). Those
hooks are documented in torchrl as "provisional ... will be replaced by get_dist once we have a
better masking strategy" — i.e. the intended extension seam. The reason the stock path OOMs on
large-vocab models (Qwen3.5 = 248320) is NOT the distribution object; it's that those hooks run the
model forward and materialize a full `[K, seq, vocab]` logits tensor BEFORE any distribution exists.

So instead of subclassing the loss, we return our OWN distribution from the wrapper's hooks:
`ChunkedMaskedCategorical` holds `(shifted_hidden_states, lm_head_weight)` instead of logits and
computes `.log_prob()` / `.entropy()` chunked over the sequence (peak vocab memory O(chunk·vocab),
independent of K and seq). `GRPOLoss` stays 100% stock, and the KL-to-reference path — which also
goes through these hooks — gets the same memory bound for free.
"""
from __future__ import annotations

import torch
from torchrl.modules.llm import TransformersWrapper

_IGNORE_INDEX = -100  # GRPOLoss pads the action tokens with this; must never reach gather()


# --- memory-bounded kernels -------------------------------------------------------------------
# Compute per-token log-prob / entropy from hidden states + the lm_head WITHOUT ever materializing
# the full [B, T, vocab] logits. Chunking over B*T caps peak logit memory at O(chunk*vocab),
# independent of K and seq — the Cut-Cross-Entropy / Liger idea, and what makes K a free knob.

def chunked_token_log_probs(hidden_states, lm_head_weight, target_ids, chunk_size=1024,
                            lm_head_bias=None):
    """log P(target_ids[b,t] | hidden_states[b,t]) for every position, chunked over B*T.

    hidden_states [B,T,H] must already be shifted so position t predicts target_ids[b,t];
    lm_head_weight is [V,H]. Returns [B,T] float32.
    """
    B, T, H = hidden_states.shape
    flat_h = hidden_states.reshape(B * T, H)
    flat_tgt = target_ids.reshape(B * T)
    out = torch.empty(B * T, dtype=torch.float32, device=hidden_states.device)
    for i in range(0, B * T, chunk_size):
        logits = torch.nn.functional.linear(flat_h[i:i + chunk_size], lm_head_weight,
                                            lm_head_bias).float()  # [c, V]
        logsumexp = torch.logsumexp(logits, dim=-1)
        chosen = logits.gather(-1, flat_tgt[i:i + chunk_size, None]).squeeze(-1)
        out[i:i + chunk_size] = chosen - logsumexp
    return out.reshape(B, T)


def chunked_token_entropy(hidden_states, lm_head_weight, chunk_size=1024, lm_head_bias=None):
    """Per-position categorical entropy from hidden states, chunked over B*T (no full logits).
    Sibling of `chunked_token_log_probs`; used for the GRPO entropy bonus."""
    B, T, H = hidden_states.shape
    flat_h = hidden_states.reshape(B * T, H)
    out = torch.empty(B * T, dtype=torch.float32, device=hidden_states.device)
    for i in range(0, B * T, chunk_size):
        logits = torch.nn.functional.linear(flat_h[i:i + chunk_size], lm_head_weight,
                                            lm_head_bias).float()  # [c, V]
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        out[i:i + chunk_size] = -(logp.exp() * logp).sum(-1)
    return out.reshape(B, T)


class ChunkedMaskedCategorical:
    """Lazy stand-in for torchrl's `LLMMaskedCategorical`, over the SHIFTED-logits convention
    (position ``t`` scores token ``t`` from ``hidden[t-1]``; position 0's predecessor is the zero
    vector → uniform → ``-log(vocab)``). Never materializes `[K, seq, vocab]` logits.

    Satisfies exactly the interface `GRPOLoss` touches: ``.mask``, ``.log_prob(action)``,
    ``.entropy()``. `.entropy()` is finite everywhere, so the loss never falls back to Monte-Carlo
    entropy (which would need `.sample`/`.rsample`).
    """

    def __init__(self, shifted_hidden, lm_head_weight, mask, chunk_size, lm_head_bias=None):
        self._hidden = shifted_hidden          # [K, seq, H], already zero-prepended + shifted
        self._w, self._b = lm_head_weight, lm_head_bias
        self.mask = mask                        # [K, seq] bool — assistant/attention positions
        self._chunk = chunk_size

    def log_prob(self, action):
        # action == ("tokens","full"), padded with _IGNORE_INDEX at masked positions. Clamp those to
        # a valid id purely so gather() is in-range; they're zeroed out by the mask immediately after
        # (matching stock's cross_entropy(ignore_index) → 0 at those positions).
        targets = torch.where(self.mask, action, torch.zeros_like(action))
        lp = chunked_token_log_probs(self._hidden, self._w, targets, self._chunk, self._b)
        return torch.where(self.mask, lp, torch.zeros_like(lp))

    def entropy(self):
        ent = chunked_token_entropy(self._hidden, self._w, self._chunk, self._b)
        return torch.where(self.mask, ent, torch.zeros_like(ent))


class ChunkedTransformersWrapper(TransformersWrapper):
    """`TransformersWrapper` whose dist hooks return `ChunkedMaskedCategorical` — so any stock LLM
    loss (GRPOLoss, and its ref-KL path) trains with K-independent peak memory. `logprob_chunk_size`
    caps peak vocab memory at ``chunk·vocab``; tune per model/GPU (user-configurable via LLMTrainer).
    """

    def __init__(self, *args, logprob_chunk_size: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.logprob_chunk_size = logprob_chunk_size

    def _chunked_dist(self, tensordict, mask_kind: str) -> ChunkedMaskedCategorical:
        input_ids = tensordict.get(("tokens", "full"), as_padded_tensor=True,
                                   padding_side="left", padding_value=self.padding_value)
        attn = tensordict.get(("masks", "all_attention_mask"), as_padded_tensor=True,
                              padding_side="left", padding_value=0).bool()
        if mask_kind == "attention":
            mask = attn
        else:  # "assistant"
            # tensordict.get returns None (not KeyError) for a missing key on a LazyStackedTensorDict,
            # so a bare .bool() would be a cryptic AttributeError — fail with a clear message instead.
            mask = tensordict.get(("masks", "all_assistant_mask"), as_padded_tensor=True,
                                  padding_side="left", padding_value=0)
            if mask is None:
                raise KeyError("rlhf/sft masking requires ('masks', 'all_assistant_mask')")
            mask = mask.bool()

        # Get ONLY the final hidden state via the base decoder — NOT output_hidden_states=True on the
        # CausalLM: that flag retains EVERY layer's activations and DEFEATS gradient checkpointing
        # (measured 27.5GB vs 7.6GB at K=2,seq=512 on Qwen3.5-4B). get_decoder() is the transformer
        # stack (LoRA adapters injected in-place, so LoRA grads still flow), runs under GC, and
        # returns last_hidden_state directly — never touching the [K,seq,vocab] lm_head at all.
        decoder = self.model.get_decoder() if hasattr(self.model, "get_decoder") else None
        if decoder is not None:
            hidden = decoder(input_ids=input_ids, attention_mask=attn,
                             use_cache=False).last_hidden_state
        else:  # fallback for models without get_decoder(): retains all hidden states (defeats GC)
            hidden = self.model(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=True, logits_to_keep=1,
                                use_cache=False).hidden_states[-1]
        lm_head = self.model.get_output_embeddings()

        # position t scores token t from hidden[t-1]; position 0 gets the zero vector (uniform).
        shifted = torch.cat([torch.zeros_like(hidden[:, :1, :]), hidden[:, :-1, :]], dim=1)
        return ChunkedMaskedCategorical(shifted, lm_head.weight, mask, self.logprob_chunk_size,
                                        getattr(lm_head, "bias", None))

    # torchrl delegates to whichever of these matches the loss's masking_strategy.
    def _get_rlhf_dist(self, tensordict, **kwargs):     # assistant tokens only (our default)
        return self._chunked_dist(tensordict, "assistant")

    def _get_generic_dist(self, tensordict, **kwargs):  # all attended tokens
        return self._chunked_dist(tensordict, "attention")

    def _get_sft_dist(self, tensordict, **kwargs):
        # Stock's sft path slices out the prompt when ('tokens','prompt') is present via
        # padding-side-sensitive logic torchrl itself flags as provisional; reproducing it risks a
        # subtle divergence. Without a prompt key it degenerates to the assistant mask, which we do
        # support. Steer training to masking_strategy='rlhf' to avoid this entirely.
        if ("tokens", "prompt") in tensordict.keys(True):
            raise NotImplementedError(
                "ChunkedTransformersWrapper: masking_strategy='sft' with ('tokens','prompt') is not "
                "supported (stock prompt-slice masking is provisional). Use masking_strategy='rlhf'."
            )
        return self._chunked_dist(tensordict, "assistant")

    def get_dist(self, tensordict, *args, **kwargs):  # generic fallback == all attended tokens
        return self._chunked_dist(tensordict, "attention")
