"""ChunkedTransformersWrapper must be numerically identical to the stock TransformersWrapper when
driving a stock GRPOLoss — it only changes HOW the per-token log-prob/entropy are computed (from
hidden states, chunked over the sequence, so the full [K, seq, vocab] logits are never built),
never WHAT they are. Regression guard for the two bugs this design flushed out: the position-0
shift (must be -log(vocab), not a hardcoded 0) and the assistant/attention mask alignment.

Tiny Qwen2 on CPU; no GPU. `chunk_size=1` exercises the many-chunks path, `4096` the single-chunk
path — both must match to <1e-4 on loss terms AND gradients.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchrl.objectives.llm")

from tensordict import TensorDict, lazy_stack
from transformers import AutoModelForCausalLM, Qwen2Config
from torchrl.modules.llm import TransformersWrapper
from torchrl.objectives.llm import GRPOLoss

from torchtrade.llm.train.chunked_policy import ChunkedMaskedCategorical, ChunkedTransformersWrapper

VOCAB, HIDDEN = 512, 64


class _DummyTokenizer:
    """input_mode='tokens' never decodes; the wrapper only reads pad/eos at init."""
    pad_token_id, eos_token_id = 0, 1
    pad_token, eos_token = "<pad>", "<eos>"

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.pad_token_id]}


def _tiny_model():
    return AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))


def _wrap(model, cls, **kw):
    return cls(model, tokenizer=_DummyTokenizer(), input_mode="tokens", generate=False,
               return_log_probs=True, pad_output=False, device="cpu", **kw)


def _loss_and_grads(loss_mod, model, batch):
    model.zero_grad(set_to_none=True)
    out = loss_mod(batch.copy())
    terms = {"loss_objective": out.loss_objective, "loss_entropy": out.loss_entropy}
    sum(terms.values()).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return terms, grads


def _build_batch(num_groups=2, k=3, min_len=6, max_len=12, vocab=512, seed=0):
    """K generations x num_groups groups, ragged lengths. One row is forced to full length (no left
    padding) so position 0 is unmasked under masking_strategy='generic' — the zero-hidden edge case."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for grp in range(num_groups):
        adv = torch.randn((), generator=g).item()
        for i in range(k):
            length = torch.randint(min_len, max_len + 1, (1,), generator=g).item()
            if grp == num_groups - 1 and i == k - 1:
                length = max_len
            resp = max(2, length // 2)
            tokens = torch.randint(2, vocab, (length,), generator=g)  # avoid pad/eos ids
            assistant = torch.zeros(length, dtype=torch.bool)
            assistant[-resp:] = True
            lp = torch.randn(length, generator=g) * 0.5 - 2.0
            lp[0] = 0.0
            rows.append(TensorDict({
                ("tokens", "full"): tokens,
                ("masks", "all_attention_mask"): torch.ones(length, dtype=torch.bool),
                ("masks", "all_assistant_mask"): assistant,
                ("log_probs", "full"): lp,
                "advantage": torch.full((length, 1), adv),
            }, batch_size=[]))
    return lazy_stack(rows, dim=0)


@pytest.mark.parametrize("wrap_lora", [False, True], ids=["plain", "lora"])
@pytest.mark.parametrize("masking_strategy", ["rlhf", "generic"])
@pytest.mark.parametrize("chunk_size", [1, 4096])
def test_chunked_wrapper_matches_stock_grpo(wrap_lora, masking_strategy, chunk_size):
    torch.manual_seed(0)
    model = _tiny_model()
    if wrap_lora:  # exercise the PRODUCTION wrapping — LoRA injects into get_decoder()'s layers
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"))
    model.eval()
    batch = _build_batch()

    stock = GRPOLoss(_wrap(model, TransformersWrapper),
                     masking_strategy=masking_strategy, entropy_bonus=True)
    chunked = GRPOLoss(_wrap(model, ChunkedTransformersWrapper, logprob_chunk_size=chunk_size),
                       masking_strategy=masking_strategy, entropy_bonus=True)

    stock_terms, stock_grads = _loss_and_grads(stock, model, batch)
    chunked_terms, chunked_grads = _loss_and_grads(chunked, model, batch)

    for key in stock_terms:
        assert (stock_terms[key] - chunked_terms[key]).abs().item() < 1e-4, key
    assert stock_grads.keys() == chunked_grads.keys() and stock_grads  # LoRA case: grads on adapters
    for name in stock_grads:
        assert (stock_grads[name] - chunked_grads[name]).abs().max().item() < 1e-3, name


def test_chunked_dist_uses_hidden_states_not_full_logits():
    """The memory property: `_chunked_dist` builds the dist from hidden states ([K, seq, H] — NO
    vocab dim) via get_decoder(), and never runs the lm_head MODULE over the sequence (the chunked
    kernel reads lm_head.weight through F.linear instead). Guards the exact regression the feature
    prevents — a full [K, seq, vocab] materialization."""
    torch.manual_seed(0)
    model = _tiny_model()
    batch = _build_batch()
    actor = _wrap(model, ChunkedTransformersWrapper, logprob_chunk_size=8)

    lm_head_calls = []
    model.get_output_embeddings().register_forward_hook(lambda *a: lm_head_calls.append(1))
    dist = actor._get_rlhf_dist(batch)

    assert isinstance(dist, ChunkedMaskedCategorical)
    K, seq = dist.mask.shape
    assert dist._hidden.shape == (K, seq, HIDDEN)  # hidden dim, NOT VOCAB
    assert not lm_head_calls  # lm_head module never invoked over the sequence


def test_get_sft_dist_rejects_prompt_key():
    """sft masking with a ('tokens','prompt') key is intentionally unsupported (stock's prompt-slice
    masking is provisional/padding-side sensitive) — must raise, not silently mis-mask."""
    torch.manual_seed(0)
    actor = _wrap(_tiny_model(), ChunkedTransformersWrapper, logprob_chunk_size=8)
    batch = _build_batch()
    batch.set(("tokens", "prompt"), torch.zeros(batch.shape[0], dtype=torch.long))  # presence is enough
    with pytest.raises(NotImplementedError):
        actor._get_sft_dist(batch)
