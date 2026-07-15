import pytest
import torch

from teacher.extract import get_attentions
from teacher.pool import pool_attention, rollup
from teacher.validate import check_causal
from teacher.chunked_extract import get_pooled_targets


MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BLOCK = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# short prompt so the EAGER reference (full matrix) can run for comparison
PROMPT = ("def add(a, b):\n    return a + b\n" * 40) + "\n# what does add do?"


def eager_reference(model, tok, prompt):
    attentions, seq = get_attentions(model, tok, prompt, device=DEVICE)  # tuple[L] of [1,H,q,k]
    layers = [pool_attention(a.float().cpu(), BLOCK) for a in attentions]  # each [1,H,qb,kb]
    return torch.stack(layers, dim=1), seq   # [1,L,H,qb,kb]


@pytest.mark.integration
def test_chunked_extract_matches_eager_reference():
    transformers = pytest.importorskip("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp32 + eager: exact reference on a short prompt (do NOT do this at 16K)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE).eval()

    ref, seq = eager_reference(model, tok, PROMPT)
    chunk, _pooledQ, _pooledK, seq2 = get_pooled_targets(
        model, tok, PROMPT, device=DEVICE, block_size=BLOCK)

    assert seq == seq2, (seq, seq2)
    assert ref.shape == chunk.shape, (ref.shape, chunk.shape)

    diff = (ref - chunk).abs()
    assert torch.allclose(ref, chunk, atol=1e-4, rtol=1e-3), (
        diff.max().item(), diff.mean().item())

    # causal + diagonal-heavy checks on aggregated leaf (sum over layers+heads)
    agg = chunk.sum(dim=(1, 2))[0]                       # [qb,kb]
    causal_ok, viol = check_causal(agg[None, None])
    diag = agg.diagonal().sum().item()
    total = agg.sum().item()
    assert causal_ok, viol
    assert diag / total > 0
    assert rollup(agg[None, None])
