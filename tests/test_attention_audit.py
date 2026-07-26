import torch

from scripts.audit_dense_attention import block_mass, block_rank, find_question_span


def test_block_mass_sums_tokens_into_blocks():
    probs = torch.zeros(2, 8)          # H=2, T=8, block_size=4 -> kb=2
    probs[0, 0] = 0.25
    probs[0, 5] = 0.75
    probs[1, 3] = 1.0
    out = block_mass(probs, block_size=4, kb=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out[0], torch.tensor([0.25, 0.75]))
    assert torch.allclose(out[1], torch.tensor([1.0, 0.0]))


def test_block_mass_ragged_final_block():
    probs = torch.ones(1, 6) / 6.0     # T=6, block_size=4 -> kb=2 (last block ragged)
    out = block_mass(probs, block_size=4, kb=2)
    assert torch.allclose(out.sum(), torch.tensor(1.0))
    assert torch.allclose(out[0, 0], torch.tensor(4 / 6))


def test_block_rank():
    scores = torch.tensor([[0.1, 0.5, 0.4], [0.9, 0.05, 0.05]])
    assert block_rank(scores, 1).tolist() == [0, 1]
    assert block_rank(scores, 2).tolist() == [1, 1]


def test_find_question_span_last_occurrence():
    prompt = [5, 1, 2, 3, 9, 9, 1, 2, 3, 7]
    span = find_question_span(prompt, [1, 2, 3])
    assert span == (6, 9)


def test_find_question_span_missing():
    assert find_question_span([1, 2, 3], [4, 5]) is None
