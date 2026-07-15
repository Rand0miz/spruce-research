"""Tests for FlatGate MaxSim."""
import math

import torch

from selector.gate import FlatGate


L, G, qb, kb, P, d = 2, 2, 3, 3, 4, 5


def test_shape():
    gate = FlatGate(L, d, proj_dim=d).eval()
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, kb, P, d)
    with torch.no_grad():
        s = gate(q, k)
    assert s.shape == (L, G, qb, kb), s.shape


def test_matches_bruteforce_maxsim():
    gate = FlatGate(L, d, proj_dim=d).eval()
    q = torch.randn(L, G, qb, P, d)
    k = torch.randn(L, G, kb, P, d)
    with torch.no_grad():
        s = gate(q, k)
        # brute force: project, all P*P pairs, max, scale
        qp = torch.einsum("lgqid,ldp->lgqip", q, gate.Wq)
        kp = torch.einsum("lgkjd,ldp->lgkjp", k, gate.Wk)
        pair = torch.einsum("lgqip,lgkjp->lgqkij", qp, kp)   # [L,G,qb,kb,P,P]
        ref = pair.amax(dim=(-1, -2)) / math.sqrt(gate.proj_dim)
    assert torch.allclose(s, ref, atol=1e-5), (s - ref).abs().max().item()


def test_gradient_flow():
    # Both tests above run under torch.no_grad(); this pins that the trained module
    # is actually differentiable end to end (the training path in selector/train.py).
    gate = FlatGate(L, d, proj_dim=d)
    q = torch.randn(L, G, qb, P, d, requires_grad=True)
    k = torch.randn(L, G, kb, P, d, requires_grad=True)
    s = gate(q, k)
    loss = s.pow(2).sum()
    loss.backward()
    assert gate.Wq.grad is not None and gate.Wk.grad is not None
    assert torch.count_nonzero(gate.Wq.grad) > 0, "Wq.grad is all zero"
    assert torch.count_nonzero(gate.Wk.grad) > 0, "Wk.grad is all zero"
