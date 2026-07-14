"""Standalone test for FlatGate MaxSim. Run: python scripts/test_gate_maxsim.py"""
import os, sys, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


if __name__ == "__main__":
    test_shape()
    test_matches_bruteforce_maxsim()
    print("OK test_gate_maxsim")
