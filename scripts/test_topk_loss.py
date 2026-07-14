"""Standalone test for top-k ranking loss. Run: python scripts/test_topk_loss.py"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from selector.loss import topk_set_loss


def test_topk_loss_prefers_teacher_topk():
    # One row with five causal keys. Teacher top-2 are keys 1 and 3.
    target = torch.tensor([[[[0.05, 0.45, 0.10, 0.35, 0.05]]]])
    cmask = torch.ones(1, 5, dtype=torch.bool)

    good = torch.tensor([[[[0.0, 5.0, 1.0, 4.0, -1.0]]]])
    bad = torch.tensor([[[[5.0, 0.0, 4.0, -1.0, 3.0]]]])

    good_loss, good_rows = topk_set_loss(good, target, cmask, topk=2)
    bad_loss, bad_rows = topk_set_loss(bad, target, cmask, topk=2)
    assert good_rows == bad_rows == 1
    assert good_loss.item() < bad_loss.item(), (good_loss.item(), bad_loss.item())


if __name__ == "__main__":
    test_topk_loss_prefers_teacher_topk()
    print("OK test_topk_loss")
