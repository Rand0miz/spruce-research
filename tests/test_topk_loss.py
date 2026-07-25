"""Tests for top-k membership loss."""
import torch

from selector.loss import needle_topk_loss, topk_membership_loss, topk_set_loss


def test_topk_loss_prefers_teacher_topk():
    # One row with five causal keys. Teacher top-2 are keys 1 and 3.
    target = torch.tensor([[[[0.05, 0.45, 0.10, 0.35, 0.05]]]])
    cmask = torch.ones(1, 5, dtype=torch.bool)

    good = torch.tensor([[[[0.0, 5.0, 1.0, 4.0, -1.0]]]])
    bad = torch.tensor([[[[5.0, 0.0, 4.0, -1.0, 3.0]]]])

    good_loss, good_pos = topk_membership_loss(good, target, cmask, k=2)
    bad_loss, bad_pos = topk_membership_loss(bad, target, cmask, k=2)
    assert good_pos == bad_pos == 2
    assert good_loss.item() < bad_loss.item(), (good_loss.item(), bad_loss.item())


def test_legacy_wrapper_still_works():
    target = torch.tensor([[[[0.2, 0.8]]]])
    cmask = torch.ones(1, 2, dtype=torch.bool)
    scores = torch.tensor([[[[0.0, 1.0]]]])
    loss, positives = topk_set_loss(scores, target, cmask, topk=1)
    assert positives == 1
    assert loss.isfinite()


def test_needle_loss_prefers_needle_above_topk_threshold():
    target = torch.tensor([[[[0.05, 0.10, 0.80, 0.05]]]])
    cmask = torch.ones(1, 4, dtype=torch.bool)
    bad = torch.tensor([[[[3.0, 2.0, -1.0, 1.0]]]])
    good = torch.tensor([[[[1.0, 0.0, 4.0, -1.0]]]])

    bad_loss, bad_groups = needle_topk_loss(
        bad, target, cmask, needle_block=2, k=1)
    good_loss, good_groups = needle_topk_loss(
        good, target, cmask, needle_block=2, k=1)

    assert bad_groups == good_groups == 1
    assert good_loss.item() < bad_loss.item()
