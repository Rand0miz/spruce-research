import pytest
import torch

from benchmarks.all_blocks_equivalence import (
    DEFAULT_ACCEPTANCE,
    all_causal_selected_blocks,
    passes_acceptance,
    select_matched_records,
    tensor_error,
)
from interfaces.validator import validate_selected_blocks


def test_all_causal_selected_blocks_fills_every_causal_row():
    selected = all_causal_selected_blocks(
        num_layers=2, num_kv_groups=2, seq_len=130, block_size=64)
    assert selected.shape == (1, 2, 2, 3, 3)
    assert selected[0, 0, 0].tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
    ]
    assert torch.equal(selected[0, 0, 0], selected[0, 1, 1])
    validate_selected_blocks(selected)


def test_select_matched_records_prefers_middle_depth_stably():
    records = []
    for case_id, depth in (("zeta", 0.1), ("alpha", 0.5)):
        for length in (16_384, 32_768):
            records.append({
                "source_case_id": case_id,
                "depth": depth,
                "seed": 7,
                "variant": 0,
                "requested_length": length,
                "candidate_id": f"{case_id}-{length}",
            })
    chosen = select_matched_records(records, [16_384, 32_768])
    assert [record["candidate_id"] for record in chosen] == [
        "alpha-16384", "alpha-32768"]


def test_tensor_error_is_zero_for_identical_tensors():
    value = torch.tensor([[1.0, -2.0]])
    metrics = tensor_error(value, value)
    assert metrics == {
        "actual_finite": True,
        "expected_finite": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rmse": 0.0,
        "reference_rms": pytest.approx(2.5 ** 0.5),
        "relative_rmse": 0.0,
    }


def test_acceptance_rejects_nonfinite_or_changed_top1():
    logits = {
        "actual_finite": True,
        "expected_finite": True,
        "top1_same": True,
        "top10_overlap": 1.0,
        "cosine": 1.0,
    }
    hidden = {
        "actual_finite": True,
        "expected_finite": True,
        "relative_rmse": 0.0,
    }
    assert passes_acceptance(logits, hidden, DEFAULT_ACCEPTANCE)
    logits["actual_finite"] = False
    assert not passes_acceptance(logits, hidden, DEFAULT_ACCEPTANCE)
    logits["actual_finite"] = True
    logits["top1_same"] = False
    assert not passes_acceptance(logits, hidden, DEFAULT_ACCEPTANCE)
