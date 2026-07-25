import pytest
import torch

from scripts.eval_natural_gates import (
    aggregate_cases,
    parse_gate_specs,
    route_metrics,
    selected_blocks_to_mask,
)


def test_parse_gate_specs_supports_labels_and_rejects_duplicates():
    gates = parse_gate_specs(["old=old.pt", "new.pt"])
    assert gates[0][0] == "old"
    assert gates[1][0] == "new"
    with pytest.raises(ValueError, match="duplicate"):
        parse_gate_specs(["same=a.pt", "same=b.pt"])


def test_selected_blocks_mask_and_route_metrics():
    selected = torch.tensor(
        [[[[[0, -1], [0, 1], [0, 2]]]]], dtype=torch.int32)
    mask = selected_blocks_to_mask(selected, key_blocks=3)
    assert mask[0, 0, -1].tolist() == [True, False, True]

    target = torch.tensor([[[[1.0, 0.0, 0.0],
                              [0.3, 0.7, 0.0],
                              [0.2, 0.1, 0.7]]]])
    cmask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    metrics = route_metrics(
        mask, target, cmask, budgets=(1, 2), needle_block=2,
        selection_budget=2)
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(2.9 / 3)
    assert metrics["needle_union_ratio"] == pytest.approx(1.0)


def test_aggregate_cases_groups_requested_lengths():
    cases = [
        {"requested_length": 16_384, "metrics": {"recall@10": 0.8}},
        {"requested_length": 16_384, "metrics": {"recall@10": 1.0}},
        {"requested_length": 32_768, "metrics": {"recall@10": 0.7}},
    ]
    aggregate = aggregate_cases(cases)
    assert aggregate["cases"] == 3
    assert aggregate["overall"]["recall@10"] == pytest.approx(2.5 / 3)
    assert aggregate["by_length"]["16384"]["recall@10"] == pytest.approx(0.9)
