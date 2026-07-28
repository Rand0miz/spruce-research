import pytest
import torch

from benchmarks.diagnose_residual_summaries import (
    combine_error_sums,
    error_sums,
    finish_error,
    sampled_positions,
    select_smallest_near_best,
)


def test_sampled_positions_are_unique_and_include_endpoints():
    positions = sampled_positions(10, 4)
    assert positions.tolist() == sorted(set(positions.tolist()))
    assert positions[0].item() == 0
    assert positions[-1].item() == 9


def test_error_sums_combine_to_globally_weighted_rmse():
    first = error_sums(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0]))
    second = error_sums(torch.tensor([3.0]), torch.tensor([1.0]))
    result = finish_error(combine_error_sums([first, second]))
    assert result["element_count"] == 3
    assert result["rmse"] == pytest.approx((8 / 3) ** 0.5)
    assert result["reference_rms"] == pytest.approx((1 / 3) ** 0.5)


def test_selects_smallest_p_within_five_percent_of_best():
    selected, threshold = select_smallest_near_best(
        {1: 1.06, 2: 1.04, 4: 1.0}
    )
    assert selected == 2
    assert threshold == pytest.approx(1.05)

