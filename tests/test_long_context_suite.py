from benchmarks.run_long_context_suite import (
    DEFAULT_DEPTHS,
    DEFAULT_LENGTHS,
    HELDOUT_CASES,
    _safe_remove_target_bucket,
)


def test_default_suite_has_dense_length_and_depth_coverage():
    assert DEFAULT_LENGTHS == tuple(range(64_000, 128_001, 8_000))
    assert DEFAULT_DEPTHS == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert len(DEFAULT_LENGTHS) * len(DEFAULT_DEPTHS) * HELDOUT_CASES == 270


def test_target_cleanup_removes_only_one_length_bucket(tmp_path):
    targets = tmp_path / "targets"
    bucket = targets / "64000"
    bucket.mkdir(parents=True)
    artifact = bucket / "feature.pt"
    artifact.write_bytes(b"test")

    _safe_remove_target_bucket(str(targets), str(bucket))

    assert targets.is_dir()
    assert not bucket.exists()


def test_target_cleanup_refuses_targets_root(tmp_path):
    targets = tmp_path / "targets"
    targets.mkdir()

    try:
        _safe_remove_target_bucket(str(targets), str(targets))
    except RuntimeError as error:
        assert "complete targets directory" in str(error)
    else:
        raise AssertionError("cleanup must refuse the targets root")
