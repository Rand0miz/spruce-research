import pytest
import torch

import teacher.chunked_extract as chunked


def _fill_captures():
    chunked._CAPTURE.clear()
    chunked._CAPTURE_QK.clear()
    for layer in range(2):
        chunked._CAPTURE[layer] = torch.full(
            (1, 2, 3, 3), float(layer), dtype=torch.float32)
        chunked._CAPTURE_QK[layer] = (
            torch.full((1, 1, 3, 2, 4), float(layer), dtype=torch.float16),
            torch.full((1, 1, 3, 2, 4), float(layer), dtype=torch.float16),
        )


def test_stack_releases_layerwise_capture_copies():
    _fill_captures()
    pooled, pooled_q, pooled_k = chunked._stack_and_clear_captures([0, 1])

    assert pooled.shape == (1, 2, 2, 3, 3)
    assert pooled_q.shape == (1, 2, 1, 3, 2, 4)
    assert pooled_k.shape == (1, 2, 1, 3, 2, 4)
    assert chunked._CAPTURE == {}
    assert chunked._CAPTURE_QK == {}


def test_stack_failure_still_releases_capture_buffers():
    _fill_captures()
    with pytest.raises(KeyError):
        chunked._stack_and_clear_captures([0, 1, 2])
    assert chunked._CAPTURE == {}
    assert chunked._CAPTURE_QK == {}
