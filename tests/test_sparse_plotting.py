import torch

from sparse.plotting import selected_block_matrix


def test_selected_block_matrix_marks_only_real_block_ids():
    selected = torch.tensor([[[[[0, -1, -1], [0, 1, -1], [0, 1, 2]]]]], dtype=torch.int32)
    actual = selected_block_matrix(selected, layer=0, kv_group=0)
    expected = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)
