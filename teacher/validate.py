import torch

def check_causal(pooled_layer, tolerance=1e-6):
    """
    pooled_layer: [batch, heads, q_blocks, k_blocks] for ONE level.
    A block is causal if query-block i never attends to key-block j > i.
    Returns True/False, and the (i, j) of the first violation if any.
    """
    b, h, q_blocks, k_blocks = pooled_layer.shape
    for i in range(q_blocks):
        for j in range(i + 1, k_blocks):
            leak = pooled_layer[:, :, i, j].abs().max().item()
            if leak > tolerance:
                return False, (i, j)
    return True, None