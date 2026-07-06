import torch

def pool_attention(attn_layer, block_size):
    """
    attn_layer: [batch, heads, q_len, k_len] for one layer
    Returns: [batch, heads, q_blocks, k_blocks] pooled block-mass matrix
    """
    b, h, q_len, k_len = attn_layer.shape
    q_blocks = (q_len + block_size - 1) // block_size
    k_blocks = (k_len + block_size - 1) // block_size

    # Pad so q_len/k_len divide evenly into blocks.
    pad_q = q_blocks * block_size - q_len
    pad_k = k_blocks * block_size - k_len
    padded = torch.nn.functional.pad(attn_layer, (0, pad_k, 0, pad_q))

    # Reshape into blocks and sum mass each block
    padded = padded.view(b, h, q_blocks, block_size, k_blocks, block_size)
    pooled = padded.sum(dim=(3, 5))  # sum over block dimensions

    return pooled # [batch, heads, q_blocks, k_blocks]

def rollup(pooled):
    """
    pooled: [batch, heads, q_blocks, k_blocks] — leaf level.
    Returns a list of levels, leaf first, each half the size of the one before,
    stopping when a dimension can't be halved further.
    A parent cell's value is the SUM of its 2x2 children (start with sum, not max).
    """
    levels = [pooled]
    current = pooled
    while current.shape[-1] >= 2 and current.shape[-2] >= 2:
        b, h, q_blocks, k_blocks = current.shape
        # If odd, stop
        if q_blocks % 2 != 0 or k_blocks % 2 != 0:
            break
        current = current.view(b, h, q_blocks // 2, 2, k_blocks // 2, 2).sum(dim=(3, 5))
        levels.append(current)
    return levels