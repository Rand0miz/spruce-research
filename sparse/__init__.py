"""Sparse-prefill attention integration.

+The PyTorch path in :mod:`sparse.attention` is the correctness reference for
Stage 3.2.  Stage 3.3 can replace its score/value computation with Triton
without changing the selected-block contract.
"""

from sparse.attention import (
    SPARSE_PREFILL_ATTENTION,
    register_sparse_prefill_attention,
    sparse_prefill_attention_forward,
)
from sparse.plotting import save_sparse_replay_plot

__all__ = [
    "SPARSE_PREFILL_ATTENTION",
    "register_sparse_prefill_attention",
    "sparse_prefill_attention_forward",
    "save_sparse_replay_plot",
]
