import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed, write PNGs straight to disk
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from teacher.extract import get_attentions
from teacher.validate import check_causal
from teacher.pool import pool_attention, rollup
from eval.haystack import build_haystack

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BLOCK = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep target_tokens small on a laptop: full attention is [layers,heads,n,n].
# 1500 tokens ~= 1.5 GB fp32. Drop to 1024 if you OOM.
LENGTHS = [1024, 1500]
DEPTHS  = [0.0, 0.25, 0.5, 0.75, 1.0]
BUDGETS = (1, 2, 4, 8, 16)   # top-k block budgets to sweep
HEATMAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ks1_heatmaps")

NEEDLE   = "The secret access code is heron-4417."
FILLER   = "The quarterly logistics report noted no unusual activity."
QUESTION = "\n\nWhat is the secret access code? Answer with the exact sentence."

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    attn_implementation="eager",
).to(DEVICE)
model.eval()

def needle_block_index(prompt, needle=NEEDLE, block_size=BLOCK):
    """Which key-block the needle lands in (block-level, off-by-one-token tolerant)."""
    char_idx = prompt.index(needle)
    prefix_len = len(tok(prompt[:char_idx])["input_ids"])
    return prefix_len // block_size

def aggregate_pooled(attentions):
    """Sum pooled block-mass over ALL layers and heads -> [q_blocks, k_blocks].
    This is the 'is the signal present in aggregate attention' proxy; a real
    predictor would learn a layer/head weighting, but for an existence check
    the summed teacher signal is what matters."""
    agg = None
    for layer in attentions:                       # each: [1, heads, q, k]
        pooled = pool_attention(layer.to(torch.float32), BLOCK)  # [1, h, qb, kb]
        s = pooled.sum(dim=1)[0]                    # sum heads -> [qb, kb]
        agg = s if agg is None else agg + s
    return agg

def save_heatmap(agg, n_blk, target_tokens, depth):
    """Block heatmap of aggregated attention mass [q_blocks, k_blocks].
    Log-scaled (mass spans orders of magnitude). Marks the needle key-block
    (vertical line) and the reader query-block = last row (horizontal line):
    their crossing is where the model 'looks at the needle when answering'.
    """
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    grid = agg.detach().cpu().to(torch.float32).numpy()
    grid = np.log1p(grid)                      # compress dynamic range

    qb, kb = grid.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.axvline(n_blk + 0.5, color="red", lw=1.0, ls="--")   # needle key-block
    ax.axvline(n_blk - 0.5, color="red", lw=1.0, ls="--")
    ax.axhline(qb - 1, color="white", lw=1.0, ls=":")        # reader (question) row
    ax.set_title(f"pooled attention mass  len={target_tokens} depth={depth}\n"
                 f"needle key-block={n_blk}/{kb}  (red), reader row (white)")
    ax.set_xlabel("key block")
    ax.set_ylabel("query block")
    fig.colorbar(im, ax=ax, label="log(1 + mass)")
    path = os.path.join(HEATMAP_DIR, f"heatmap_{target_tokens}_{depth}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def ks1_lite(target_tokens, depth, plot=True):
    prompt, ndl = build_haystack(tok, target_tokens, NEEDLE, depth, FILLER)
    n_blk = needle_block_index(prompt, ndl, BLOCK)
    full = prompt + QUESTION

    attentions, seq_len = get_attentions(model, tok, full, device=DEVICE)
    agg = aggregate_pooled(attentions) # [q_blocks, k_blocks]

    # sanity: pooled leaf must not leak future (upper triangle ~ 0)
    causal_ok, _ = check_causal(agg[None, None])

    # Check A: leaf top-k contains the needle block
    reader = agg[-1]                               # last query block = the question
    order = torch.argsort(reader, descending=True) # key-blocks best -> worst
    rank = (order == n_blk).nonzero().item()       # 0 = needle block ranked #1
    hit_at_k = {k: (n_blk in set(order[:k].tolist())) for k in BUDGETS} 

     # Check B: traversal-safe (needle's ancestor stays in top-k each level)
    # This is the O(log n) pruning condition: descend only into top branches;
    # if the needle's ancestor ever falls outside budget, the subtree gets
    # pruned and the needle is lost before you reach the leaf.
    levels = rollup(agg[None, None])               # [ [1,1,qb,kb], ... coarser ]
    traversal_safe = {}
    for k in BUDGETS:
        safe = True
        for L, lvl in enumerate(levels):
            row = lvl[0, 0, -1]                    # reader row at this level
            ancestor = n_blk // (2 ** L)           # needle's ancestor block here
            top = set(torch.argsort(row, descending=True)[:k].tolist())
            if ancestor not in top:
                safe = False
                break
        traversal_safe[k] = safe
    
    heatmap_path = save_heatmap(agg, n_blk, target_tokens, depth) if plot else None

    return {
        "target": target_tokens, "depth": depth,
        "needle_block": n_blk, "num_blocks": agg.shape[-1],
        "needle_rank": rank, "causal_ok": causal_ok,
        "hit@k": hit_at_k, "traversal_safe@k": traversal_safe,
        "heatmap": heatmap_path,
    }

if __name__ == "__main__":
    for target in LENGTHS:
        for depth in DEPTHS:
            result = ks1_lite(target, depth)
            print(f"target={target:>7}, depth={depth:<4} "
                  f"needle_block={result['needle_block']:<3} "
                  f"num_blocks={result['num_blocks']:<3} "
                  f"needle_rank={result['needle_rank']:<3} "
                  f"causal_ok={result['causal_ok']} "
                  f"hit@k={result['hit@k']} "
                  f"traversal_safe@k={result['traversal_safe@k']}")