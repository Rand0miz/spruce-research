import os, sys, argparse, torch, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.haystack import build_haystack
from teacher.chunked_extract import get_pooled_targets

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_BANK = os.path.join(ROOT, "scripts", "prompt_banks", "train.json")
DEFAULT_HELDOUT_BANK = os.path.join(ROOT, "scripts", "prompt_banks", "heldout.json")


def _safe_id(s):
    """Filesystem-safe prompt id fragment."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s.strip())
    return s.strip("-") or "prompt"


def load_prompt_bank(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'cases' list")
    required = {"id", "needle", "filler", "question"}
    for i, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"{path} case {i} missing keys: {sorted(missing)}")
        for key in required:
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError(f"{path} case {i} key {key!r} must be a non-empty string")
    return cases


def needle_block_index(tok, prompt, needle, block_size):
    """Which key-block the needle lands in (so downstream knows where the signal is)."""
    char_idx = prompt.index(needle)
    prefix_len = len(tok(prompt[:char_idx])["input_ids"])
    return prefix_len // block_size


def save_heatmap(pooled_stack, n_blk, seq_len, depth, path):
    """Block heatmap of teacher target mass, summed over all layers+heads -> [qb, kb].
    Log-scaled (mass spans orders of magnitude). Red dashed = needle key-block,
    white dotted = reader (question) row = last query block; their crossing is
    where the model looks at the needle when answering. Same convention as
    ks1_lite.save_heatmap so figures are comparable across scripts.
    """
    import matplotlib
    matplotlib.use("Agg")                       # headless: write PNG straight to disk
    import matplotlib.pyplot as plt
    import numpy as np

    agg = pooled_stack[0].sum(dim=(0, 1))       # [1,L,H,qb,kb] -> sum L,H -> [qb,kb]
    grid = np.log1p(agg.detach().cpu().to(torch.float32).numpy())
    qb, kb = grid.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.axvline(n_blk + 0.5, color="red", lw=1.0, ls="--")   # needle key-block
    ax.axvline(n_blk - 0.5, color="red", lw=1.0, ls="--")
    ax.axhline(qb - 1, color="white", lw=1.0, ls=":")        # reader (question) row
    ax.set_title(f"teacher pooled mass  len={seq_len} depth={depth}\n"
                 f"needle key-block={n_blk}/{kb} (red), reader row (white)")
    ax.set_xlabel("key block")
    ax.set_ylabel("query block")
    fig.colorbar(im, ax=ax, label="log(1 + mass)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768],
                    help="context lengths to extract; defaults to 16k and 32k")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--depth", type=float, default=None,
                    help="single fixed needle depth; overrides random depth and --depths")
    ap.add_argument("--depths", type=float, nargs="+", default=None,
                    help="candidate needle depths to randomly sample from; overrides random depth")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.05, 0.95],
                    metavar=("MIN", "MAX"),
                    help="continuous random depth range used when --depth/--depths are omitted; default: 0.05 0.95")
    ap.add_argument("--store-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teacher_targets"))
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU run; default refuses to run without CUDA")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="skip writing the per-length pooled-mass heatmap PNG")
    ap.add_argument("--heldout", type=bool, default=False,
                    help="if True, use the heldout needle for teacher target extraction")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU not available (torch is CPU-only or driver missing). "
            "Refusing to run on CPU. Install a CUDA torch build "
            "(pip install torch --index-url https://download.pytorch.org/whl/cu126 "
            "--force-reinstall --no-deps), or pass --allow-cpu to override.")

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp16 weights; <=32768 native context -> no rope_scaling. sdpa is the restore target.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", attn_implementation="sdpa"
    ).to(DEVICE).eval()

    store_dtype = getattr(torch, args.store_dtype)
    q_len = len(tok(QUESTION)["input_ids"])

    for L in args.lengths:
        assert L <= 32768, f"{L} > 32768 native max; needs YaRN rope_scaling (not set)."
        prompt, ndl = build_haystack(tok, L - q_len, NEEDLE if not args.heldout else HELDOUT_NEEDLE, args.depth, FILLER)
        full = prompt + (QUESTION if not args.heldout else HELDOUT_QUESTION)
        n_blk = needle_block_index(tok, full, ndl, args.block)

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pooled_stack, pooledQ_stack, pooledK_stack, seq_len = get_pooled_targets(
            model, tok, full, device=DEVICE, block_size=args.block)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        assert seq_len <= 32768, f"seq_len {seq_len} exceeded native max after suffix."

        path = os.path.join(args.out, f"teacher_len{seq_len}_blk{args.block}_d{args.depth}.pt")
        torch.save({
            "pooled": pooled_stack.to(store_dtype),      # [1, L, H, qb, kb]  TARGET
            "pooledQ": pooledQ_stack.to(store_dtype),    # [1, L, H, qb, d]   selector input
            "pooledK": pooledK_stack.to(store_dtype),    # [1, L, kv, kb, d]  selector input
            "head_dim": pooledQ_stack.shape[-1],
            "num_kv_heads": pooledK_stack.shape[2],
            "seq_len": seq_len,
            "block_size": args.block,
            "model": MODEL,
            "depth": args.depth,
            "needle_block": n_blk,
            "num_layers": pooled_stack.shape[1],
            "num_heads": pooled_stack.shape[2],
            "store_dtype": args.store_dtype,
        }, path)
        print(f"len={seq_len:>6}  blocks={pooled_stack.shape[-1]:>4}  "
              f"needle_block={n_blk:>4}  peak={peak_gb:.2f}GB  saved={path}")

        if not args.no_heatmap:
            png = path[:-3] + ".png"
            save_heatmap(pooled_stack, n_blk, seq_len, args.depth, png)
            print(f"          heatmap={png}")


if __name__ == "__main__":
    main()
