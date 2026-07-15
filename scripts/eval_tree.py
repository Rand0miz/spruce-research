"""Evaluate a trained gate against teacher targets at every key-tree level.

This is the tree-aware version of scripts/eval_gate.py. It scores each query block
against every key-tree level, reports set recall / teacher-mass coverage, and treats
needle hits at parent levels as "selected node range contains the needle block".

Usage:
  python scripts/eval_tree.py --gate selector_ckpt/flat_gate.pt \
         --targets teacher_targets/heldout_*.pt
"""
import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from selector.gate import FlatGate
from selector.loss import kl_loss
from selector.targets import load_teacher
from selector.tree import build_key_tree, build_target_tree


def load_gate(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    gate = FlatGate(cfg["num_layers"], cfg["head_dim"], cfg["proj_dim"]).to(device)
    gate.load_state_dict(ckpt["state_dict"])
    gate.eval()
    return gate, cfg


def expand_paths(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    return paths


@torch.no_grad()
def tree_recall_metrics(scores, target, cmask, starts, ends, budgets, needle_block=None):
    """Tree-aware recall metrics.

    scores/target: [L,G,qb,num_nodes]
    cmask: [qb,num_nodes]
    starts/ends: [num_nodes] leaf-block ranges for each tree node
    """
    L, G, qb, num_nodes = scores.shape
    neg_inf = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~cmask[None, None], neg_inf)

    valid = target.sum(dim=-1) > 0.5
    visible_count = cmask.sum(dim=-1)
    out = {}

    for r in budgets:
        rr = min(r, num_nodes)
        s_top = masked.topk(rr, dim=-1).indices
        t_top = target.topk(rr, dim=-1).indices

        member = torch.zeros_like(target, dtype=torch.bool)
        member.scatter_(-1, s_top, True)
        overlap = member.gather(-1, t_top).sum(dim=-1).float()

        enough = (visible_count >= rr)[None, None].expand(L, G, qb)
        m = valid & enough
        denom = m.sum().clamp_min(1)
        out[f"recall@{r}"] = float((overlap / rr * m).sum() / denom)

        coverage = (target * member).sum(dim=-1)
        out[f"coverage@{r}"] = float((coverage * m).sum() / denom)

        omember = torch.zeros_like(target, dtype=torch.bool)
        omember.scatter_(-1, t_top, True)
        ocov = (target * omember).sum(dim=-1)
        out[f"oracle_cov@{r}"] = float((ocov * m).sum() / denom)

    if needle_block is not None and 0 <= needle_block:
        contains_needle = (starts <= needle_block) & (needle_block < ends)
        if bool(contains_needle.any()):
            reader = masked[:, :, -1, :]
            treader = target.masked_fill(~cmask[None, None], 0.0)[:, :, -1, :]
            for k in budgets:
                kk = min(k, num_nodes)
                top = reader.topk(kk, dim=-1).indices
                hit = contains_needle[top].any(dim=-1).float()
                out[f"needle_hit@{k}"] = float(hit.mean())

                ttop = treader.topk(kk, dim=-1).indices
                thit = contains_needle[ttop].any(dim=-1).float()
                out[f"teacher_needle@{k}"] = float(thit.mean())

    return out


def format_budget_metrics(metrics, budgets, prefix):
    return "  ".join(f"{prefix}@{b}={metrics[f'{prefix}@{b}']:.3f}" for b in budgets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "selector_ckpt", "flat_gate.pt"))
    ap.add_argument("--targets", nargs="+", required=True,
                    help="HELD-OUT teacher .pt paths or globs")
    ap.add_argument("--radix", type=int, default=2)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="leaf-level recall@8 pass bar")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = expand_paths(args.targets)
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    gate, cfg = load_gate(args.gate, args.device)
    print(f"loaded gate {args.gate}  L={cfg['num_layers']} d={cfg['head_dim']} "
          f"proj={cfg['proj_dim']}  radix={args.radix}")

    worst_leaf_r8 = 1.0
    level_sums = {}
    level_counts = {}

    with torch.no_grad():
        for p in paths:
            doc = load_teacher(p, device=args.device)
            m = doc["meta"]
            if m["num_layers"] != cfg["num_layers"] or m["head_dim"] != cfg["head_dim"]:
                raise SystemExit(f"{p} shape {m} mismatches gate config {cfg}")

            key_levels = build_key_tree(doc["k_feat"], radix=args.radix)
            target_levels = build_target_tree(
                doc["target"], row_mass=doc.get("row_mass"), radix=args.radix)

            print(f"\n{os.path.basename(p)}  seq={m['seq_len']} qb={m['qb']} "
                  f"kb={m['kb']} needle_blk={m['needle_block']}")

            for key_level, target_level in zip(key_levels, target_levels):
                scores = gate(doc["q_feat"], key_level.features)
                kl, n_valid = kl_loss(scores, target_level.target, target_level.cmask)
                met = tree_recall_metrics(
                    scores, target_level.target, target_level.cmask,
                    target_level.starts, target_level.ends,
                    tuple(args.budgets), m["needle_block"])

                if key_level.level == 0:
                    worst_leaf_r8 = min(worst_leaf_r8, met.get("recall@8", float("nan")))

                level_sums.setdefault(key_level.level, {})
                level_counts[key_level.level] = level_counts.get(key_level.level, 0) + 1
                level_sums[key_level.level]["kl"] = (
                    level_sums[key_level.level].get("kl", 0.0) + float(kl))
                for name, val in met.items():
                    level_sums[key_level.level][name] = (
                        level_sums[key_level.level].get(name, 0.0) + float(val))

                rec = format_budget_metrics(met, args.budgets, "recall")
                cov = "  ".join(f"cov@{b}={met[f'coverage@{b}']:.3f}"
                                f"/{met[f'oracle_cov@{b}']:.3f}"
                                for b in args.budgets)
                ndl = "  ".join(f"ndl@{b}={met[f'needle_hit@{b}']:.2f}"
                                for b in args.budgets if f"needle_hit@{b}" in met)
                tndl = "  ".join(f"tndl@{b}={met[f'teacher_needle@{b}']:.2f}"
                                 for b in args.budgets if f"teacher_needle@{b}" in met)

                print(f"  level {key_level.level:<2} nodes={key_level.features.shape[2]:<4} "
                      f"KL={float(kl):.4f} rows={int(n_valid)}")
                print(f"    {rec}")
                print(f"    {cov}")
                if ndl:
                    print(f"    {ndl}")
                if tndl:
                    print(f"    {tndl}")

    if len(paths) > 1:
        print("\nmean over documents")
        for level in sorted(level_sums):
            n = level_counts[level]
            avg = {name: val / n for name, val in level_sums[level].items()}
            rec = format_budget_metrics(avg, args.budgets, "recall")
            cov = "  ".join(f"cov@{b}={avg[f'coverage@{b}']:.3f}"
                            f"/{avg[f'oracle_cov@{b}']:.3f}"
                            for b in args.budgets)
            print(f"  level {level:<2} KL={avg['kl']:.4f}")
            print(f"    {rec}")
            print(f"    {cov}")

    ok = worst_leaf_r8 >= args.threshold
    print(f"\nworst leaf recall@8 = {worst_leaf_r8:.3f}  "
          f"{'PASS' if ok else 'FAIL'} (bar {args.threshold})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
