"""Evaluate a trained flat gate against teacher targets (the KS1 recall proxy).

Loads a saved flat_gate.pt, scores it on one or more teacher .pt targets, and
reports recall@k and needle_hit@k. This is the honest accuracy check that
train.py's inline eval is NOT — point --targets at documents the gate did not
train on, or the numbers are overfit and meaningless.

NOTE: recall@k is the block-level proxy, not the KS1 ">95% of dense RULER"
number. That true number needs the sparse generate path (kernel), which does
not exist yet. Use this to gate progress: if recall is bad here, RULER cannot
be good later.

Usage:
  python scripts/eval_gate.py --gate selector_ckpt/flat_gate.pt \
         --targets teacher_targets/heldout_*.pt
"""
import os, sys, glob, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from selector.targets import load_teacher
from selector.gate import FlatGate
from selector.recall import recall_metrics


def load_gate(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    gate = FlatGate(cfg["num_layers"], cfg["head_dim"], cfg["proj_dim"]).to(device)
    gate.load_state_dict(ckpt["state_dict"])
    gate.eval()
    return gate, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "selector_ckpt", "flat_gate.pt"))
    ap.add_argument("--targets", nargs="+", required=True,
                    help="HELD-OUT teacher .pt paths or globs")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="recall@8 pass bar (KS1 proxy)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = []
    for p in args.targets:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    gate, cfg = load_gate(args.gate, args.device)
    print(f"loaded gate {args.gate}  L={cfg['num_layers']} d={cfg['head_dim']} "
          f"proj={cfg['proj_dim']}")

    worst_r8 = 1.0
    with torch.no_grad():
        for p in paths:
            doc = load_teacher(p, device=args.device)
            m = doc["meta"]
            if m["num_layers"] != cfg["num_layers"] or m["head_dim"] != cfg["head_dim"]:
                raise SystemExit(f"{p} shape {m} mismatches gate config {cfg}")
            sc = gate(doc["q_feat"], doc["k_feat"])
            met = recall_metrics(sc, doc["target"], doc["cmask"],
                                 tuple(args.budgets), m["needle_block"])
            r8 = met.get("recall@8", float("nan"))
            worst_r8 = min(worst_r8, r8)
            rec = "  ".join(f"r@{b}={met[f'recall@{b}']:.3f}" for b in args.budgets)
            cov = "  ".join(f"cov@{b}={met[f'coverage@{b}']:.3f}"
                            f"/{met[f'oracle_cov@{b}']:.3f}" for b in args.budgets)
            ndl = "  ".join(f"ndl@{b}={met[f'needle_hit@{b}']:.2f}"
                            for b in args.budgets if f"needle_hit@{b}" in met)
            tndl = "  ".join(f"tndl@{b}={met[f'teacher_needle@{b}']:.2f}"
                             for b in args.budgets if f"teacher_needle@{b}" in met)
            print(f"\n{os.path.basename(p)}  seq={m['seq_len']} "
                  f"needle_blk={m['needle_block']}")
            print(f"  {rec}")
            print(f"  {cov}")
            if ndl:
                print(f"  {ndl}")
            if tndl:
                print(f"  {tndl}")

    ok = worst_r8 >= args.threshold
    print(f"\nworst recall@8 = {worst_r8:.3f}  "
          f"{'PASS' if ok else 'FAIL'} (bar {args.threshold})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
