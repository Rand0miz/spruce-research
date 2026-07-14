"""Train the flat selector gate against dumped teacher targets.

Full supervision, no pruning at train time: score every key-block for every query-block,
forward-KL against the teacher marginal, backprop into Wq/Wk only. The base model is
never loaded here — training reads the dumped tensors, so it is cheap to run many epochs
on the laptop. Each document is one "batch"; the per-layer gate is shared across
documents of different lengths (that is the whole point of pooling to fixed-dim features).

Usage:
  python -m selectors.train --targets teacher_targets/teacher_*.pt --epochs 200
"""
import os, sys, glob, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from selector.targets import load_teacher
from selector.gate import FlatGate
from selector.loss import combined_loss
from selector.recall import recall_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="teacher .pt paths or globs (must have pooledQ/pooledK)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-topk", type=float, default=0.5,
                    help="weight of the top-k membership term (needle retention)")
    ap.add_argument("--topk", type=int, default=8,
                    help="k for the membership term; match the KS1 budget")
    ap.add_argument("--proj-dim", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "selector_ckpt", "flat_gate.pt"))
    args = ap.parse_args()

    paths = []
    for p in args.targets:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not paths:
        raise SystemExit(f"no target files matched {args.targets}")

    docs = [load_teacher(p, device=args.device) for p in paths]
    L = docs[0]["meta"]["num_layers"]
    d = docs[0]["meta"]["head_dim"]
    for doc, p in zip(docs, paths):
        m = doc["meta"]
        assert m["num_layers"] == L and m["head_dim"] == d, \
            f"{p} shape {m} mismatches first doc (L={L}, d={d})"
        print(f"loaded {os.path.basename(p)}  seq={m['seq_len']} qb={m['qb']} "
              f"kb={m['kb']} G={m['num_groups']} needle_blk={m['needle_block']}")

    gate = FlatGate(L, d, args.proj_dim).to(args.device)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    print(f"gate params: {sum(p.numel() for p in gate.parameters())}  device={args.device}")

    for epoch in range(1, args.epochs + 1):
        gate.train()
        tot, kl_tot, bce_tot, nrows = 0.0, 0.0, 0.0, 0
        for doc in docs:
            scores = gate(doc["q_feat"], doc["k_feat"])
            loss, parts = combined_loss(scores, doc["target"], doc["cmask"],
                                        lambda_topk=args.lambda_topk, k=args.topk)
            opt.zero_grad(); loss.backward(); opt.step()
            nv = parts["n"]
            tot += loss.item() * nv; kl_tot += parts["kl"] * nv
            bce_tot += parts["bce"] * nv; nrows += nv
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            gate.eval()
            nr = max(nrows, 1)
            line = [f"epoch {epoch:>4}  loss={tot/nr:.4f} KL={kl_tot/nr:.4f} "
                    f"BCE={bce_tot/nr:.4f}"]
            with torch.no_grad():
                for doc in docs:
                    sc = gate(doc["q_feat"], doc["k_feat"])
                    met = recall_metrics(sc, doc["target"], doc["cmask"],
                                         tuple(args.budgets), doc["meta"]["needle_block"])
                    r8 = met.get("recall@8", float("nan"))
                    nh = met.get("needle_hit@8", None)
                    tag = f"len{doc['meta']['seq_len']}"
                    line.append(f"{tag} r@8={r8:.3f}" +
                                (f" ndl@8={nh:.2f}" if nh is not None else ""))
            print("  ".join(line))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": gate.state_dict(),
                "config": {"num_layers": L, "head_dim": d,
                           "proj_dim": args.proj_dim or d}}, args.out)
    print(f"saved gate -> {args.out}")


if __name__ == "__main__":
    main()
