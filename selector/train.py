"""Train the flat selector gate against dumped teacher targets.

Full supervision, no pruning at train time: score every key-block for every query-block,
forward-KL against the teacher marginal, backprop into Wq/Wk only. The base model is
never loaded here -- training reads the dumped tensors, so it is cheap to run many epochs
on the laptop. Each document is one "batch"; the per-layer gate is shared across
documents of different lengths (that is the whole point of pooling to fixed-dim features).

Usage:
  python -m selector.train --targets teacher_targets/teacher_*.pt --epochs 200
  python -m selector.train --targets teacher_targets/teacher_*.pt --epochs 200 \
      --resume selector_ckpt/flat_gate.resume.pt
"""
import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from selector.targets import load_teacher
from selector.gate import FlatGate
from selector.loss import kl_loss, topk_membership_loss
from selector.recall import recall_metrics


def gate_config(num_layers, head_dim, proj_dim):
    return {"num_layers": num_layers, "head_dim": head_dim,
            "proj_dim": proj_dim or head_dim}


def checkpoint_path(out_path):
    root, ext = os.path.splitext(out_path)
    return f"{root}.resume{ext or '.pt'}"


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_gate(path, gate, config):
    atomic_torch_save({"state_dict": gate.state_dict(), "config": config}, path)


def save_resume_checkpoint(path, gate, opt, config, args, epoch, paths):
    atomic_torch_save({
        "kind": "spruce_selector_train_checkpoint",
        "epoch": int(epoch),
        "state_dict": gate.state_dict(),
        "optimizer": opt.state_dict(),
        "config": config,
        "train_args": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_topk": args.lambda_topk,
            "topk": args.topk,
            "proj_dim": args.proj_dim,
            "eval_every": args.eval_every,
            "budgets": list(args.budgets),
            "out": args.out,
        },
        "targets": list(paths),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }, path)


def load_resume_checkpoint(path, gate, opt, config, device):
    ckpt = torch.load(path, map_location=device)
    ckpt_config = ckpt.get("config")
    if ckpt_config != config:
        raise SystemExit(
            f"resume checkpoint config {ckpt_config} does not match current {config}")

    gate.load_state_dict(ckpt["state_dict"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"].cpu())
    if (device == "cuda" and ckpt.get("cuda_rng_state_all") is not None
            and torch.cuda.is_available()):
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

    return int(ckpt.get("epoch", 0))


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
    ap.add_argument("--resume", default=None,
                    help="resume from a full training checkpoint saved by --save-every")
    ap.add_argument("--save-every", type=int, default=25,
                    help="save a resume checkpoint every N epochs; 0 disables periodic saves")
    ap.add_argument("--resume-out", default=None,
                    help="path for resume checkpoints; default is --out with .resume.pt suffix")
    ap.add_argument("--keep-epoch-checkpoints", action="store_true",
                    help="also keep per-epoch checkpoint files beside the rolling resume file")
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

    config = gate_config(L, d, args.proj_dim)
    gate = FlatGate(L, d, args.proj_dim).to(args.device)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    print(f"gate params: {sum(p.numel() for p in gate.parameters())}  device={args.device}")

    resume_path = args.resume_out or checkpoint_path(args.out)
    start_epoch = 1
    if args.resume:
        resumed_epoch = load_resume_checkpoint(args.resume, gate, opt, config, args.device)
        start_epoch = resumed_epoch + 1
        print(f"resumed checkpoint {args.resume} at epoch {resumed_epoch}; "
              f"continuing through epoch {args.epochs}")

    if start_epoch > args.epochs:
        print(f"checkpoint is already past requested epochs "
              f"({start_epoch - 1} >= {args.epochs}); saving final gate")

    for epoch in range(start_epoch, args.epochs + 1):
        gate.train()
        tot, nrows = 0.0, 0
        topk_tot, topk_rows = 0.0, 0
        for doc in docs:
            scores = gate(doc["q_feat"], doc["k_feat"])
            kl, nv = kl_loss(scores, doc["target"], doc["cmask"])
            loss = kl
            if args.lambda_topk:
                tk, tnv = topk_membership_loss(
                    scores, doc["target"], doc["cmask"], k=args.topk)
                loss = loss + args.lambda_topk * tk
                topk_tot += tk.item() * tnv
                topk_rows += tnv
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += kl.item() * nv
            nrows += nv
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            gate.eval()
            line = [f"epoch {epoch:>4}  KL={tot / max(nrows, 1):.4f}"]
            if args.lambda_topk:
                line.append(f"topk={topk_tot / max(topk_rows, 1):.4f}")
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

        if args.save_every and (
                epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs):
            save_resume_checkpoint(resume_path, gate, opt, config, args, epoch, paths)
            print(f"saved resume checkpoint -> {resume_path}")
            if args.keep_epoch_checkpoints:
                root, ext = os.path.splitext(resume_path)
                epoch_path = f"{root}.e{epoch:04d}{ext or '.pt'}"
                save_resume_checkpoint(epoch_path, gate, opt, config, args, epoch, paths)
                print(f"saved epoch checkpoint -> {epoch_path}")

    save_gate(args.out, gate, config)
    print(f"saved gate -> {args.out}")


if __name__ == "__main__":
    main()
