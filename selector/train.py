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
from selector.loss import kl_loss, needle_topk_loss, topk_membership_loss
from selector.plotting import load_json, save_training_plot, write_json
from selector.recall import recall_metrics

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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


def save_resume_checkpoint(path, gate, opt, config, args, epoch, paths, history):
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
            "lambda_needle": args.lambda_needle,
            "topk": args.topk,
            "needle_topk": args.needle_topk,
            "proj_dim": args.proj_dim,
            "eval_every": args.eval_every,
            "budgets": list(args.budgets),
            "out": args.out,
            "seed": args.seed,
            "init_gate": args.init_gate,
            "shuffle_targets": args.shuffle_targets,
        },
        "targets": list(paths),
        "history": history,
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

    return int(ckpt.get("epoch", 0)), ckpt.get("history")


def make_progress(args, start_epoch):
    if args.progress == "off":
        return None
    if tqdm is None:
        if args.progress == "on":
            print("tqdm is not installed; progress bar disabled")
        return None
    total = max(args.epochs - start_epoch + 1, 0)
    return tqdm(range(start_epoch, args.epochs + 1),
                total=total, desc="training", unit="epoch")


def emit(msg, progress):
    if progress is not None:
        progress.write(msg)
    else:
        print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="teacher .pt paths or globs (must have pooledQ/pooledK)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional torch RNG seed; use one fixed seed for sweeps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-topk", type=float, default=0.5,
                    help="weight of the top-k membership term (needle retention)")
    ap.add_argument("--lambda-needle", type=float, default=0.0,
                    help="weight of reader-row needle top-k retention; 0 disables it")
    ap.add_argument("--topk", type=int, default=8,
                    help="k for the membership term; match the KS1 budget")
    ap.add_argument("--needle-topk", type=int, default=None,
                    help="needle retention budget; defaults to --topk")
    ap.add_argument("--proj-dim", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--init-gate",
        help="initialize weights from a gate checkpoint; cannot combine with --resume")
    ap.add_argument(
        "--shuffle-targets", action=argparse.BooleanOptionalAction, default=True,
        help="shuffle document order every epoch (default: enabled)")
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
    ap.add_argument("--progress", choices=["auto", "on", "off"], default="auto",
                    help="show a tqdm epoch progress bar when available")
    ap.add_argument("--plot", default=None,
                    help="training graph path; default is --out with .training.png suffix")
    ap.add_argument("--history", default=None,
                    help="metric history JSON path; default is --out with .history.json suffix")
    args = ap.parse_args()

    if args.resume and args.init_gate:
        raise SystemExit("--resume and --init-gate cannot be combined")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
    if args.init_gate:
        initial = torch.load(args.init_gate, map_location=args.device)
        if initial.get("config") != config:
            raise SystemExit(
                f"initial gate config {initial.get('config')} does not "
                f"match requested config {config}")
        gate.load_state_dict(initial["state_dict"])
        print(f"initialized gate weights from {args.init_gate}")
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    print(f"gate params: {sum(p.numel() for p in gate.parameters())}  device={args.device}")

    resume_path = args.resume_out or checkpoint_path(args.out)
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_graphs")
    model_name = os.path.splitext(os.path.basename(args.out))[0]
    plot_path = args.plot or os.path.join(graph_dir, f"{model_name}.png")
    history_path = args.history or os.path.join(graph_dir, f"{model_name}.json")
    empty_history = {
        "epochs": [], "kl": [], "topk": [], "needle": [], "eval_epochs": [],
        "eval_recall8": {}, "eval_needle8": {},
    }
    # A fresh run must not inherit a partial history left by a prior crash.
    # Resume checkpoints carry their own matching history below.
    history = load_json(history_path, empty_history) if args.resume else empty_history
    history.setdefault("needle", [])
    start_epoch = 1
    if args.resume:
        resumed_epoch, checkpoint_history = load_resume_checkpoint(
            args.resume, gate, opt, config, args.device)
        if checkpoint_history is not None:
            history = checkpoint_history
        start_epoch = resumed_epoch + 1
        print(f"resumed checkpoint {args.resume} at epoch {resumed_epoch}; "
              f"continuing through epoch {args.epochs}")

    if start_epoch > args.epochs:
        print(f"checkpoint is already past requested epochs "
              f"({start_epoch - 1} >= {args.epochs}); saving final gate")

    progress = make_progress(args, start_epoch)
    epoch_iter = progress if progress is not None else range(start_epoch, args.epochs + 1)

    for epoch in epoch_iter:
        gate.train()
        tot, nrows = 0.0, 0
        topk_tot, topk_rows = 0.0, 0
        needle_tot, needle_groups = 0.0, 0
        document_order = (
            torch.randperm(len(docs)).tolist()
            if args.shuffle_targets else range(len(docs))
        )
        for document_index in document_order:
            doc = docs[document_index]
            scores = gate(doc["q_feat"], doc["k_feat"])
            kl, nv = kl_loss(scores, doc["target"], doc["cmask"])
            loss = kl
            if args.lambda_topk:
                tk, tnv = topk_membership_loss(
                    scores, doc["target"], doc["cmask"], k=args.topk)
                loss = loss + args.lambda_topk * tk
                topk_tot += tk.item() * tnv
                topk_rows += tnv
            if args.lambda_needle:
                needle_k = args.needle_topk or args.topk
                nl, nng = needle_topk_loss(
                    scores, doc["target"], doc["cmask"],
                    doc["meta"]["needle_block"], k=needle_k,
                )
                loss = loss + args.lambda_needle * nl
                needle_tot += nl.item() * nng
                needle_groups += nng
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += kl.item() * nv
            nrows += nv

        kl_avg = tot / max(nrows, 1)
        topk_avg = topk_tot / max(topk_rows, 1) if args.lambda_topk else None
        needle_avg = needle_tot / max(needle_groups, 1) if args.lambda_needle else None
        history["epochs"].append(epoch)
        history["kl"].append(kl_avg)
        history["topk"].append(topk_avg)
        history["needle"].append(needle_avg)
        if progress is not None:
            postfix = {"KL": f"{kl_avg:.4f}"}
            if topk_avg is not None:
                postfix["topk"] = f"{topk_avg:.4f}"
            if needle_avg is not None:
                postfix["needle"] = f"{needle_avg:.4f}"
            progress.set_postfix(postfix)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            gate.eval()
            line = [f"epoch {epoch:>4}  KL={kl_avg:.4f}"]
            if args.lambda_topk:
                line.append(f"topk={topk_avg:.4f}")
            if args.lambda_needle:
                line.append(f"needle={needle_avg:.4f}")
            history["eval_epochs"].append(epoch)
            buckets = {}
            with torch.no_grad():
                for doc in docs:
                    sc = gate(doc["q_feat"], doc["k_feat"])
                    met = recall_metrics(sc, doc["target"], doc["cmask"],
                                         tuple(args.budgets), doc["meta"]["needle_block"])
                    r8 = met.get("recall@8", float("nan"))
                    nh = met.get("needle_hit@8", None)
                    requested = doc["meta"].get(
                        "requested_length", doc["meta"]["seq_len"])
                    tag = f"len{requested}"
                    bucket = buckets.setdefault(
                        tag, {"recall": [], "needle": []})
                    bucket["recall"].append(r8)
                    if nh is not None:
                        bucket["needle"].append(nh)
            for tag in sorted(buckets):
                recall_values = buckets[tag]["recall"]
                needle_values = buckets[tag]["needle"]
                mean_recall = sum(recall_values) / len(recall_values)
                mean_needle = (
                    sum(needle_values) / len(needle_values)
                    if needle_values else None
                )
                history["eval_recall8"].setdefault(
                    tag, []).append(mean_recall)
                history["eval_needle8"].setdefault(
                    tag, []).append(mean_needle)
                line.append(
                    f"{tag} n={len(recall_values)} r@8={mean_recall:.3f}"
                    + (
                        f" ndl@8={mean_needle:.2f}"
                        if mean_needle is not None else ""
                    )
                )
            emit("  ".join(line), progress)
            write_json(history_path, history)
            try:
                if not save_training_plot(history, plot_path):
                    emit(
                        "matplotlib not installed; skipped training graph",
                        progress,
                    )
            except ValueError as error:
                emit(f"skipped malformed training graph: {error}", progress)

        if args.save_every and (
                epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs):
            save_resume_checkpoint(resume_path, gate, opt, config, args, epoch, paths, history)
            emit(f"saved resume checkpoint -> {resume_path}", progress)
            if args.keep_epoch_checkpoints:
                root, ext = os.path.splitext(resume_path)
                epoch_path = f"{root}.e{epoch:04d}{ext or '.pt'}"
                save_resume_checkpoint(epoch_path, gate, opt, config, args, epoch, paths, history)
                emit(f"saved epoch checkpoint -> {epoch_path}", progress)

    if progress is not None:
        progress.close()

    save_gate(args.out, gate, config)
    print(f"saved gate -> {args.out}")
    print(f"training history -> {history_path}")
    if os.path.exists(plot_path):
        print(f"training graph -> {plot_path}")


if __name__ == "__main__":
    main()



