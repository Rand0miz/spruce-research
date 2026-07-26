"""Train the shared selector scorer against dumped teacher targets.

Leaf-only mode preserves the original flat-gate recipe. ``--tree-supervision``
scores every key node at every discriminative level with no pruning, using the
same scorer later applied during recursive traversal. The base model is never
loaded: training reads dumped tensors and updates Wq/Wk only. Each document is
one batch; the per-layer gate is shared across lengths through fixed-width
pooled features.

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
from selector.loss import (
    kl_loss,
    needle_topk_loss,
    needle_union_topk_loss,
    topk_boundary_loss,
    topk_membership_loss,
)
from selector.plotting import load_json, save_training_plot, write_json
from selector.recall import recall_metrics
from selector.tree import (
    ancestor_node_id,
    iter_tree_levels,
    tree_node_counts,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def gate_config(num_layers, head_dim, proj_dim):
    return {"num_layers": num_layers, "head_dim": head_dim,
            "proj_dim": proj_dim or head_dim}


def expand_target_paths(patterns):
    paths = []
    for pattern in patterns or ():
        matches = (
            sorted(glob.glob(pattern, recursive=True))
            if any(character in pattern for character in "*?[")
            else [pattern]
        )
        paths.extend(matches)
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def _sample_pool(indices, count):
    """Sample a pool without replacement until another cycle is required."""
    indices = list(indices)
    if count <= 0:
        return []
    if not indices:
        raise ValueError("cannot sample from an empty replay pool")
    sampled = []
    while len(sampled) < count:
        order = torch.randperm(len(indices)).tolist()
        remaining = count - len(sampled)
        sampled.extend(indices[index] for index in order[:remaining])
    return sampled


def mixed_epoch_order(
        natural_indices, replay_indices, natural_fraction, shuffle=True):
    """Use every natural document and sample replay to the requested ratio."""
    natural_indices = list(natural_indices)
    replay_indices = list(replay_indices)
    if not natural_indices:
        raise ValueError("natural target pool is empty")
    if not 0.0 < natural_fraction <= 1.0:
        raise ValueError("natural_fraction must be in (0, 1]")
    replay_count = round(
        len(natural_indices) * (1.0 - natural_fraction) / natural_fraction)
    replay_sample = _sample_pool(replay_indices, replay_count)
    combined = natural_indices + replay_sample
    if shuffle and len(combined) > 1:
        order = torch.randperm(len(combined)).tolist()
        combined = [combined[index] for index in order]
    return combined


def checkpoint_path(out_path):
    root, ext = os.path.splitext(out_path)
    return f"{root}.resume{ext or '.pt'}"


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def training_recipe(args):
    """Serializable settings that define the optimized selector objective."""
    return {
        "lr": args.lr,
        "lambda_topk": args.lambda_topk,
        "lambda_boundary": args.lambda_boundary,
        "lambda_needle": args.lambda_needle,
        "topk": args.topk,
        "topk_margin": args.topk_margin,
        "needle_topk": args.needle_topk,
        "needle_margin": args.needle_margin,
        "needle_objective": args.needle_objective,
        "needle_eligibility": args.needle_eligibility,
        "tree_supervision": args.tree_supervision,
        "tree_radix": args.tree_radix,
        "tree_beam": args.tree_beam,
        "natural_fraction": args.natural_fraction,
        "shuffle_targets": args.shuffle_targets,
    }


def save_gate(path, gate, config, train_args=None):
    payload = {"state_dict": gate.state_dict(), "config": config}
    if train_args is not None:
        payload["train_args"] = dict(train_args)
    atomic_torch_save(payload, path)


def save_resume_checkpoint(path, gate, opt, config, args, epoch, paths, history):
    atomic_torch_save({
        "kind": "spruce_selector_train_checkpoint",
        "epoch": int(epoch),
        "state_dict": gate.state_dict(),
        "optimizer": opt.state_dict(),
        "config": config,
        "train_args": {
            "epochs": args.epochs,
            "proj_dim": args.proj_dim,
            "eval_every": args.eval_every,
            "budgets": list(args.budgets),
            "out": args.out,
            "seed": args.seed,
            "init_gate": args.init_gate,
            "shuffle_targets": args.shuffle_targets,
            "offload_targets": args.offload_targets,
            "natural_fraction": args.natural_fraction,
            "natural_targets": list(args.natural_targets or ()),
            "replay_targets": list(args.replay_targets or ()),
            **training_recipe(args),
        },
        "targets": list(paths),
        "history": history,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }, path)


def validate_resume_recipe(saved_args, expected):
    """Reject objective drift while accepting pre-tree flat checkpoints."""
    saved_args = dict(saved_args or {})
    legacy_defaults = {
        "lambda_boundary": 0.0,
        "topk_margin": 0.0,
        "needle_margin": 0.0,
        "needle_objective": "group",
        "needle_eligibility": "teacher",
        "tree_supervision": False,
        "tree_radix": 2,
        "tree_beam": 8,
    }
    mismatches = []
    for name, wanted in expected.items():
        if name in saved_args:
            actual = saved_args[name]
        elif name in legacy_defaults:
            actual = legacy_defaults[name]
        else:
            # Very old resume files did not serialize every sampler setting.
            # Only compare fields that the checkpoint actually knows.
            continue
        if actual != wanted:
            mismatches.append(f"{name}: saved={actual!r} requested={wanted!r}")
    if mismatches:
        raise SystemExit(
            "resume training recipe differs from the checkpoint: "
            + "; ".join(mismatches))


def load_resume_checkpoint(
        path, gate, opt, config, device, expected_train_args=None):
    ckpt = torch.load(path, map_location=device)
    ckpt_config = ckpt.get("config")
    if ckpt_config != config:
        raise SystemExit(
            f"resume checkpoint config {ckpt_config} does not match current {config}")
    if expected_train_args is not None:
        validate_resume_recipe(
            ckpt.get("train_args"), expected_train_args)

    gate.load_state_dict(ckpt["state_dict"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"].cpu())
    if (device == "cuda" and ckpt.get("cuda_rng_state_all") is not None
            and torch.cuda.is_available()):
        # ``map_location="cuda"`` also moves RNG byte tensors to CUDA, but
        # PyTorch's RNG restoration API requires CPU ByteTensors.
        cuda_rng_states = [
            state.detach().cpu().to(dtype=torch.uint8)
            for state in ckpt["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

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


def move_document(doc, device):
    """Move one compact CPU teacher document and normalize it on compute."""
    device = torch.device(device)
    if doc["q_feat"].device == device and "mass" not in doc:
        return doc
    moved = {
        name: (
            value.to(
                device,
                dtype=(
                    torch.float32
                    if torch.is_floating_point(value) else value.dtype
                ),
                non_blocking=device.type == "cuda",
            )
            if torch.is_tensor(value) else value
        )
        for name, value in doc.items()
    }
    if "mass" in moved:
        mass = moved.pop("mass")
        eps = float(moved.pop("normalization_eps", 1e-9))
        row = mass.sum(dim=-1, keepdim=True)
        moved["target"] = mass / row.clamp_min(eps)
        moved["row_mass"] = row.squeeze(-1)
    return moved


def train_document(gate, optimizer, doc, args):
    """Accumulate one optimizer step across every supervised tree level.

    Each level is built and backpropagated independently, so the whole tree's
    activations are never retained at once. Component losses are normalized by
    the number of levels where they can affect a pruning decision; this keeps
    their lambda scales comparable to the former leaf-only recipe.
    """
    needle_k = args.needle_topk or args.topk
    if args.tree_supervision:
        node_counts = tree_node_counts(
            doc["meta"]["kb"], radix=args.tree_radix)
    else:
        node_counts = [doc["meta"]["kb"]]
    if not node_counts:
        raise ValueError("selector training requires at least two key blocks")

    kl_levels = len(node_counts)
    topk_levels = max(
        sum(nodes > args.topk for nodes in node_counts), 1)
    needle_levels = max(
        sum(nodes > needle_k for nodes in node_counts), 1)
    needle_loss_fn = (
        needle_union_topk_loss
        if args.needle_objective == "union"
        else needle_topk_loss
    )

    stats = {
        "kl_total": 0.0, "kl_rows": 0,
        "topk_total": 0.0, "topk_rows": 0,
        "boundary_total": 0.0, "boundary_rows": 0,
        "needle_total": 0.0, "needle_units": 0,
        "levels": 0,
    }
    optimizer.zero_grad()
    for key_level, target_level in iter_tree_levels(
            doc["k_feat"], doc["target"], doc["cmask"],
            radix=args.tree_radix):
        nodes = int(key_level.features.shape[2])
        if nodes == 1:
            break
        if not args.tree_supervision and key_level.level > 0:
            break

        scores = gate(doc["q_feat"], key_level.features)
        kl, valid_rows = kl_loss(
            scores, target_level.target, target_level.cmask)
        level_loss = kl / kl_levels
        stats["kl_total"] += kl.item() * valid_rows
        stats["kl_rows"] += valid_rows

        if args.lambda_topk and nodes > args.topk:
            membership, positives = topk_membership_loss(
                scores, target_level.target, target_level.cmask,
                k=args.topk)
            level_loss = (
                level_loss
                + args.lambda_topk * membership / topk_levels
            )
            stats["topk_total"] += membership.item() * positives
            stats["topk_rows"] += positives

        if args.lambda_boundary and nodes > args.topk:
            boundary, boundary_rows = topk_boundary_loss(
                scores, target_level.target, target_level.cmask,
                k=args.topk, margin=args.topk_margin)
            level_loss = (
                level_loss
                + args.lambda_boundary * boundary / topk_levels
            )
            stats["boundary_total"] += boundary.item() * boundary_rows
            stats["boundary_rows"] += boundary_rows

        if args.lambda_needle and nodes > needle_k:
            needle_node = ancestor_node_id(
                doc["meta"]["needle_block"],
                target_level.starts, target_level.ends)
            needle, needle_units = needle_loss_fn(
                scores, target_level.target, target_level.cmask,
                needle_node, k=needle_k, margin=args.needle_margin,
                require_teacher_topk=(
                    getattr(args, "needle_eligibility", "teacher") == "teacher"))
            level_loss = (
                level_loss
                + args.lambda_needle * needle / needle_levels
            )
            stats["needle_total"] += needle.item() * needle_units
            stats["needle_units"] += needle_units

        level_loss.backward()
        stats["levels"] += 1
        del scores, level_loss

    optimizer.step()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+",
                    help="ordinary teacher .pt paths or recursive globs")
    ap.add_argument(
        "--natural-targets", nargs="+",
        help="primary natural teacher paths/globs used once per epoch")
    ap.add_argument(
        "--replay-targets", nargs="+",
        help="old teacher paths/globs sampled as regression replay")
    ap.add_argument(
        "--natural-fraction", type=float, default=0.8,
        help="natural share of each grouped epoch; default: 0.8")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional torch RNG seed; use one fixed seed for sweeps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-topk", type=float, default=0.5,
                    help="weight of broad teacher top-k membership BCE")
    ap.add_argument(
        "--lambda-boundary", type=float, default=0.0,
        help="weight of hard teacher-top-k versus strongest-negative ranking")
    ap.add_argument("--lambda-needle", type=float, default=0.0,
                    help="weight of reader-row needle top-k retention; 0 disables it")
    ap.add_argument("--topk", type=int, default=8,
                    help="k for the membership term; match the KS1 budget")
    ap.add_argument(
        "--topk-margin", type=float, default=0.0,
        help="score margin for the hard top-k boundary loss")
    ap.add_argument("--needle-topk", type=int, default=None,
                    help="needle retention budget; defaults to --topk")
    ap.add_argument(
        "--needle-margin", type=float, default=0.0,
        help="score margin above the needle top-k threshold")
    ap.add_argument(
        "--needle-objective", choices=("group", "union"), default="group",
        help="per-group legacy loss or any-group-per-layer retrieval objective")
    ap.add_argument(
        "--needle-eligibility", choices=("teacher", "always"), default="teacher",
        help="teacher: needle losses fire only where the teacher's top-k keeps "
             "the evidence (legacy); always: the known evidence block is a "
             "hard positive for every valid reader row, even where block-pooled "
             "teacher mass misses it (LOG 2026-07-26)")
    ap.add_argument(
        "--tree-supervision", action=argparse.BooleanOptionalAction,
        default=False,
        help="train on every discriminative key-tree level (default: leaf only)")
    ap.add_argument("--tree-radix", type=int, default=2)
    ap.add_argument(
        "--tree-beam", type=int, default=8,
        help="deployed traversal beam; must match --topk for tree training")
    ap.add_argument("--proj-dim", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--offload-targets", action=argparse.BooleanOptionalAction,
        default=False,
        help="cache teacher tensors in CPU RAM and move one document at a time")
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
    grouped_inputs = bool(args.natural_targets or args.replay_targets)
    if args.targets and grouped_inputs:
        raise SystemExit(
            "--targets cannot be combined with --natural-targets/"
            "--replay-targets")
    if grouped_inputs and not (
            args.natural_targets and args.replay_targets):
        raise SystemExit(
            "grouped training requires both --natural-targets and "
            "--replay-targets")
    if not args.targets and not grouped_inputs:
        raise SystemExit(
            "provide --targets or both --natural-targets and "
            "--replay-targets")
    if not 0.0 < args.natural_fraction <= 1.0:
        raise SystemExit("--natural-fraction must be in (0, 1]")
    if args.topk < 1 or (args.needle_topk is not None and args.needle_topk < 1):
        raise SystemExit("--topk and --needle-topk must be >= 1")
    if args.tree_radix < 2 or args.tree_beam < 1:
        raise SystemExit("--tree-radix must be >= 2 and --tree-beam >= 1")
    if args.tree_supervision and args.topk != args.tree_beam:
        raise SystemExit(
            "tree-aware top-k supervision must match the deployed traversal "
            "beam: set --topk equal to --tree-beam")
    for name in ("lambda_topk", "lambda_boundary", "lambda_needle",
                 "topk_margin", "needle_margin"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    natural_paths = expand_target_paths(args.natural_targets)
    replay_paths = expand_target_paths(args.replay_targets)
    paths = (
        natural_paths + replay_paths
        if grouped_inputs else expand_target_paths(args.targets)
    )
    if not paths:
        raise SystemExit("no target files matched the supplied inputs")
    if grouped_inputs:
        if not natural_paths:
            raise SystemExit(
                f"no natural targets matched {args.natural_targets}")
        if not replay_paths:
            raise SystemExit(
                f"no replay targets matched {args.replay_targets}")
        overlap = set(natural_paths) & set(replay_paths)
        if overlap:
            raise SystemExit(
                "natural and replay pools overlap: "
                f"{sorted(overlap)[:3]}")

    data_device = "cpu" if args.offload_targets else args.device
    docs = [
        load_teacher(
            p, device=data_device,
            defer_normalization=args.offload_targets)
        for p in paths
    ]
    natural_indices = list(range(len(natural_paths)))
    replay_indices = list(range(len(natural_paths), len(paths)))
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
    print(
        f"teacher cache: device={data_device} "
        f"per_document_transfer={args.offload_targets}")
    print(
        "training objective: "
        f"{'all tree levels' if args.tree_supervision else 'leaf only'} "
        f"radix={args.tree_radix} beam={args.tree_beam} topk={args.topk} "
        f"lambda_topk={args.lambda_topk:g} "
        f"lambda_boundary={args.lambda_boundary:g} "
        f"lambda_needle={args.lambda_needle:g} "
        f"needle_objective={args.needle_objective}"
    )
    if grouped_inputs:
        preview_replay = round(
            len(natural_indices)
            * (1.0 - args.natural_fraction)
            / args.natural_fraction
        )
        preview_total = len(natural_indices) + preview_replay
        print(
            f"mixed epoch: natural={len(natural_indices)} replay_draws="
            f"{preview_replay}/{len(replay_indices)} total={preview_total} "
            f"natural_fraction={len(natural_indices) / preview_total:.3f}")

    resume_path = args.resume_out or checkpoint_path(args.out)
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_graphs")
    model_name = os.path.splitext(os.path.basename(args.out))[0]
    plot_path = args.plot or os.path.join(graph_dir, f"{model_name}.png")
    history_path = args.history or os.path.join(graph_dir, f"{model_name}.json")
    empty_history = {
        "epochs": [], "kl": [], "topk": [], "boundary": [], "needle": [],
        "eval_epochs": [],
        "eval_recall8": {}, "eval_needle8": {}, "eval_needle_union8": {},
    }
    # A fresh run must not inherit a partial history left by a prior crash.
    # Resume checkpoints carry their own matching history below.
    history = load_json(history_path, empty_history) if args.resume else empty_history
    history.setdefault("needle", [])
    history.setdefault("boundary", [])
    history.setdefault("eval_needle_union8", {})
    start_epoch = 1
    if args.resume:
        resumed_epoch, checkpoint_history = load_resume_checkpoint(
            args.resume, gate, opt, config, args.device,
            expected_train_args=training_recipe(args))
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
        boundary_tot, boundary_rows = 0.0, 0
        needle_tot, needle_units = 0.0, 0
        document_order = (
            mixed_epoch_order(
                natural_indices,
                replay_indices,
                args.natural_fraction,
                shuffle=args.shuffle_targets,
            )
            if grouped_inputs
            else (
                torch.randperm(len(docs)).tolist()
                if args.shuffle_targets else range(len(docs))
            )
        )
        for document_index in document_order:
            stored_doc = docs[document_index]
            doc = move_document(stored_doc, args.device)
            step = train_document(gate, opt, doc, args)
            tot += step["kl_total"]
            nrows += step["kl_rows"]
            topk_tot += step["topk_total"]
            topk_rows += step["topk_rows"]
            boundary_tot += step["boundary_total"]
            boundary_rows += step["boundary_rows"]
            needle_tot += step["needle_total"]
            needle_units += step["needle_units"]
            if doc is not stored_doc:
                del doc

        kl_avg = tot / max(nrows, 1)
        topk_avg = topk_tot / max(topk_rows, 1) if args.lambda_topk else None
        boundary_avg = (
            boundary_tot / max(boundary_rows, 1)
            if args.lambda_boundary else None)
        needle_avg = (
            needle_tot / max(needle_units, 1)
            if args.lambda_needle else None)
        history["epochs"].append(epoch)
        history["kl"].append(kl_avg)
        history["topk"].append(topk_avg)
        history["boundary"].append(boundary_avg)
        history["needle"].append(needle_avg)
        if progress is not None:
            postfix = {"KL": f"{kl_avg:.4f}"}
            if topk_avg is not None:
                postfix["topk"] = f"{topk_avg:.4f}"
            if boundary_avg is not None:
                postfix["boundary"] = f"{boundary_avg:.4f}"
            if needle_avg is not None:
                postfix["needle"] = f"{needle_avg:.4f}"
            progress.set_postfix(postfix)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            gate.eval()
            line = [f"epoch {epoch:>4}  KL={kl_avg:.4f}"]
            if args.lambda_topk:
                line.append(f"topk={topk_avg:.4f}")
            if args.lambda_boundary:
                line.append(f"boundary={boundary_avg:.4f}")
            if args.lambda_needle:
                line.append(f"needle={needle_avg:.4f}")
            history["eval_epochs"].append(epoch)
            buckets = {}
            with torch.no_grad():
                for stored_doc in docs:
                    doc = move_document(stored_doc, args.device)
                    sc = gate(doc["q_feat"], doc["k_feat"])
                    met = recall_metrics(sc, doc["target"], doc["cmask"],
                                         tuple(args.budgets), doc["meta"]["needle_block"])
                    r8 = met.get("recall@8", float("nan"))
                    nh = met.get("needle_hit@8", None)
                    nu = met.get("needle_union@8", None)
                    tu = met.get("teacher_needle_union@8", None)
                    requested = doc["meta"].get(
                        "requested_length", doc["meta"]["seq_len"])
                    tag = f"len{requested}"
                    bucket = buckets.setdefault(
                        tag, {"recall": [], "needle": [], "union": [],
                              "teacher_union": []})
                    bucket["recall"].append(r8)
                    if nh is not None:
                        bucket["needle"].append(nh)
                    if nu is not None:
                        bucket["union"].append(nu)
                    if tu is not None:
                        bucket["teacher_union"].append(tu)
                    if doc is not stored_doc:
                        del doc
            for tag in sorted(buckets):
                recall_values = buckets[tag]["recall"]
                needle_values = buckets[tag]["needle"]
                union_values = buckets[tag]["union"]
                teacher_union_values = buckets[tag]["teacher_union"]
                mean_recall = sum(recall_values) / len(recall_values)
                mean_needle = (
                    sum(needle_values) / len(needle_values)
                    if needle_values else None
                )
                mean_union = (
                    sum(union_values) / len(union_values)
                    if union_values else None
                )
                mean_teacher_union = (
                    sum(teacher_union_values) / len(teacher_union_values)
                    if teacher_union_values else None
                )
                history["eval_recall8"].setdefault(
                    tag, []).append(mean_recall)
                history["eval_needle8"].setdefault(
                    tag, []).append(mean_needle)
                history["eval_needle_union8"].setdefault(
                    tag, []).append(mean_union)
                line.append(
                    f"{tag} n={len(recall_values)} r@8={mean_recall:.3f}"
                    + (
                        f" ndl@8={mean_needle:.2f}"
                        if mean_needle is not None else ""
                    )
                    + (
                        f" ndl_u@8={mean_union:.2f}"
                        if mean_union is not None else ""
                    )
                    + (
                        f" tchr_u@8={mean_teacher_union:.2f}"
                        if mean_teacher_union is not None else ""
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

    save_gate(args.out, gate, config, train_args=training_recipe(args))
    print(f"saved gate -> {args.out}")
    print(f"training history -> {history_path}")
    if os.path.exists(plot_path):
        print(f"training graph -> {plot_path}")


if __name__ == "__main__":
    main()



