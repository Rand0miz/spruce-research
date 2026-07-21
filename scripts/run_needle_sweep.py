"""Run a controlled lambda-needle sweep, then evaluate every resulting gate.

Example:
  python -m scripts.run_needle_sweep \
      --targets "teacher_targets/train/teacher_*.pt" \
      --heldout-targets "teacher_targets/heldout/teacher_*.pt"

Evaluation failures are recorded but do not stop the remaining sweep members;
eval_gate uses a non-zero exit code when recall@8 is below its pass bar.
"""
import argparse
import json
import os
import subprocess
import sys


def run(command, check):
    print("\n+", " ".join(command), flush=True)
    return subprocess.run(command, check=check).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="training teacher-target paths or globs")
    ap.add_argument("--heldout-targets", nargs="+", required=True,
                    help="held-out teacher-target paths or globs")
    ap.add_argument("--out-dir", default="selector_ckpt")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-topk", type=float, default=0.75)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for weight in args.lambdas:
        tag = f"lamneedle{weight:g}".replace(".", "p")
        gate = os.path.join(
            args.out_dir,
            f"flat_gate_lamtopk{args.lambda_topk:g}_{tag}_lr{args.lr:g}_e{args.epochs}.pt",
        )
        train = [
            sys.executable, "-m", "selector.train", "--targets", *args.targets,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--lambda-topk", str(args.lambda_topk),
            "--lambda-needle", str(weight), "--topk", str(args.topk),
            "--eval-every", str(args.eval_every), "--seed", str(args.seed),
            "--progress", "off", "--out", gate,
        ]
        if args.device:
            train.extend(["--device", args.device])
        run(train, check=True)

        evaluate = [
            sys.executable, "-m", "scripts.eval_gate", "--gate", gate,
            "--targets", *args.heldout_targets,
        ]
        if args.device:
            evaluate.extend(["--device", args.device])
        results.append({"lambda_needle": weight, "gate": gate,
                        "eval_exit_code": run(evaluate, check=False)})

    path = os.path.join(args.out_dir, "needle_sweep_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "runs": results}, f, indent=2)
    print(f"\nsweep summary -> {path}")


if __name__ == "__main__":
    main()
