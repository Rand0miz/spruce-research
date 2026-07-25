"""Optional matplotlib helpers used by selector training and evaluation CLIs."""
import json
import os


def write_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def load_json(path, default):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pyplot_or_none():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _aligned_eval_series(eval_epochs, values):
    """Collapse legacy per-document values into one mean per eval epoch."""
    epochs = list(eval_epochs)
    values = list(values)
    if len(values) == len(epochs):
        return epochs, values
    if epochs and len(values) % len(epochs) == 0:
        per_epoch = len(values) // len(epochs)
        means = [
            sum(values[start:start + per_epoch]) / per_epoch
            for start in range(0, len(values), per_epoch)
        ]
        return epochs, means
    count = min(len(epochs), len(values))
    return epochs[-count:], values[-count:]


def save_training_plot(history, path):
    plt = pyplot_or_none()
    if plt is None:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    epochs = history.get("epochs", [])
    axes[0, 0].plot(epochs, history.get("kl", []))
    axes[0, 0].set(title="Training KL", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].grid(alpha=0.25)
    topk = history.get("topk", [])
    needle = history.get("needle", [])
    if any(v is not None for v in topk):
        axes[0, 1].plot(epochs, [float("nan") if v is None else v for v in topk],
                         color="tab:orange", label="generic top-k")
    if any(v is not None for v in needle):
        axes[0, 1].plot(epochs, [float("nan") if v is None else v for v in needle],
                         color="tab:purple", label="needle top-k")
    if any(v is not None for v in topk) or any(v is not None for v in needle):
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set(title="Retention Losses", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].grid(alpha=0.25)
    for name, values in history.get("eval_recall8", {}).items():
        x_values, y_values = _aligned_eval_series(
            history.get("eval_epochs", []), values)
        axes[1, 0].plot(x_values, y_values, marker="o", label=name)
    axes[1, 0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1, label="0.95 bar")
    axes[1, 0].set(title="Recall@8 During Training", xlabel="Epoch", ylabel="Recall", ylim=(0, 1.02))
    axes[1, 0].grid(alpha=0.25)
    if history.get("eval_recall8"):
        axes[1, 0].legend(fontsize=8)
    for name, values in history.get("eval_needle8", {}).items():
        x_values, y_values = _aligned_eval_series(
            history.get("eval_epochs", []), values)
        axes[1, 1].plot(x_values, y_values, marker="o", label=name)
    axes[1, 1].set(title="Needle Hit@8 During Training", xlabel="Epoch", ylabel="Hit rate", ylim=(0, 1.02))
    axes[1, 1].grid(alpha=0.25)
    if history.get("eval_needle8"):
        axes[1, 1].legend(fontsize=8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def save_flat_eval_plot(results, budgets, path, title):
    plt = pyplot_or_none()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for name, met in results.items():
        axes[0].plot(budgets, [met[f"recall@{b}"] for b in budgets], marker="o", label=name)
        axes[1].plot(budgets, [met[f"coverage@{b}"] for b in budgets], marker="o", label=name)
        if f"needle_hit@{budgets[0]}" in met:
            axes[2].plot(budgets, [met.get(f"needle_hit@{b}", float("nan")) for b in budgets], marker="o", label=name)
    axes[0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set(title="Recall", xlabel="Block budget", ylabel="Recall", ylim=(0, 1.02))
    axes[1].set(title="Teacher mass coverage", xlabel="Block budget", ylabel="Coverage", ylim=(0, 1.02))
    axes[2].set(title="Needle retrieval", xlabel="Block budget", ylabel="Hit rate", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def save_tree_plot(results, x_values, budgets, path, title, x_label):
    plt = pyplot_or_none()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for budget in budgets:
        axes[0].plot(x_values, [met[f"recall@{budget}"] for met in results], marker="o", label=f"recall@{budget}")
    axes[1].plot(x_values, [met["coverage"] if "coverage" in met else met["coverage@8"] for met in results], marker="o", label="selected coverage")
    oracle_key = "oracle_coverage" if "oracle_coverage" in results[0] else "oracle_cov@8"
    axes[1].plot(x_values, [met[oracle_key] for met in results], marker="o", linestyle="--", label="oracle coverage")
    if "needle_hit" in results[0]:
        axes[2].plot(x_values, [met.get("needle_hit", float("nan")) for met in results], marker="o", label="needle hit")
    else:
        axes[2].plot(x_values, [met.get("needle_hit@8", float("nan")) for met in results], marker="o", label="needle@8")
    axes[0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set(title="Recall", xlabel=x_label, ylabel="Recall", ylim=(0, 1.02))
    axes[1].set(title="Coverage", xlabel=x_label, ylabel="Teacher mass", ylim=(0, 1.02))
    axes[2].set(title="Needle retrieval", xlabel=x_label, ylabel="Hit rate", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True

