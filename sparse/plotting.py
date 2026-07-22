"""Visual diagnostics for sparse-prefill replay."""
import os

import torch


def selected_block_matrix(selected_blocks, layer, kv_group):
    """Convert one selected_blocks routing slice into [query_block, key_block]."""
    if selected_blocks.dim() != 5 or selected_blocks.shape[0] != 1:
        raise ValueError("plotting currently expects selected_blocks with batch size 1")
    _, layers, groups, qblocks, _ = selected_blocks.shape
    if not -layers <= layer < layers or not -groups <= kv_group < groups:
        raise ValueError(f"layer/group out of range: layer={layer}, group={kv_group}")
    routing = selected_blocks[0, layer, kv_group].cpu()
    matrix = torch.zeros((qblocks, qblocks), dtype=torch.float32)
    for query_block, row in enumerate(routing):
        ids = row[row >= 0].to(torch.long)
        matrix[query_block, ids] = 1.0
    return matrix


def _token_label(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text if text.strip() else repr(text)


def save_sparse_replay_plot(logits, tokenizer, selected_blocks, meta, path,
                            layer=0, kv_group=0, top_k=15):
    """Save routing heatmap plus final-position next-token probability chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib") from error

    routing = selected_block_matrix(selected_blocks, layer, kv_group)
    flat_logits = logits.detach().float().cpu().reshape(-1)
    probabilities = torch.softmax(flat_logits, dim=0)
    count = min(int(top_k), probabilities.numel())
    probs, ids = probabilities.topk(count)

    fig, (routing_ax, tokens_ax) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    routing_ax.imshow(routing.numpy(), cmap="magma", interpolation="nearest", aspect="auto", origin="lower")
    routing_ax.set(
        title=f"Selected blocks: layer {layer}, KV group {kv_group}",
        xlabel="Key block", ylabel="Query block",
    )
    needle_block = int(meta.get("needle_block", -1))
    if needle_block >= 0:
        routing_ax.axvline(needle_block, color="cyan", linestyle="--", linewidth=1, label="needle block")
        routing_ax.legend(loc="upper left", fontsize=8)

    labels = [_token_label(tokenizer, token_id) for token_id in ids]
    tokens_ax.barh(range(count), probs.flip(0).numpy(), color="tab:blue")
    tokens_ax.set_yticks(range(count), labels=list(reversed(labels)), fontsize=8)
    tokens_ax.set(title="Sparse replay: top next-token probabilities", xlabel="Probability")
    tokens_ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"SPRUCE sparse replay — {meta.get('case_id', 'unknown case')} "
        f"(depth={meta.get('depth', 'unknown')})"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
