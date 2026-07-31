import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/paper/figures
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))    # <repo>
T = os.path.join(_ROOT, "benchmarks", "outputs",
                 "natural_yarn_beam16_full_results",
                 "paper_artifacts", "tables") + os.sep
OUT = _HERE + os.sep

# IEEE print geometry: textwidth 7.16in, columnwidth 3.40in.
# Render at final size so nothing is scaled -> fonts land at their stated pt.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 220,
})

DENSE, COMP, ACC = "#2b2b2b", "#009e73", "#5b4bd6"

rows = list(csv.DictReader(open(T + "by_length.csv")))
L = [int(r["requested_length"]) // 1024 for r in rows]
x = np.arange(len(L))
f = lambda k: [float(r[k]) for r in rows]

def style(ax, ylab, xlab="Requested context (Ki tokens)"):
    ax.set_xticks(x); ax.set_xticklabels(L)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, linewidth=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------- Fig 1: fully charged latency + speedup, full text width ----------
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.10))

a = axs[0]
a.plot(x, f("median_dense_request_seconds"), "o-", color=DENSE, label="Dense")
a.plot(x, f("median_compiled_request_seconds"), "o-", color=COMP, label="SPRUCE")
a.set_ylim(0, 11.5); a.legend(frameon=False, loc="upper left")
style(a, "Median request time (s)")
a.set_title("(a) Fully charged request latency")

a = axs[1]
a.plot(x, f("sum_weighted_speedup"), "o-", color=ACC)
a.axhline(1.0, ls="--", lw=0.7, color="0.4")
a.text(0.12, 1.4, "parity", fontsize=6.8, color="0.35")
a.set_ylim(0, 15.5)
style(a, r"Speedup ($\times$)")
a.set_title("(b) Sum-weighted dense / SPRUCE speedup")

fig.tight_layout(pad=0.35, w_pad=1.6)
fig.savefig(OUT + "fig_latency_speedup.pdf")
fig.savefig(OUT + "fig_latency_speedup.png", dpi=300)
plt.close(fig)

# ---------- Fig 2: length x depth heatmaps, full text width ----------
ld = list(csv.DictReader(open(T + "by_length_depth.csv")))
depths = sorted({float(r["depth"]) for r in ld}, reverse=True)
D = {(int(r["requested_length"]) // 1024, float(r["depth"])): r for r in ld}
grid = lambda k: np.array([[float(D[(l, d)][k]) for l in L] for d in depths])

dn, cp = grid("dense_exact_rate"), grid("compiled_exact_rate")
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.74))
for ax, M, cmap, ttl, vlim in [
    (axs[0], dn, "Greys", "(a) Dense exact", (0, 1)),
    (axs[1], cp, "Blues", "(b) SPRUCE exact", (0, 1)),
    (axs[2], cp - dn, "RdYlGn", "(c) SPRUCE $-$ dense", (-1, 1)),
]:
    im = ax.imshow(M, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect="auto")
    ax.set_xticks(x); ax.set_xticklabels(L)
    ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d:.1f}" for d in depths])
    ax.set_xlabel("Context (Ki tokens)")
    if ax is axs[0]: ax.set_ylabel("Evidence depth")
    ax.set_title(ttl)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            lo, hi = vlim
            dark = (v - lo) / (hi - lo) > 0.6 if cmap != "RdYlGn" else False
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.2, color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.5)

fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(OUT + "fig_heatmap.pdf"); plt.close(fig)

# ---------- Fig 3: allocated memory, one column ----------
fig, ax = plt.subplots(figsize=(3.40, 1.86))
ax.plot(x, f("median_dense_peak_memory_allocated_gb"), "o-", color=DENSE, label="Dense")
ax.plot(x, f("median_compiled_peak_memory_allocated_gb"), "o-", color=COMP, label="SPRUCE")
ax.set_ylim(0, 17); ax.legend(frameon=False, loc="upper left")
style(ax, "Peak allocated memory (GB)")
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig_memory.pdf"); plt.close(fig)

# ---------- Fig 4: evidence recall, one column ----------
fig, ax = plt.subplots(figsize=(3.40, 1.86))
ax.plot(x, f("direct_evidence_recall"), "o-", color="#d55e00", label="Direct $M{=}4$")
ax.plot(x, f("expanded_evidence_recall"), "o-", color="#0072b2", label="Radius-1 expanded")
ax.set_ylim(0.6, 1.02); ax.legend(frameon=False, loc="lower left")
style(ax, "Evidence recall")
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig_recall.pdf"); plt.close(fig)
print("ok")

# ---------- Fig 4: compiled request components, one column ----------
import json, io
_REP = os.path.join(_ROOT, "benchmarks", "outputs",
                    "natural_yarn_beam16_full_results", "paper_artifacts",
                    "combined_report.json")
cases = json.load(io.open(_REP, encoding="utf-8"))["cases"]

GROUPS = [
    ("Offset tokenization ", ["layout_tokenize_seconds"], "#3b2a6b"),
    ("Index + traversal",   ["index_seconds", "selection_seconds"], "#3f6fa8"),
    ("Span stitching",      ["compile_seconds"], "#2a9d8f"),
    ("Compact tok. + transfer",
     ["compact_tokenize_seconds", "input_transfer_seconds"], "#6fc08a"),
    ("Compact prefill",     ["prefill_seconds"], "#a8d84f"),
    ("Decode",              ["decode_seconds"], "#d9e021"),
]

means = {g[0]: [] for g in GROUPS}
for l in L:
    sel = [c["compiled"] for c in cases
           if int(c["requested_length"]) // 1024 == l]
    for name, keys, _ in GROUPS:
        vals = [sum(float(c.get(k, 0.0)) for k in keys) for c in sel]
        means[name].append(sum(vals) / len(vals))

fig, ax = plt.subplots(figsize=(3.40, 1.78))
bottom = np.zeros(len(L))
for name, _, colr in GROUPS:
    v = np.array(means[name])
    ax.bar(x, v, 0.72, bottom=bottom, color=colr, label=name,
           edgecolor="white", linewidth=0.3)
    bottom += v
style(ax, "Mean seconds")
ax.legend(fontsize=6.0, ncol=3, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.28), handlelength=0.9,
          columnspacing=0.7, handletextpad=0.35, borderaxespad=0.0)
fig.savefig(OUT + "fig_compiled_latency.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT + "fig_compiled_latency.png", dpi=300,
            bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print("components ok")
