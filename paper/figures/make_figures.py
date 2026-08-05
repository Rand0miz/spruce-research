import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# D=1024 / NVIDIA L4 run -- the authoritative paper source
T = os.path.join(_ROOT, "benchmarks", "outputs",
                 "natural_yarn_beam16_paper_v2",
                 "paper_artifacts", "tables") + os.sep
OUT = _HERE + os.sep

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.6,
    "grid.linewidth": 0.4, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 220,
})
DENSE, COMP = "#2b2b2b", "#009e73"

rows = list(csv.DictReader(open(T + "by_length.csv")))
L = [int(r["requested_length"]) // 1024 for r in rows]
x = np.arange(len(L))
f = lambda k: [float(r[k]) for r in rows]

def style(ax, ylab, xlab="Requested context (Ki tokens)"):
    ax.set_xticks(x); ax.set_xticklabels(L)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, linewidth=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------- cached-index run (D=1024, L4) ----------
import json
CI = os.path.join(_ROOT, "benchmarks", "outputs", "cached_index_v2")
ci = json.load(open(os.path.join(CI, "summary.json")))
def ci_len(arm, key):
    return [ci["by_length"][str(l * 1024)][arm][key] for l in L]

# ---------- Fig 1: fully charged request latency, dense / cold / cached ----------
fig, ax = plt.subplots(figsize=(3.40, 2.10))
ax.plot(x, f("median_dense_request_seconds"), "o-", color=DENSE, label="Dense")
ax.plot(x, f("median_compiled_request_seconds"), "o-", color=COMP, label="SPRUCE (cold)")
ax.plot(x, ci_len("cached", "median_request_seconds"), "s--", color="#0072b2",
        label="SPRUCE (reused index)")
ax.set_yscale("log")
ax.legend(
    frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
    ncol=2, fontsize=6.5, columnspacing=0.9, handletextpad=0.45)
style(ax, "Median request time (s, log)")
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig_latency.pdf"); plt.close(fig)

# ---------- Fig 2: length x depth accuracy heatmaps ----------
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
            v = M[i, j]; lo, hi = vlim
            dark = (v - lo) / (hi - lo) > 0.6 if cmap != "RdYlGn" else False
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.2, color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.5)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(OUT + "fig_heatmap.pdf"); plt.close(fig)

# ---------- Fig 3: traversal cost vs tree depth (the logarithmic term) ----------
import numpy as _np
lev = [ci["by_length"][str(l * 1024)]["build"]["median_tree_levels"] for l in L]
trav = ci_len("cold", "median_selection_ms")
fig, ax = plt.subplots(figsize=(3.40, 1.95))
ax.plot(lev, trav, "o", color="#0072b2")
m, b = _np.polyfit(lev, trav, 1)
xs = _np.linspace(min(lev) - 0.3, max(lev) + 0.3, 10)
ax.plot(xs, m * xs + b, "-", color="#0072b2", lw=1.0)
ax.set_xticks(sorted(set(lev)))
ax.set_xlabel("Tree depth (levels)"); ax.set_ylabel("Traversal (ms)")
ax.grid(alpha=0.3, linewidth=0.4)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.annotate(f"{m:+.3f} ms / level", xy=(0.05, 0.86), xycoords="axes fraction", fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig_traversal.pdf"); plt.close(fig)

# ---------- Fig 4: cold request components (what reuse later removes) ----------
import glob
COMPS = [("Offset tokenization", "layout_tokenize_seconds", "#4c2a85"),
         ("Index construction",  "index_seconds",           "#6a51a3"),
         ("Traversal",           "selection_seconds",       "#126b4f"),
         ("Stitch + tokenize",   None,                       "#55a868"),
         ("Compact prefill",     "prefill_seconds",         "#245b91"),
         ("Decode",              "decode_seconds",          "#4c8fc4")]
med = {c[0]: [] for c in COMPS}
for l in L:
    cases = json.load(open(os.path.join(CI, "length_reports", f"length_{l*1024}.json")))["cases"]
    cold = [c["cold"] for c in cases]
    for name, key, _ in COMPS:
        if key:
            vals = [float(c[key]) for c in cold]
        else:
            vals = [float(c["compile_seconds"]) + float(c["compact_tokenize_seconds"])
                    + float(c["input_transfer_seconds"]) for c in cold]
        med[name].append(1000.0 * (sorted(vals)[len(vals)//2]))
fig, ax = plt.subplots(figsize=(3.40, 2.05))
bottom = _np.zeros(len(L))
for name, _, colr in COMPS:
    v = _np.array(med[name])
    ax.bar(x, v, 0.72, bottom=bottom, color=colr, label=name,
           edgecolor="white", linewidth=0.3)
    bottom += v
style(ax, "Median cold request (ms)")
ax.legend(fontsize=5.8, ncol=2, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.30), handlelength=1.0,
          columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
fig.savefig(OUT + "fig_cold_components.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print("regenerated from D=1024 / L4")
