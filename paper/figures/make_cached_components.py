"""Build the paper's cached-request component figure from the L4 run."""
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPORTS = (
    HERE.parent.parent
    / "benchmarks" / "outputs" / "cached_index_v2" / "length_reports"
)
OUTPUT = HERE / "fig_cached_components.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 220,
})

reports = sorted(
    (
        json.loads(path.read_text(encoding="utf-8"))
        for path in REPORTS.glob("length_*.json")
    ),
    key=lambda report: int(report["requested_length"]),
)
if len(reports) != 8:
    raise RuntimeError(f"expected 8 cached-index length reports, found {len(reports)}")

lengths = [int(report["requested_length"]) // 1024 for report in reports]
positions = list(range(len(lengths)))
components = (
    ("selection_seconds", "traversal", "#159D82"),
    ("compile_seconds", "stitch", "#1D8EA6"),
    ("compact_tokenize_seconds", "compact tokenize", "#287BA8"),
    ("prefill_seconds", "prefill", "#5B4BD6"),
    ("decode_seconds", "decode", "#9B70CF"),
)

cold_request_ms = [
    1000.0 * float(case["cold"]["request_seconds"])
    for report in reports
    for case in report["cases"]
]
cold_average_ms = statistics.mean(cold_request_ms)

fig, axis = plt.subplots(figsize=(3.40, 2.05))
bottom = [0.0] * len(reports)
for field, label, color in components:
    values = [
        1000.0 * statistics.median(
            float(case["cached"][field]) for case in report["cases"])
        for report in reports
    ]
    axis.bar(
        positions, values, width=0.62, bottom=bottom,
        label=label, color=color)
    bottom = [base + value for base, value in zip(bottom, values)]

axis.axhline(
    cold_average_ms, color="#555555", linewidth=1.0,
    linestyle=":", label=f"cold average ({cold_average_ms:.0f} ms)")
axis.set(
    title="Cached request components",
    xlabel="Requested context (Ki tokens)",
    ylabel="Request time (ms)",
    ylim=(0, 800),
    yticks=(0, 200, 400, 600, 800),
    xticks=positions,
    xticklabels=lengths,
)
axis.set_axisbelow(True)
axis.grid(axis="y", alpha=0.3, linewidth=0.4)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(OUTPUT)
plt.close(fig)

print(f"wrote {OUTPUT}")
print(f"cold average: {cold_average_ms:.3f} ms")
