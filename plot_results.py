"""
Plots N-Queens solver performance across versions.
Add new entries to RESULTS as versions complete.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# (version_label, {n: elapsed_seconds})
RESULTS = [
    ("v1 baseline", {
        8: 0.003, 9: 0.014, 10: 0.067, 11: 0.369, 12: 2.188, 13: 13.728, 14: 95.367,
    }),
    ("v2 sets O(1)", {
        8: 0.001, 9: 0.004, 10: 0.017, 11: 0.082, 12: 0.452, 13: 2.577, 14: 16.950, 15: 107.177,
    }),
    ("v3 + symmetry", {
        8: 0.001, 9: 0.003, 10: 0.011, 11: 0.049, 12: 0.226, 13: 1.392, 14: 7.957, 15: 58.246,
    }),
    ("v4 bitmasks", {
        9: 0.001, 10: 0.003, 11: 0.015, 12: 0.060, 13: 0.359, 14: 1.934, 15: 13.054, 16: 81.622,
    }),
    ("v5 iterative stack", {
        9: 0.001, 10: 0.004, 11: 0.020, 12: 0.079, 13: 0.470, 14: 2.544, 15: 17.344, 16: 107.974,
    }),
    ("v6 Numba JIT", {
        12: 0.002, 13: 0.011, 14: 0.059, 15: 0.399, 16: 2.488, 17: 18.816, 18: 135.263,
    }),
    ("v7 multiprocessing", {
        13: 0.879, 14: 1.008, 15: 1.124, 16: 1.515, 17: 6.443, 18: 22.633, 19: 288.433,
    }),
    ("v9 prange+unit prop", {
        13: 0.002, 14: 0.009, 15: 0.059, 16: 0.355, 17: 2.601, 18: 18.377, 19: 149.250,
    }),
    ("v10 MRV (negative)", {
        13: 0.014, 14: 0.067, 15: 0.401, 16: 2.424, 17: 17.690, 18: 119.031,
    }),
    ("v11 clean bitmask", {
        13: 0.002, 14: 0.009, 15: 0.059, 16: 0.366, 17: 2.781, 18: 20.374,
    }),
]

PALETTE = [
    "#e63946",  # v1 — red
    "#f4a261",  # v2 — orange
    "#2a9d8f",  # v3 — teal
    "#457b9d",  # v4 — steel blue
    "#a8dadc",  # v5 — ice
    "#c77dff",  # v6 — violet
    "#ffbe0b",  # v7 — amber
    "#fb5607",  # v8 — deep orange
    "#8ecae6",  # v9 — sky
    "#06d6a0",  # v10 — mint
    "#118ab2",  # v11 — deep blue
]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

for i, (label, data) in enumerate(RESULTS):
    ns = sorted(data)
    ts = [data[n] for n in ns]
    color = PALETTE[i % len(PALETTE)]
    ax.plot(ns, ts, "o-", color=color, linewidth=2, markersize=6, label=label, zorder=3)

# 60-second deadline line
ax.axhline(60, color="#ffffff", linewidth=1, linestyle="--", alpha=0.4, label="60s deadline")

ax.set_yscale("log")
ax.set_xlabel("N (board size)", color="#cccccc", fontsize=13)
ax.set_ylabel("Runtime (seconds, log scale)", color="#cccccc", fontsize=13)
ax.set_title("N-Queens Solver: Optimization Progress", color="#ffffff", fontsize=15, fontweight="bold")

ax.tick_params(colors="#aaaaaa")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(True, which="major", color="#222222", linewidth=0.8)
ax.grid(True, which="minor", color="#1a1a1a", linewidth=0.4)

legend = ax.legend(
    facecolor="#1a1a1a", edgecolor="#444444",
    labelcolor="#dddddd", fontsize=11,
    loc="upper left",
)

plt.tight_layout()
plt.savefig("results.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved results.png")
