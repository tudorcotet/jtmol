"""Generate the headline jtmol-vs-upstream-tmol speedup figure.

Reads numbers from ``docs/BENCHMARKS_SIZE_NL.md`` (full neighbor-list
sweep on Modal A100-40GB). Writes ``docs/figures/jtmol_vs_tmol.png``.

Style: seaborn whitegrid, sentence-case labels.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# Numbers from docs/BENCHMARKS_SIZE_NL.md.
residues = [75, 100, 150, 200, 300, 400, 500, 600]
atoms    = [1116, 1591, 2465, 3039, 4466, 5899, 7561, 8800]
jtmol_e  = [0.254, 0.467, 0.923, 1.001, 0.853, 0.992, 1.147, 1.410]
up_e     = [28.314, 30.882, 38.035, 28.899, 29.931, 30.562, 29.287, 28.788]
jtmol_g  = [1.166, 1.949, 3.778, 4.467, 6.532, 9.449, 11.391, 13.907]
up_g     = [31.520, 35.686, 33.847, 34.166, 40.151, 33.497, 30.852, 32.521]

speedup_e = [u / j for u, j in zip(up_e, jtmol_e)]
speedup_g = [u / j for u, j in zip(up_g, jtmol_g)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Left: per-pose latency (log)
ax = axes[0]
ax.plot(residues, jtmol_e, marker="o", linewidth=2.2, label="jtmol — forward")
ax.plot(residues, up_e, marker="o", linewidth=2.2, label="upstream tmol — forward")
ax.plot(residues, jtmol_g, marker="s", linewidth=2.2, linestyle="--", label="jtmol — gradient")
ax.plot(residues, up_g, marker="s", linewidth=2.2, linestyle="--", label="upstream tmol — gradient")
ax.set_yscale("log")
ax.set_xlabel("protein size (residues)")
ax.set_ylabel("per-pose latency (ms, log scale)")
ax.set_title("scoring latency vs protein size")
ax.legend(loc="lower right", fontsize=11, frameon=True)
ax.grid(True, which="both", alpha=0.3)

# Right: speedup factor
ax = axes[1]
import numpy as np
x = np.arange(len(residues))
width = 0.38
ax.bar(x - width / 2, speedup_e, width, label="forward speedup", color=sns.color_palette()[0])
ax.bar(x + width / 2, speedup_g, width, label="gradient speedup", color=sns.color_palette()[2])
for i, v in enumerate(speedup_e):
    ax.text(i - width / 2, v + 2, f"{v:.0f}×", ha="center", fontsize=10, color=sns.color_palette()[0])
for i, v in enumerate(speedup_g):
    ax.text(i + width / 2, v + 2, f"{v:.0f}×", ha="center", fontsize=10, color=sns.color_palette()[2])
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in residues])
ax.set_xlabel("protein size (residues)")
ax.set_ylabel("speedup factor (jtmol vs upstream)")
ax.set_title("jtmol speedup vs upstream tmol")
ax.legend(loc="upper right", fontsize=11, frameon=True)
ax.set_ylim(0, max(speedup_e) * 1.2)

fig.suptitle(
    "jtmol vs upstream tmol — A100-40GB, beta_nov2016_cart, 22 channels, K=128 per call",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

out_dir = pathlib.Path(__file__).resolve().parent
out_path = out_dir / "jtmol_vs_tmol.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"wrote {out_path}")
