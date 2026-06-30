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

# Numbers from docs/BENCHMARKS_SIZE_NL.md (forward measured under
# torch.no_grad for a fair forward-vs-forward comparison).
residues = [75, 100, 150, 200, 300, 400, 500, 600]
atoms    = [1116, 1591, 2465, 3039, 4466, 5899, 7561, 8800]
jtmol_e  = [0.236, 0.474, 0.749, 0.712, 0.896, 1.004, 1.143, 1.411]
up_e     = [30.004, 29.456, 29.712, 27.559, 28.071, 28.939, 28.231, 28.905]
jtmol_g  = [1.159, 1.943, 3.784, 4.462, 6.542, 9.448, 11.448, 13.957]
up_g     = [32.397, 32.873, 32.241, 31.880, 32.711, 35.392, 32.607, 34.432]

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
