"""Task 1.3 - Visual EDA for VisionGuard.

Reads reports/audit.csv (made by src/audit.py) and saves:
  reports/figures/fig_samples.png     - 8 random images per class
  reports/figures/fig_histograms.png  - width, height, file size per class
Run from the project root:  python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from PIL import Image

AUDIT = Path("reports/audit.csv")
FIG = Path("reports/figures")
SEED = 42                                  # same random sample every run
N = 8                                      # images per class in the grid
COLORS = {"AiArtData": "#2a78d6", "RealArt": "#eb6834"}   # colour-blind-safe pair

df = pd.read_csv(AUDIT)
df = df[~df.corrupt]
classes = sorted(df.cls.unique())
FIG.mkdir(parents=True, exist_ok=True)

# ---------- Figure 1: sample grid ----------
fig, axes = plt.subplots(len(classes), N, figsize=(2.2 * N, 2.7 * len(classes)))
for r, c in enumerate(classes):
    sample = df[df.cls == c].sample(N, random_state=SEED)
    for ax, (_, row) in zip(axes[r], sample.iterrows()):
        with Image.open(row.path) as im:
            im = im.convert("RGB")        # same conversion the model will use
            im.thumbnail((256, 256))
            ax.imshow(im)
        ax.set_title(f"{int(row.width)}x{int(row.height)}  {row.format}", fontsize=8)
        ax.axis("off")
    axes[r][0].text(-0.1, 0.5, c, transform=axes[r][0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=13, weight="bold")
fig.suptitle(f"Random samples per class (seed={SEED})", fontsize=14)
plt.tight_layout()
fig.savefig(FIG / "fig_samples.png", dpi=150)
plt.close(fig)

# ---------- Figure 2: histograms per class ----------
cols = [("width", "Width (px)"), ("height", "Height (px)"), ("size_kb", "File size (KB)")]
df["size_kb"] = df.size_bytes / 1024
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (col, label) in zip(axes, cols):
    bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 35)
    for c in classes:
        x = df.loc[df.cls == c, col]
        w = np.full(len(x), 100 / len(x))  # y = % of that class, fair despite 539 vs 436
        ax.hist(x, bins=bins, weights=w, histtype="step", linewidth=2,
                color=COLORS[c], label=f"{c} (n={len(x)}, median={x.median():.0f})")
        ax.axvline(x.median(), color=COLORS[c], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 5000)
             if df[col].min() <= t <= df[col].max()]
    ax.set_xticks(ticks)                                   # plain readable ticks
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter()) 
    ax.set_xlabel(label + "  [log scale]")
    ax.set_ylabel("% of images in class")
    ax.set_title(f"{label} per class")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

plt.tight_layout()
fig.savefig(FIG / "fig_histograms.png", dpi=150)
plt.close(fig)
print("Saved:", *sorted(p.as_posix() for p in FIG.glob("*.png")), sep="\n  ")
