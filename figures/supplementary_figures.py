"""Supplementary figures S1-S4 (linked from the poster QR code).

S1  ROI-to-ROI leakage (dSPM point spread, median over subjects) -- input: npz from
    analyses/source_leakage_crosstalk.py
S2  Low-level visual properties of the one-word stimuli, Word vs Non-word -- input: csv written by
    analyses/stimulus_visual_properties.py (--csv)
S3  Word > Non-word effect (dz) of the tag (FT envelope) and of the evoked response in the three a-priori
    cascade windows (80-130, 150-225, 250-600 ms) -- input: results/block4_cascade_windows.json
S4  Block-3 gradient: log ratio vs raw difference, and whole epoch vs pre-stimulus vs 250-600 ms --
    input: results/block3_gradient_ratio_vs_difference.json

Usage: python supplementary_figures.py <leakage.npz> <stimuli.csv> <cascade.json> <gradient.json> <out_dir>
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEAK, STIM, CASC, GRAD, OUT = (Path(a) for a in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
ROIS = ["V1", "Post. IT", "FFC", "VWFA", "Post. STG", "Post. MTG", "Ant. STG", "Ant. MTG", "TP dorsal", "TP ventral"]


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# S1 leakage matrix
z = np.load(LEAK)
L = np.median(z["leak"], 0)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(L, cmap="magma_r", vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if L[i, j] > 0.6 else "black")
ax.set_xticks(range(10), ROIS, rotation=45, ha="right")
ax.set_yticks(range(10), ROIS)
ax.set_xlabel("true source ROI (unit dipoles)")
ax.set_ylabel("ROI where the dSPM estimate appears")
ax.set_title(f"S1 · ROI leakage of the block-3 dSPM pipeline (median, N = {z['leak'].shape[0]})", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="estimate / estimate in the source ROI")
save(fig, "S1_leakage_matrix")

# S2 stimulus properties
rows = list(csv.DictReader(open(STIM)))
props = [("letters", "letters"), ("ink", "ink area (px)"), ("height", "vertical extent (px)"), ("desc", "descender letters")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, (k, lab) in zip(axes, props):
    for x, cond, col in ((0, "word", "#2a78d6"), (1, "non-word", "#eb6834")):
        v = np.array([float(r[k]) for r in rows if r["condition"] == cond])
        ax.bar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color=col, width=0.6, capsize=4)
    ax.set_xticks([0, 1], ["Word", "Non-word"])
    ax.set_title(lab, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("S2 · Stimulus low-level properties (both lists, mean ± SEM; rendered in Courier New Bold, "
             "metric twin of the task font)", fontsize=11)
fig.tight_layout()
save(fig, "S2_stimulus_properties")

# S3 cascade windows
d = json.load(open(CASC, encoding="utf-8"))
rois = d["rois"]
cols = {k.replace("�", "–"): v for k, v in d["cols"].items()}
wins = ["80–130", "150–225", "250–600"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
x = np.arange(len(rois))
for ax, w in zip(axes, wins):
    for key, col, lab in ((f"FT env {w}", "#7b3fa0", "tag (FT envelope)"), (f"Evoked {w}", "#2a9d8f", "evoked")):
        c = cols[key]
        dz, pf = np.array(c["dz"]), np.array(c["p_fdr"])
        ax.plot(x, dz, "-", color=col, lw=1.5, label=lab)
        ax.scatter(x, dz, s=36, color=np.where(pf < 0.05, col, "white"), edgecolor=col, zorder=3)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_title(f"{w} ms", fontsize=11)
    ax.set_xticks(x, rois, rotation=55, ha="right", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Word > Non-word (dz)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"S3 · Tag vs evoked in the a-priori cascade windows (N = {d['n']}; filled = p_FDR < .05)", fontsize=11)
fig.tight_layout()
save(fig, "S3_cascade_windows")

# S4 gradient decomposition
g = json.load(open(GRAD))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, w, title in zip(axes, ("whole", "pre", "post250_600"), ("whole epoch", "pre-stimulus (−200–0 ms)", "250–600 ms")):
    for key, col, lab in (("ratio_N22", "#444444", "log ratio, N = 22"), ("diff_raw_N19", "#2a78d6", "difference, N = 19"),
                          ("diff_scaled_N22", "#eb6834", "difference, N = 22 (global scale)")):
        r = g[w][key]
        dz, p = np.array(r["dz"]), np.array(r["p"])
        ax.plot(range(10), dz, "-o", color=col, ms=4, lw=1.3, label=f"{lab}: slope t = {r['slope_t']:.2f}")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(10), ROIS, rotation=55, ha="right", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Word > Non-word (dz)")
fig.suptitle("S4 · Block-3 gradient is not a ratio artifact; its anterior part is present before word onset",
             fontsize=11)
fig.tight_layout()
save(fig, "S4_gradient_decomposition")
print("saved S1-S4 in", OUT)
