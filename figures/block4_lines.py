"""Block 4 (A0): Word-vs-Non-word effect size along the pathway, FT vs evoked, same per-word epochs (p3, N=17).
(a) dz per source ROI (posterior -> anterior), two DVs: FT = per-word tag SNR (log), evoked = |dSPM| 250-600 ms
    (log); bootstrap 95% CI; filled marker = p_FDR < .05 over the 10 ROIs.
(b) same on the a-priori sensor ROIs, with the direct paired test dz(evoked) - dz(FT)."""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mne.stats import fdr_correction  # noqa: E402
from scipy import stats  # noqa: E402

q = np.load(sys.argv[1])
OUT = Path(os.environ.get("POSTER_OUT", "figures_out"))
t = q["times"]
C = ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")
SRC = ["V1", "Post. IT", "FFC", "VWFA", "Post. STG", "Post. MTG", "Ant. STG", "Ant. MTG", "TP dorsal", "TP ventral"]
SENS = ["Occipital", "Left-temporal"]
C_FT, C_EV = "#7b3fa0", "#2a9d8f"
rng = np.random.default_rng(42)


def lex(space, kind, win=None):
    L = np.log(np.stack([q[f"{space}_{kind}_{c}"] for c in C]))
    if win is not None:
        L = L[..., (t >= win[0]) & (t <= win[1])].mean(-1)
    return 0.5 * ((L[0] - L[2]) + (L[1] - L[3]))  # sub x roi


def dz(x):
    return x.mean(0) / x.std(0, ddof=1)


def boot_ci(x):
    n = x.shape[0]
    b = np.array([dz(x[rng.integers(0, n, n)]) for _ in range(5000)])
    return np.percentile(b, [2.5, 97.5], axis=0)


data = {("src", "FT"): lex("src", "snr"), ("src", "Evoked"): lex("src", "evoked", (0.25, 0.60)),
        ("sens", "FT"): lex("sens", "snr"), ("sens", "Evoked"): lex("sens", "evoked", (0.25, 0.60))}
sens_order = [list(q["sens_rois"]).index(s) for s in SENS]
n = data[("src", "FT")].shape[0]

fig = plt.figure(figsize=(11.4, 2.85))
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.25, left=0.07, right=0.99, top=0.88, bottom=0.27)
ax = fig.add_subplot(gs[0])
x = np.arange(len(SRC))
for dv, col, off in (("FT", C_FT, -0.08), ("Evoked", C_EV, 0.08)):
    X = data[("src", dv)]
    d, (lo, hi) = dz(X), boot_ci(X)
    _, pf = fdr_correction(stats.ttest_1samp(X, 0).pvalue)
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(x, d, color=col, lw=2.5, label="FT: tag SNR" if dv == "FT" else "Evoked: 250–600 ms")
    for i in range(len(SRC)):
        ax.plot(x[i], d[i], "o", ms=9, color=col if pf[i] < 0.05 else "white", mec=col, mew=2, zorder=3)
    print(dv, " ".join(f"{s}:{v:+.2f}{'*' if p < .05 else ''}" for s, v, p in zip(SRC, d, pf)))
ax.axhline(0, color="0.5", lw=1, ls="--")
ax.axvspan(-0.5, 3.5, color="0.93", zorder=0)
ax.text(1.5, 1.28, "ventral stream", ha="center", fontsize=12, color="0.4")
ax.text(6.5, 1.28, "temporal cortex", ha="center", fontsize=12, color="0.4")
ax.set_xticks(x, SRC, fontsize=11.5, rotation=30, ha="right")
ax.set_xlim(-0.5, len(SRC) - 0.5)
ax.set_ylim(-0.9, 1.45)
ax.set_ylabel("Word > Non-word (dz)", fontsize=14)
ax.tick_params(axis="y", labelsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=12, frameon=False, loc="lower right", ncol=2)
ax.text(-0.05, 1.03, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", ha="right", va="bottom")

ax2 = fig.add_subplot(gs[1])
for k, s in enumerate(SENS):
    for dv, col, off in (("FT", C_FT, -0.12), ("Evoked", C_EV, 0.12)):
        X = data[("sens", dv)][:, sens_order[k]]
        d = X.mean() / X.std(ddof=1)
        b = np.array([(lambda y: y.mean() / y.std(ddof=1))(X[rng.integers(0, n, n)]) for _ in range(5000)])
        lo, hi = np.percentile(b, [2.5, 97.5])
        p = stats.ttest_1samp(X, 0).pvalue
        ax2.plot([k + off] * 2, [lo, hi], color=col, lw=2.5)
        ax2.plot(k + off, d, "o", ms=9, color=col if p < 0.05 else "white", mec=col, mew=2)
ev, ft = data[("sens", "Evoked")][:, sens_order[1]], data[("sens", "FT")][:, sens_order[1]]
bd = []
for _ in range(10000):
    i = rng.integers(0, n, n)
    bd.append(ev[i].mean() / ev[i].std(ddof=1) - ft[i].mean() / ft[i].std(ddof=1))
bd = np.array(bd)
dd = ev.mean() / ev.std(ddof=1) - ft.mean() / ft.std(ddof=1)
pb = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
ax2_title = (f"left-temporal: Δdz = {dd:.2f}, p = {pb:.2f}").replace("0.", ".")
ax2.axhline(0, color="0.5", lw=1, ls="--")
ax2.set_xticks([0, 1], ["Occipital", "Left-\ntemporal"], fontsize=11.5)
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.9, 2.7)
ax2.tick_params(axis="y", labelsize=12)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("sensors", fontsize=12, color="0.4")
ax2.text(0.5, 0.99, ax2_title.replace(": ", ":" + chr(10)), transform=ax2.transAxes, ha="center", va="top",
         fontsize=11, color="0.2")
ax2.text(-0.12, 1.03, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", ha="right", va="bottom")
print(f"N = {n}; left-temporal sensors Δdz = {dd:.2f}, p = {pb:.3f}")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block4_lines.{ext}", dpi=300)
