"""Block 3 (A0): source-space Word vs Non-word in log space (scale-invariant, fixes the
sub-06/07/21 dSPM gain bug without exclusion). (a) per-ROI sustained effect: mean
log(Word/Non-word) over the whole word epoch (-200..1000 ms), paired t-test, FDR over 10 ROIs;
(b) time course in the two strongest anterior-temporal ROIs, log-ratio to pre-stimulus."""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mne.stats import fdr_correction  # noqa: E402
from scipy import stats  # noqa: E402

NPZ, OUT = Path(sys.argv[1]), Path(os.environ.get("POSTER_OUT", "figures_out"))
W, H = 11.4, 4.4
TICK, LABEL = 13, 16
ORDER = [("vwfa_v1", "V1"), ("vwfa_pit", "Post. IT"), ("vwfa_ffc", "FFC"), ("vwfa_vwfa", "VWFA"),
         ("vwfa_pstg", "Post. STG"), ("vwfa_pmtg", "Post. MTG"), ("vwfa_astg", "Ant. STG"),
         ("vwfa_amtg", "Ant. MTG"), ("L_TGd_ROI-lh", "TP dorsal"), ("L_TGv_ROI-lh", "TP ventral")]
C_WORD, C_NON, C_VENT, C_TEMP = "#2a78d6", "#eb6834", "0.55", "#1b4f8a"

z = np.load(NPZ)
t, rois = z["times"], list(z["roi_names"])
C = ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")
L = np.log(np.stack([z[c] for c in C]))  # cond x sub x roi x time
n = L.shape[1]
lex = 0.5 * ((L[0] - L[2]) + (L[1] - L[3]))  # sub x roi x time, log(Word/Non-word)
whole = lex.mean(2)
tv, pv = stats.ttest_1samp(whole, 0)
_, pfdr = fdr_correction(pv)
res = {}
fig = plt.figure(figsize=(W, H), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
ax = fig.add_subplot(gs[0])
for k, (roi, lab) in enumerate(ORDER):
    r = rois.index(roi)
    x = whole[:, r]
    m, ci = x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    col = C_TEMP if k >= 4 else C_VENT
    y = len(ORDER) - 1 - k
    ax.plot([m - ci, m + ci], [y, y], color=col, lw=3)
    ax.plot(m, y, "o", color=col, ms=9, mec="white")
    star = "*" if pfdr[r] < 0.05 else ""
    ax.text(0.33, y, f"{star}", va="center", fontsize=18, color=col)
    res[lab] = dict(mean=float(m), t=float(tv[r]), p=float(pv[r]), p_fdr=float(pfdr[r]))
ax.axvline(0, color="0.5", ls="--", lw=1)
ax.set_yticks(range(len(ORDER)), [lab for _, lab in ORDER][::-1], fontsize=TICK)
ax.set_xlabel("log(Word / Non-word)\nwhole epoch", fontsize=LABEL)
ax.tick_params(axis="x", labelsize=TICK)
ax.set_xlim(-0.2, 0.38)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.02, 1.01, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom", ha="right")
pre = t < 0
for j, (roi, lab) in enumerate((("vwfa_astg", "Ant. STG"), ("L_TGv_ROI-lh", "TP ventral"))):
    axj = fig.add_subplot(gs[j + 1])
    r = rois.index(roi)
    for (ci_, cond, col) in ((0, "Word", C_WORD), (2, "Non-word", C_NON)):
        x = 0.5 * (L[ci_, :, r] + L[ci_ + 1, :, r])
        x = x - L[:, :, r][:, :, pre].mean(axis=(0, 2))[:, None]  # condition-blind pre-stimulus reference
        m, se = x.mean(0), x.std(0, ddof=1) / np.sqrt(n)
        axj.fill_between(t * 1000, m - se, m + se, color=col, alpha=0.2, lw=0)
        axj.plot(t * 1000, m, color=col, lw=1.6, label=cond)
    axj.axvline(0, color="0.5", ls=":", lw=1)
    axj.axhline(0, color="0.7", lw=0.8)
    axj.set_title(lab, fontsize=16, fontweight="bold")
    axj.set_xlabel("Time (ms)", fontsize=LABEL)
    axj.tick_params(labelsize=TICK)
    axj.spines[["top", "right"]].set_visible(False)
    if j == 0:
        axj.set_ylabel("log dSPM (rel. to pre-stim.)", fontsize=LABEL - 2)
        axj.legend(fontsize=13, frameon=False, loc="upper left")
        axj.text(-0.02, 1.01, "b", transform=axj.transAxes, fontsize=22, fontweight="bold", va="bottom", ha="right")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block3_source_log.{ext}", dpi=300)
json.dump(res, open(OUT / "block3_source_log_stats.json", "w"), indent=1)
print(f"N = {n}")
for k, v in res.items():
    print(f"  {k:11s} mean={v['mean']:+.3f} t({n-1})={v['t']:+.2f} p={v['p']:.4f} p_FDR={v['p_fdr']:.4f}")
