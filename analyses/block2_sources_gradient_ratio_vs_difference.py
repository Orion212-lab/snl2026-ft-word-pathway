"""Analysis 1: is the block-3 posterior -> anterior gradient a ratio (denominator) artifact?

Block-3 data (N = 22, dSPM, per-word epochs, fixation baseline, equal counts). ROI values are
mean |dSPM| (noise-SD units), so a raw difference W - NW is comparable across ROIs without any
per-ROI normalisation. Three subjects (06/07/21) carry a 1e5-1e7 dSPM scale error, handled two ways:
  A. drop them (N = 19), raw dSPM differences;
  B. keep all 22, divide each subject by ONE condition- and ROI-blind scalar (median over all
     conditions x ROIs x samples) -- corrects the gain bug without per-ROI normalisation.
Ratio version (the poster's statistic) is recomputed for reference.
Windows: whole epoch (poster), pre (-200-0), post 250-600, and post - pre (word-locked change).
Tag ripple: 4 Hz low-pass (zero-phase) before windowing, so only the slow evoked part is kept.
"""
import sys
import numpy as np
from scipy import stats, signal

z = np.load(sys.argv[1])
t = z["times"]
subs = list(z["subjects"])
rois = list(z["roi_names"])
ORDER = ["vwfa_v1", "vwfa_pit", "vwfa_ffc", "vwfa_vwfa", "vwfa_pstg", "vwfa_pmtg", "vwfa_astg", "vwfa_amtg",
         "L_TGd_ROI-lh", "L_TGv_ROI-lh"]
LAB = ["V1", "Post. IT", "FFC", "VWFA", "Post. STG", "Post. MTG", "Ant. STG", "Ant. MTG", "TP dorsal", "TP ventral"]
idx = [rois.index(r) for r in ORDER]
X = np.stack([z[c][:, idx] for c in ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")])  # cond x sub x roi x t
sf = 1.0 / (t[1] - t[0])
b, a = signal.butter(4, 4.0 / (sf / 2), "low")
Xlp = signal.filtfilt(b, a, X, axis=-1)

W = {"whole": np.ones_like(t, bool), "pre": t < 0, "post250_600": (t >= .25) & (t <= .6)}
rank = np.arange(1, 11)
BAD = ["06", "07", "21"]


def lexdiff(Y):  # (W - NW), F1/F2 averaged: sub x roi x t
    return 0.5 * ((Y[0] - Y[2]) + (Y[1] - Y[3]))


def lexlog(Y):
    L = np.log(Y)
    return 0.5 * ((L[0] - L[2]) + (L[1] - L[3]))


def report(name, M):
    """M: sub x roi. Slope over pathway rank + ROI-wise dz + ant-vs-post temporal."""
    n = M.shape[0]
    sl = np.array([np.polyfit(rank, s, 1)[0] for s in M])
    r = stats.ttest_1samp(sl, 0)
    ant, post, vent = M[:, 6:10].mean(1), M[:, 4:6].mean(1), M[:, 0:4].mean(1)
    ap = stats.ttest_rel(ant, post)
    tv = stats.ttest_1samp(M, 0)
    dz = M.mean(0) / M.std(0, ddof=1)
    print(f"  {name:34s} N={n:2d}  slope t({n-1})={r.statistic:+5.2f} p={r.pvalue:.4f} "
          f"({int((sl > 0).sum())}/{n} +) | ant-post t={ap.statistic:+5.2f} p={ap.pvalue:.4f}")
    print("      dz per ROI: " + "  ".join(f"{l[:6]}={d:+.2f}{'*' if p < .05 else ''}"
                                         for l, d, p in zip(LAB, dz, tv.pvalue)))
    return dict(slope_t=float(r.statistic), slope_p=float(r.pvalue), n=n, dz=dz.tolist(),
                p=tv.pvalue.tolist(), ap_t=float(ap.statistic), ap_p=float(ap.pvalue))


keepA = [i for i, s in enumerate(subs) if s not in BAD]
scale = np.median(X.reshape(4, len(subs), -1).transpose(1, 0, 2).reshape(len(subs), -1), axis=1)
print("per-subject global dSPM scale (median):",
      " ".join(f"{s}:{v:.3g}" for s, v in zip(subs, scale)))

out = {}
for wname, m in W.items():
    print(f"\n=== window: {wname}")
    out[wname] = {}
    out[wname]["ratio_N22"] = report("RATIO log(W/NW)  [poster]", lexlog(X)[..., m].mean(-1))
    out[wname]["diff_raw_N19"] = report("DIFF raw dSPM, lp4Hz, no 06/07/21", lexdiff(Xlp[:, keepA])[..., m].mean(-1))
    Xs = Xlp / scale[None, :, None, None]
    out[wname]["diff_scaled_N22"] = report("DIFF global-scaled, lp4Hz", lexdiff(Xs)[..., m].mean(-1))
    # shared (condition-independent) response per ROI: mean over all 4 conditions
    shared = X[:, keepA][..., m].mean(-1).mean(0).mean(0)  # roi
    print("      shared |dSPM| per ROI (N=19): " + "  ".join(f"{l[:6]}={v:.2f}" for l, v in zip(LAB, shared)))

print("\n=== word-locked change: (W-NW)[250-600] - (W-NW)[pre]")
for name, Y, keep in (("DIFF raw, no 06/07/21", Xlp, keepA), ("DIFF global-scaled", Xlp / scale[None, :, None, None], None)):
    D = lexdiff(Y if keep is None else Y[:, keep])
    report(name, D[..., W["post250_600"]].mean(-1) - D[..., W["pre"]].mean(-1))
L = lexlog(X)
report("RATIO, post - pre", L[..., W["post250_600"]].mean(-1) - L[..., W["pre"]].mean(-1))

import json
json.dump(out, open(sys.argv[2], "w"), indent=1)
