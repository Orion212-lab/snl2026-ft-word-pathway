"""FT vs evoked per cascade stage (windows fixed a priori from the Background panel):
vision ~100 ms (80-130), letters/vOT (150-225), meaning (250-600). Same per-word epochs (p3, N=17), log space.
Outputs: effect-size table per ROI x DV, paired bootstrap of dz(evoked) - dz(FT SNR) per window,
FDR over ROIs within each column."""
import json
import sys

import numpy as np
from mne.stats import fdr_correction
from scipy import stats

q = np.load(sys.argv[1])
t = q["times"]
LAB = {"L_V1_ROI-lh": "V1", "L_PIT_ROI-lh": "Post. IT", "L_FFC_ROI-lh": "FFC", "VWFA_MNI-lh": "VWFA",
       "L_STSdp_ROI-lh": "Post. STG", "L_TPOJ2_ROI-lh": "Post. MTG", "L_STGa_ROI-lh": "Ant. STG",
       "L_STSva_ROI-lh": "Ant. MTG", "L_TGd_ROI-lh": "TP dorsal", "L_TGv_ROI-lh": "TP ventral",
       "Left-temporal": "Left-temporal (sensors)", "Occipital": "Occipital (sensors)"}
rois = [LAB[str(r)] for r in q["src_rois"]] + [LAB[str(r)] for r in q["sens_rois"]]
C = ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")
WIN = {"80–130": (0.08, 0.13), "150–225": (0.15, 0.225), "250–600": (0.25, 0.60)}


def get(kind):
    return np.stack([np.concatenate([q[f"src_{kind}_{c}"], q[f"sens_{kind}_{c}"]], axis=1) for c in C])


def lex(x):
    return 0.5 * ((x[0] - x[2]) + (x[1] - x[3]))


def dz(x, ax=0):
    return x.mean(ax) / x.std(ax, ddof=1)


EV, ENV, SNR = np.log(get("evoked")), np.log(get("env")), np.log(get("snr"))
pre = t < 0
cols = {"FT detect (SNR>1)": SNR.mean(0), "FT SNR (whole word)": lex(SNR)}
for w, (a, b) in WIN.items():
    m = (t >= a) & (t <= b)
    cols[f"FT env {w}"] = lex(ENV[..., m].mean(-1))
for w, (a, b) in WIN.items():
    m = (t >= a) & (t <= b)
    cols[f"Evoked {w}"] = lex(EV[..., m].mean(-1))
res = {"rois": rois, "n": int(q["subjects"].shape[0]), "cols": {}, "boot": {}}
for k, X in cols.items():
    tv, pv = stats.ttest_1samp(X, 0)
    _, pf = fdr_correction(pv)
    res["cols"][k] = {"dz": dz(X).tolist(), "p": pv.tolist(), "p_fdr": pf.tolist()}
rng = np.random.default_rng(42)
n = res["n"]
ft = cols["FT SNR (whole word)"]
for w in WIN:
    ev = cols[f"Evoked {w}"]
    out = []
    for r in range(len(rois)):
        bs = np.empty(10000)
        for i in range(10000):
            s = rng.integers(0, n, n)
            bs[i] = dz(ev[s, r]) - dz(ft[s, r])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(diff=float(dz(ev[:, r]) - dz(ft[:, r])), lo=float(lo), hi=float(hi),
                        p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    res["boot"][w] = out
print(f"N = {n}")
print(f"{'ROI':25s}" + "".join(f"{k[:18]:>20s}" for k in cols))
for i, r in enumerate(rois):
    line = f"{r:25s}"
    for k in cols:
        c = res["cols"][k]
        star = "**" if c["p_fdr"][i] < .05 else ("* " if c["p"][i] < .05 else "  ")
        line += f"{c['dz'][i]:>+17.2f} {star}"
    print(line)
print("\nDirect comparison dz(Evoked window) - dz(FT SNR), paired bootstrap:")
for w in WIN:
    for i in (rois.index("Left-temporal (sensors)"), rois.index("Occipital (sensors)"), rois.index("Ant. STG"),
              rois.index("VWFA")):
        b = res["boot"][w][i]
        print(f"  {w:8s} {rois[i]:25s} Δdz = {b['diff']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] p = {b['p']:.3f}")
json.dump(res, open(sys.argv[2], "w"), indent=1)
