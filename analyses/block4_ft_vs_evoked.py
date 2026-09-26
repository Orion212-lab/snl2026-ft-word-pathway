"""Block-4 analysis: FT as a dependent variable vs the evoked response, per ROI, same per-word
epochs (p3 job). Everything in log space (invariant to per-subject gain, incl. the
sub-06/07/21 dSPM scale error); paired effect size dz, FDR over the 12 ROIs per column."""
import json
import sys

import numpy as np
from mne.stats import fdr_correction
from scipy import stats

z = np.load(sys.argv[1])
t = z["times"]
subs = list(z["subjects"])
NAMES = {"L_V1_ROI-lh": "V1", "L_PIT_ROI-lh": "Post. IT", "L_FFC_ROI-lh": "FFC", "VWFA_MNI-lh": "VWFA",
         "L_STSdp_ROI-lh": "Post. STG", "L_TPOJ2_ROI-lh": "Post. MTG", "L_STGa_ROI-lh": "Ant. STG",
         "L_STSva_ROI-lh": "Ant. MTG", "L_TGd_ROI-lh": "TP dorsal", "L_TGv_ROI-lh": "TP ventral",
         "Left-temporal": "Left-temporal (sensors)", "Occipital": "Occipital (sensors)"}
rois = [NAMES[r] for r in z["src_rois"]] + [NAMES[r] for r in z["sens_rois"]]
C = ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")


def get(kind):
    """(cond, sub, roi[, time]) over source + sensor ROIs."""
    return np.stack([np.concatenate([z[f"src_{kind}_{c}"], z[f"sens_{kind}_{c}"]], axis=1) for c in C])


EV, ENV, SNR = np.log(get("evoked")), np.log(get("env")), np.log(get("snr"))
pre = t < 0
win = {"150–350": (t >= 0.15) & (t <= 0.35), "300–500": (t >= 0.30) & (t <= 0.50)}


def lex(x):  # log(Word) - log(Non-word), F1/F2 averaged
    return 0.5 * ((x[0] - x[2]) + (x[1] - x[3]))


cols = {
    "FT detect (SNR>1)": SNR.mean(0),                                          # sub x roi
    "Evoked detect (vs pre)": EV[..., win["300–500"]].mean(-1).mean(0) - EV[..., pre].mean(-1).mean(0),
    "FT lex (SNR)": lex(SNR),
    "FT lex (env 150–350)": lex(ENV[..., win["150–350"]].mean(-1)),
    "FT lex (env 300–500)": lex(ENV[..., win["300–500"]].mean(-1)),
    "Evoked lex (150–350)": lex(EV[..., win["150–350"]].mean(-1)),
    "Evoked lex (300–500)": lex(EV[..., win["300–500"]].mean(-1)),
}
res = {"subjects": subs, "rois": rois, "cols": {}}
for name, X in cols.items():
    tv, pv = stats.ttest_1samp(X, 0)
    _, pf = fdr_correction(pv)
    dz = X.mean(0) / X.std(0, ddof=1)
    res["cols"][name] = {"dz": dz.tolist(), "t": tv.tolist(), "p": pv.tolist(), "p_fdr": pf.tolist()}
print(f"N = {len(subs)}: {' '.join(subs)}")
hdr = f"{'ROI':25s}" + "".join(f"{k[:20]:>22s}" for k in cols)
print(hdr)
for i, r in enumerate(rois):
    row = f"{r:25s}"
    for k in cols:
        c = res["cols"][k]
        star = "**" if c["p_fdr"][i] < 0.05 else ("* " if c["p"][i] < 0.05 else "  ")
        row += f"{c['dz'][i]:>+18.2f} {star}  "
    print(row)
print("** p_FDR < .05 over 12 ROIs; * p < .05 uncorrected")
json.dump(res, open(sys.argv[2], "w"), indent=1)
