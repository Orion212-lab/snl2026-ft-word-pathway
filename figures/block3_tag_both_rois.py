"""Poster block 3 (17.95 x 4.35 in): tag vs word-locked response, left-temporal AND occipital sensors.

Left  (2 x 2): per-word phase-locked SNR spectra (2.83 s FFT, df = 0.35 Hz), rows Word / Non-word,
      columns left-temporal / occipital; blue = F1 (6.00 Hz) trials, orange = F2 (7.06 Hz) trials.
Right (1 x 4): the 4 conditions (Word/Non-word x F1/F2), one dot per subject:
      tag = sum of (SNR - 1) over harmonics 1-4 at the condition's own tag (spectra above);
      evoked = mean gradiometer field strength 250-600 ms after word onset.
      Paired Word vs Non-word per tag: dz and p above each pair.
Same subjects throughout (N = 20, sensor cohort; spectra and evoked from the same epochs).

Usage: python block3_tag_both_rois.py <spectra.npz (per-word SNR)> <perword.npz (evoked)> <out_dir>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

SPEC, EVK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "Arial"
C_FT, C_EV = "#7b3fa0", "#2a9d8f"
C_F1, C_F2 = "#1f5fa8", "#e08a2c"
COND = [("WORD_F1", "W", "#2a78d6"), ("NONWORD_F1", "NW", "#eb6834"),
        ("WORD_F2", "W", "#1b4f8a"), ("NONWORD_F2", "NW", "#b8461a")]
ROIS = [("Left-temporal", "left-temporal"), ("Occipital", "occipital")]

q, e = np.load(SPEC), np.load(EVK)
subs = list(e["subjects"])
isub = [list(q["subjects"]).index(s) for s in subs]
f = q["freqs"]
df = f[1] - f[0]
BIN = {"F1": int(round(6.0 / df)), "F2": int(round((240 / 34) / df))}
t = e["times"]
win = (t >= 0.25) & (t <= 0.60)
rng = np.random.default_rng(0)


def harm(cond, r):
    s = q[f"snr_{cond}"][isub, r]
    b = BIN[cond[-2:]]
    return sum(s[:, k * b] - 1 for k in range(1, 5))


def evoked(cond, roi):
    r = list(e["sens_rois"]).index(roi)
    return e[f"sens_evoked_{cond}"][:, r][:, win].mean(1) * 1e13


fig = plt.figure(figsize=(17.95, 4.35))
gs = fig.add_gridspec(2, 8, width_ratios=[0.5, 1.4, 1.4, 0.1, 1, 1, 1, 1], wspace=0.42, hspace=0.35,
                      left=0.005, right=0.995, top=0.88, bottom=0.2)

# ---------------- spectra
fsel = (f >= 4) & (f <= 19.2)
for c_i, (roi, lab) in enumerate(ROIS):
    r = list(q["rois"]).index(roi)
    ymax = max(q[f"snr_{c}"][isub, r][:, fsel].mean(0).max() for c, _, _ in COND) * 1.12
    for r_i, (grp, word) in enumerate((("WORD", "wary"), ("NONWORD", "hkwj"))):
        ax = fig.add_subplot(gs[r_i, 1 + c_i])
        for tag, col in (("F1", C_F1), ("F2", C_F2)):
            ax.plot(f[fsel], q[f"snr_{grp}_{tag}"][isub, r][:, fsel].mean(0), color=col, lw=1.6,
                    label=f"{tag} trials")
        for x in (6.0, 240 / 34):
            ax.axvline(x, color="0.35", ls=":", lw=1)
        ax.set_ylim(0, ymax)
        ax.set_xlim(4, 19.2)
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if r_i == 0:
            ax.set_title(lab, fontsize=15, fontweight="bold", color="0.2")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Frequency (Hz)", fontsize=13)
        if c_i == 0:
            ax.set_ylabel("SNR", fontsize=13)
            ax.text(-0.42, 0.5, f"{word}\n{'Word' if grp == 'WORD' else 'Non-word'}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, fontweight="bold", color="0.15")
        if r_i == 0 and c_i == 1:
            ax.legend(fontsize=10, frameon=False, loc="upper right")

# ---------------- violins
panels = [("Tag (FT)", C_FT, "Σ (SNR − 1), harmonics 1–4", harm, "idx"),
          ("Evoked 250–600 ms", C_EV, "|field| (fT/cm)", evoked, "roi")]
col_i = 4
for dv, color, ylab, fn, how in panels:
    for roi, lab in ROIS:
        ax = fig.add_subplot(gs[:, col_i])
        col_i += 1
        r = list(q["rois"]).index(roi)
        vals = {c: (fn(c, r) if how == "idx" else fn(c, roi)) for c, _, _ in COND}
        pos = [0, 1, 2.4, 3.4]
        for (c, lb, col), x in zip(COND, pos):
            v = vals[c]
            vp = ax.violinplot(v, positions=[x], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.25); b.set_edgecolor("none")
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=col, zorder=3)
            m, ci = v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(x, m, yerr=ci, fmt="D", color=col, mec="0.1", ms=6, capsize=3, zorder=4)
        for a, b in ((0, 1), (2, 3)):
            ca, cb = COND[a][0], COND[b][0]
            for s_i in range(len(subs)):
                ax.plot([pos[a], pos[b]], [vals[ca][s_i], vals[cb][s_i]], color="0.8", lw=0.6, zorder=1)
        top = max(v.max() for v in vals.values())
        bot = min(v.min() for v in vals.values())
        for (a, b), tag in (((0, 1), "F1"), ((2, 3), "F2")):
            d = vals[COND[a][0]] - vals[COND[b][0]]
            tt = stats.ttest_1samp(d, 0)
            ptxt = "p < .001" if tt.pvalue < 0.001 else f"p = {tt.pvalue:.3f}".replace("0.", ".")
            ax.text((pos[a] + pos[b]) / 2, top + 0.06 * (top - bot), f"dz = {d.mean()/d.std(ddof=1):+.2f}\n{ptxt}",
                    ha="center", va="bottom", fontsize=10.5, color="0.25")
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.3 * (top - bot))
        ax.set_xticks(pos, ["W", "NW", "W", "NW"], fontsize=12)
        for x, tag in ((0.5, "F1"), (2.9, "F2")):
            ax.text(x, -0.14, tag, transform=ax.get_xaxis_transform(), ha="center", fontsize=12, fontweight="bold")
        ax.set_title(f"{dv}\n{lab}", fontsize=13.5, fontweight="bold", color=color)
        ax.set_ylabel(ylab, fontsize=12)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)

for ext in ("svg", "png"):
    fig.savefig(OUT / f"block3_tag_both_rois.{ext}", dpi=300)
print("saved block3_tag_both_rois; N =", len(subs))
