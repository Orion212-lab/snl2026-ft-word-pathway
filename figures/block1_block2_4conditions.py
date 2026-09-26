"""Blocks 1-2 (A0) with the 4 conditions kept separate (WORD/F1, WORD/F2, NONWORD/F1, NONWORD/F2).

Block 1 (7.0 x 4.3 in): same per-word epochs for both DVs (p3 job), left-temporal gradiometers.
   FT: per-word tag SNR at each condition's own tag; Evoked: mean |field| 300-500 ms.
   Paired Word vs Non-word test within each tag.
Block 2 panel a (4.6 x 3.0 in): Word - Non-word effect size over time, per tag, for both DVs.
Block 2 topomaps (8.45 x 3.7 in): grand-average gradiometer field (RMS of pairs), 4 condition rows.
"""
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from mne.stats import permutation_cluster_1samp_test  # noqa: E402
from scipy import stats  # noqa: E402

P3, GA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(os.environ.get("POSTER_OUT", "figures_out"))
TICK, LABEL, TITLE = 13, 16, 16
COND = [("WORD_F1", "Word\nF1", "#2a78d6"), ("NONWORD_F1", "Non-word\nF1", "#eb6834"),
        ("WORD_F2", "Word\nF2", "#1b4f8a"), ("NONWORD_F2", "Non-word\nF2", "#b8461a")]
C_FT, C_EV = "#7b3fa0", "#2a9d8f"
rng = np.random.default_rng(0)

q = np.load(P3)
t = q["times"]
ri = list(q["sens_rois"]).index("Left-temporal")
n = q["subjects"].shape[0]
win = (t >= 0.25) & (t <= 0.6)  # "meaning" stage of the Background cascade
vals = {"FT": {c: q[f"sens_snr_{c}"][:, ri] for c, _, _ in COND},
        "Evoked": {c: q[f"sens_evoked_{c}"][:, ri][:, win].mean(1) * 1e13 for c, _, _ in COND}}


def ptxt(p):
    return "p < .001" if p < 0.001 else f"p = {p:.2f}".replace("0.", ".")


# ---------------- block 1 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.3), layout="constrained")
for ax, (dv, title, col, unit) in zip(axes, (("FT", "FT: tag SNR", C_FT, "SNR (per word)"),
                                             ("Evoked", "Evoked: 250–600 ms", C_EV, "|field| (fT/cm)"))):
    pos = [0, 1, 2.4, 3.4]
    data = [vals[dv][c] for c, _, _ in COND]
    parts = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for b, (_, _, cc) in zip(parts["bodies"], COND):
        b.set_facecolor(cc); b.set_alpha(0.22); b.set_edgecolor(cc)
    j = rng.uniform(-0.07, 0.07, n)
    for (a, b_), (x0, x1) in (((data[0], data[1]), (0, 1)), ((data[2], data[3]), (2.4, 3.4))):
        for u, v, jj in zip(a, b_, j):
            ax.plot([x0 + jj, x1 + jj], [u, v], color="0.8", lw=0.6, zorder=1)
    ymax = max(np.max(d) for d in data)
    for k, (d, x, (_, _, cc)) in enumerate(zip(data, pos, COND)):
        ax.scatter(x + j, d, s=16, color=cc, edgecolor="white", lw=0.4, zorder=2)
        ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        ax.errorbar([x], [d.mean()], yerr=[ci], fmt="D", color=cc, ms=7, mec="black", mew=0.6, capsize=4, zorder=3)
    for (a, b_), xm, tag in (((data[0], data[1]), 0.5, "F1"), ((data[2], data[3]), 2.9, "F2")):
        tt, p = stats.ttest_rel(a, b_)
        dz = (a - b_).mean() / (a - b_).std(ddof=1)
        ax.text(xm, ymax * 1.04, f"dz = {dz:.2f}\n{ptxt(p)}", ha="center", va="bottom", fontsize=11.5, color="0.25")
        print(f"  block1 {dv} {tag}: dz = {dz:.2f}, t({n-1}) = {tt:.2f}, p = {p:.4f}")
    ax.set_ylim(top=ymax * 1.28)
    ax.set_xticks(pos, ["W", "NW", "W", "NW"], fontsize=13)
    for xm, tag in ((0.5, "F1 (6.0 Hz)"), (2.9, "F2 (7.06 Hz)")):
        ax.text(xm, -0.13, tag, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12.5,
                fontweight="bold")
    ax.set_title(title, fontsize=TITLE, color=col, fontweight="bold")
    ax.set_ylabel(unit, fontsize=LABEL - 2)
    ax.tick_params(axis="y", labelsize=TICK - 1)
    ax.spines[["top", "right"]].set_visible(False)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block1_4cond_violin.{ext}", dpi=300)
plt.close(fig)

# ---------------- block 2 panel a: dz over time per tag -----------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
sel = t >= 0
k = 20
for kind, lab, col in (("evoked", "Evoked", C_EV), ("env", "FT envelope", C_FT)):
    for tag, ls in (("F1", "-"), ("F2", "--")):
        x = np.log(q[f"sens_{kind}_WORD_{tag}"][:, ri]) - np.log(q[f"sens_{kind}_NONWORD_{tag}"][:, ri])
        x = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="same"), 1, x)
        dz = x.mean(0) / x.std(0, ddof=1)
        T, cl, pv, _ = permutation_cluster_1samp_test(x[:, sel], threshold=stats.t.ppf(0.975, n - 1),
                                                      n_permutations=5000, tail=0, seed=42, out_type="mask",
                                                      verbose=False)
        for c, p in zip(cl, pv):
            if p < 0.05:
                ts = t[sel][c] * 1000
                ax.plot([ts.min(), ts.max()], [-0.55 - (0.12 if tag == "F2" else 0)] * 2, color=col, lw=4,
                        ls=ls, solid_capstyle="butt")
                print(f"  block2 {lab} {tag}: cluster {ts.min():.0f}-{ts.max():.0f} ms p={p:.3f}")
        ax.plot(t * 1000, dz, color=col, lw=1.8, ls=ls, label=f"{lab} {tag}")
ax.axhline(0, color="0.6", lw=0.8)
ax.axvline(0, color="0.5", ls=":", lw=1)
ax.set_xlabel("Time from word onset (ms)", fontsize=LABEL - 1)
ax.set_ylabel("Effect size (dz)", fontsize=LABEL - 1)
ax.tick_params(labelsize=TICK - 1)
ax.legend(fontsize=10.5, frameon=False, loc="upper left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.22, 1.03, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block2_4cond_dz.{ext}", dpi=300)
plt.close(fig)

# ---------------- block 2 topomaps: 4 condition rows --------------------------------------------
TIMES = [0.10, 0.15, 0.25, 0.35, 0.50]
ev = {c: mne.read_evokeds(GA / f"grandavg_{c.replace('_', '-')}-ave.fif", verbose=False)[0].pick("grad")
      for c, _, _ in COND}
print("  grand averages:", {c: e.nave for c, e in ev.items()})
for e in ev.values():
    ch = e.ch_names
    assert all(a[:-1] == b[:-1] for a, b in zip(ch[0::2], ch[1::2])), "gradiometers not in consecutive pairs"
fig, axes = plt.subplots(4, len(TIMES), figsize=(8.45, 3.7))
fig.subplots_adjust(left=0.13, right=0.9, top=0.92, bottom=0.01, wspace=0.02, hspace=0.04)
vmax = 0
rows = []
for c, lab, _ in COND:
    e = ev[c]
    row = []
    for t0 in TIMES:
        m = (e.times >= t0 - 0.025) & (e.times <= t0 + 0.025)
        d = e.data[:, m].mean(1)
        row.append(d * 1e13)  # mne merges each planar pair by RMS
        vmax = max(vmax, (np.sqrt((d[0::2] ** 2 + d[1::2] ** 2) / 2) * 1e13).max())  # mne merge = RMS of pair
    rows.append(row)
pos_info = ev["WORD_F1"].info
for r, ((c, lab, cc), row) in enumerate(zip(COND, rows)):
    for k2, (t0, rms) in enumerate(zip(TIMES, row)):
        im, _ = mne.viz.plot_topomap(rms, pos_info, axes=axes[r, k2], show=False, cmap="Reds", vlim=(0, vmax),
                                     contours=0, sensors=False)
        if r == 0:
            axes[r, k2].set_title(f"{t0 * 1000:.0f} ms", fontsize=13)
    axes[r, 0].text(-0.06, 0.5, lab, transform=axes[r, 0].transAxes,
                    ha="right", va="center", fontsize=11, color=cc, fontweight="bold")
cax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
cb = fig.colorbar(im, cax=cax)
cb.set_label("fT/cm", fontsize=12)
cb.ax.tick_params(labelsize=10)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block2_4cond_topo.{ext}", dpi=300)
print("saved block1_4cond_violin, block2_4cond_dz, block2_4cond_topo")
