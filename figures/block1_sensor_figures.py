"""Points 4-5: block-2 figures for the poster (same sizes as the slots they replace).

(a) 4.6 x 3.0 in: mean |t| (Word - Non-word, gradiometer pair RMS, N = 26) over the cluster window,
    cluster sensors marked -- same data and test as the text (N = 26, 5000 perm.).
(b) 4.6 x 3.0 in: GFP of the group W - NW difference with its sign-flip noise floor (95th percentile per
    sample, shaded) and the max-over-time FWER threshold (dashed); cluster span shaded.
(c) 8.45 x 3.7 in: grand-average gradiometer field (RMS of pairs), columns = 4 conditions,
    rows = 100 / 350 / 500 ms (+-25 ms), N = 25 grand averages.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

mne.set_log_level("ERROR")
NPZ, GA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Arial"  # metric-compatible with the Liberation Sans used before
TICK, LABEL, PANEL = 13, 16, 22
rng = np.random.default_rng(0)

z = np.load(NPZ, allow_pickle=True)
t, keys = z["times"], list(z["keys"])
D = z["word_grad"] - z["nonword_grad"]  # sub x sensor x time, already in fT/cm
n = D.shape[0]

adj, names = mne.channels.read_ch_adjacency("neuromag306mag")
adj = adj[[names.index(k + "1") for k in keys]][:, [names.index(k + "1") for k in keys]]
T, clu, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
    D.transpose(0, 2, 1), threshold=stats.t.ppf(0.975, n - 1), n_permutations=5000, adjacency=adj,
    seed=0, out_type="mask")
best = clu[int(np.argmin(pv))]  # time x sensor
sig_t = best.any(1)
print(f"cluster p = {pv.min():.4f}, {t[sig_t].min()*1000:.0f}-{t[sig_t].max()*1000:.0f} ms, {best.any(0).sum()} sensors")

ev = {c: mne.read_evokeds(GA / f"grandavg_{c}-ave.fif")[0] for c in ("WORD-F1", "NONWORD-F1", "WORD-F2", "NONWORD-F2")}
info_all = ev["WORD-F1"].info
mag_info = mne.pick_info(info_all, [info_all["ch_names"].index(k + "2") for k in keys])  # one position per pair (grads only in file)
from mne.io.constants import FIFF  # noqa: E402
with mag_info._unlock():
    for ch in mag_info["chs"]:  # one sensor per pair location, typed as mag so no pair-merging
        ch["coil_type"], ch["unit"] = FIFF.FIFFV_COIL_VV_MAG_T3, FIFF.FIFF_UNIT_T

# (a) |t| topomap
fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
absT = np.abs(T[sig_t]).mean(0)
im, _ = mne.viz.plot_topomap(absT, mag_info, axes=ax, show=False, cmap="Reds", contours=4, mask=best.any(0),
                             mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                                              linewidth=0, markersize=5))
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("|t| (cluster window)", fontsize=LABEL - 2)
cb.ax.tick_params(labelsize=TICK)
ax.text(-0.05, 1.02, "a", transform=ax.transAxes, fontsize=PANEL, fontweight="bold", va="bottom")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"b2a_topo_t.{ext}", dpi=300)
plt.close(fig)

# (b) GFP with sign-flip noise floor
gfp = lambda x: np.sqrt((x ** 2).mean(0))
obs = gfp(D.mean(0))
null = np.array([gfp((D * rng.choice([-1, 1], n)[:, None, None]).mean(0)) for _ in range(5000)])
p95, fwer = np.percentile(null, 95, 0), np.percentile(null.max(1), 95)
above = obs > fwer
print(f"GFP above FWER floor ({fwer:.2f} fT/cm): {t[above].min()*1000:.0f}-{t[above].max()*1000:.0f} ms")
fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
ms = t * 1000
ax.fill_between(ms, 0, 1, where=sig_t, color="0.88", transform=ax.get_xaxis_transform(), linewidth=0)
ax.fill_between(ms, 0, p95, color="#c9d6e8", linewidth=0, label="noise (95%, sign-flip)")
ax.axhline(fwer, color="#2a78d6", linestyle="--", linewidth=1.2, label="FWER threshold")
ax.plot(ms, obs, color="0.2", linewidth=1.8, label="Word − Non-word")
ax.axvline(0, color="0.4", linestyle=":", linewidth=1)
ax.set_xlabel("Time from word onset (ms)", fontsize=LABEL)
ax.set_ylabel("GFP contrast (fT/cm)", fontsize=LABEL)
ax.tick_params(labelsize=TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(obs.max(), fwer) * 1.25)
ax.set_xlim(ms[0], ms[-1])
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.text(-0.25, 1.03, "b", transform=ax.transAxes, fontsize=PANEL, fontweight="bold", va="bottom")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"b2b_gfp_floor.{ext}", dpi=300)
plt.close(fig)

# (c) topomap grid: rows = times, columns = conditions
TIMES = [0.10, 0.35, 0.50]
COND = [("WORD-F1", "Word F1", "#2a78d6"), ("NONWORD-F1", "Non-word F1", "#eb6834"),
        ("WORD-F2", "Word F2", "#1b4f8a"), ("NONWORD-F2", "Non-word F2", "#b8461a")]
grad = {c: e.copy().pick("grad") for c, e in ev.items()}
vmax, maps = 0, {}
for c, _, _ in COND:
    e = grad[c]
    for t0 in TIMES:
        m = (e.times >= t0 - 0.025) & (e.times <= t0 + 0.025)
        d = e.data[:, m].mean(1) * 1e13
        maps[c, t0] = d
        vmax = max(vmax, np.sqrt((d[0::2] ** 2 + d[1::2] ** 2) / 2).max())
fig, axes = plt.subplots(len(TIMES), len(COND), figsize=(8.45, 3.7))
fig.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.01, wspace=0.02, hspace=0.05)
for r, t0 in enumerate(TIMES):
    for k, (c, lab, cc) in enumerate(COND):
        im, _ = mne.viz.plot_topomap(maps[c, t0], grad[c].info, axes=axes[r, k], show=False, cmap="Reds",
                                     vlim=(0, vmax), contours=0, sensors=False)
        if r == 0:
            axes[r, k].set_title(lab, fontsize=13, color=cc, fontweight="bold")
    axes[r, 0].text(-0.08, 0.5, f"{t0 * 1000:.0f} ms", transform=axes[r, 0].transAxes, ha="right", va="center",
                    fontsize=13)
cax = fig.add_axes([0.9, 0.15, 0.015, 0.65])
cb = fig.colorbar(im, cax=cax)
cb.set_label("fT/cm", fontsize=12)
cb.ax.tick_params(labelsize=10)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"b2c_topo_grid.{ext}", dpi=300)
print("grand-average nave:", {c: e.nave for c, e in ev.items()})
