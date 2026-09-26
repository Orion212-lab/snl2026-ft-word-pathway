"""Block 3 (A0): (a) forest of whole-epoch log(Word/Non-word) per pathway ROI; (b) the same effect size
(Cohen's dz) painted on the fsaverage left hemisphere (inflated; lateral + ventral views).
Data: validated block-3 run (dSPM, PTP rejection, equal counts, fixation baseline, N=22), log space."""
import os
import tempfile
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mne.stats import fdr_correction  # noqa: E402
from scipy import stats  # noqa: E402

NPZ, OUT = Path(sys.argv[1]), Path(os.environ.get("POSTER_OUT", "figures_out"))
SD = Path(os.environ["SUBJECTS_DIR"])  # FreeSurfer dir containing fsaverage
FS = SD / "fsaverage"
TMP = Path(tempfile.gettempdir())
W, H = 11.4, 4.4
ORDER = [("vwfa_v1", "V1", ["L_V1_ROI-lh"]), ("vwfa_pit", "Post. IT", ["L_PIT_ROI-lh"]),
         ("vwfa_ffc", "FFC", ["L_FFC_ROI-lh"]), ("vwfa_vwfa", "VWFA", None),
         ("vwfa_pstg", "Post. STG", ["L_STSdp_ROI-lh"]), ("vwfa_pmtg", "Post. MTG", ["L_TPOJ2_ROI-lh"]),
         ("vwfa_astg", "Ant. STG", ["L_STGa_ROI-lh"]), ("vwfa_amtg", "Ant. MTG", ["L_STSva_ROI-lh"]),
         ("L_TGd_ROI-lh", "TP dorsal", ["L_TGd_ROI-lh"]), ("L_TGv_ROI-lh", "TP ventral", ["L_TGv_ROI-lh"])]

# ---------- statistics (identical to the block-3 forest) ----------
z = np.load(NPZ)
rois = list(z["roi_names"])
L = np.log(np.stack([z[c] for c in ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")]))
lexw = (0.5 * ((L[0] - L[2]) + (L[1] - L[3]))).mean(-1)
n = lexw.shape[0]
X = lexw[:, [rois.index(r) for r, _, _ in ORDER]]
tv, pv_ = stats.ttest_1samp(X, 0)
_, pf = fdr_correction(pv_)
dz = X.mean(0) / X.std(0, ddof=1)
print(f"N = {n}")
for (r, lab, _), d, p in zip(ORDER, dz, pf):
    print(f"  {lab:10s} dz = {d:+.2f}  p_FDR = {p:.4f}")

# ---------- surface + labels ----------
coords, faces = nib.freesurfer.read_geometry(str(FS / "surf" / "lh.inflated"))
sulc = nib.freesurfer.read_morph_data(str(FS / "surf" / "lh.sulc"))
labels, _, names = nib.freesurfer.read_annot(str(FS / "label" / "lh.HCPMMP1.annot"))
names = [x.decode() for x in names]
white, _ = nib.freesurfer.read_geometry(str(FS / "surf" / "lh.white"))
mni = mne.vertex_to_mni(np.arange(len(white)), hemis=0, subject="fsaverage", subjects_dir=str(SD))
vwfa_verts = np.where(np.linalg.norm(mni - np.array([-43.0, -54.0, -12.0]), axis=1) <= 8.0)[0]
roi_val = np.full(len(coords), np.nan)
for (r, lab, parcels), d in zip(ORDER, dz):
    if parcels is None:
        roi_val[vwfa_verts] = d
    else:
        for pn in parcels:
            roi_val[labels == names.index(pn.replace("-lh", ""))] = d
norm = Normalize(vmin=0.0, vmax=1.2)
cmap = plt.get_cmap("YlOrRd")
base = np.where(sulc > 0, 0.55, 0.78)
rgb = np.repeat(base[:, None], 3, 1)
m = ~np.isnan(roi_val)
rgb[m] = np.asarray(cmap(norm(np.clip(roi_val[m], 0, None))))[:, :3]
mesh = pv.PolyData(coords, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))
mesh["rgb"] = (rgb * 255).astype(np.uint8)
ctr = coords.mean(0)


def shot(pos, up, fname, zoom=1.35):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1100))
    p.set_background("white")
    p.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.1)
    p.camera_position = [tuple(ctr + np.array(pos)), tuple(ctr), up]
    p.camera.zoom(zoom)
    p.screenshot(str(TMP / fname), transparent_background=False)
    p.close()
    return plt.imread(str(TMP / fname))


lat = shot((-500, 0, 0), (0, 0, 1), "lat.png")
ven = shot((0, 0, -500), (0, 1, 0), "ven.png", zoom=1.05)


def crop(img):
    nonwhite = np.where((img[..., :3] < 0.98).any(-1))
    y0, y1, x0, x1 = nonwhite[0].min(), nonwhite[0].max(), nonwhite[1].min(), nonwhite[1].max()
    return img[y0:y1 + 1, x0:x1 + 1]


lat, ven = crop(lat), np.rot90(crop(ven), k=1)  # ventral: anterior to the left, like the lateral view

# ---------- figure ----------
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 2.6, 0.06], height_ratios=[1.35, 1.0], wspace=0.05, hspace=0.12,
                      left=0.11, right=0.9, top=0.93, bottom=0.14)
ax = fig.add_subplot(gs[:, 0])
for k, ((r, lab, _), d) in enumerate(zip(ORDER, dz)):
    x = X[:, k]
    ci = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    y = len(ORDER) - 1 - k
    col = cmap(norm(max(d, 0))) if pf[k] < 0.05 else (0.6, 0.6, 0.6, 1)
    ax.plot([x.mean() - ci, x.mean() + ci], [y, y], color=col, lw=3)
    ax.plot(x.mean(), y, "o", color=col, ms=9, mec="black" if pf[k] < 0.05 else "none", mew=0.8)
    if pf[k] < 0.05:
        ax.text(0.3, y, "*", fontsize=18, va="center", color="0.2")
ax.axvline(0, color="0.5", ls="--", lw=1)
ax.set_yticks(range(len(ORDER)), [lab for _, lab, _ in ORDER][::-1], fontsize=13)
ax.set_xlabel("log(Word / Non-word)", fontsize=15)
ax.tick_params(axis="x", labelsize=12)
ax.set_xlim(-0.15, 0.36)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.02, 1.01, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", ha="right", va="bottom")
for row, img, title in ((0, lat, "lateral"), (1, ven, "ventral")):
    a = fig.add_subplot(gs[row, 1])
    a.imshow(img)
    a.axis("off")
    a.text(0.0, 0.98, title, transform=a.transAxes, fontsize=14, color="0.3", va="top")
    if row == 0:
        a.text(-0.02, 1.0, "b", transform=a.transAxes, fontsize=22, fontweight="bold", ha="right", va="bottom")
    else:
        a.annotate("", xy=(0.12, -0.1), xytext=(0.88, -0.1), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color="0.3", lw=2))
        a.text(0.5, -0.24, "anterior  ←  posterior", transform=a.transAxes, ha="center", fontsize=13, color="0.3")
cax = fig.add_subplot(gs[:, 2])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("Word > Non-word (dz)", fontsize=13)
cb.ax.tick_params(labelsize=11)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"block3_brain.{ext}", dpi=300)
print("saved block3_brain")
